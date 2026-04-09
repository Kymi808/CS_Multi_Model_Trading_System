"""
Barra-style statistical factor risk model.

Estimates covariance via: Σ = B × F × B' + D
where B=factor exposures, F=factor covariance, D=idiosyncratic variance.

Factors: Market, Size, Value, Momentum, Volatility, Quality
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from config import RiskConfig

logger = logging.getLogger(__name__)


class FactorRiskModel:

    FACTOR_NAMES = ["market", "size", "value", "momentum", "volatility", "quality"]

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.factor_exposures: Optional[pd.DataFrame] = None
        self.factor_covariance: Optional[pd.DataFrame] = None
        self.specific_variance: Optional[pd.Series] = None
        self.cum_returns: list = []
        self.peak_equity: float = 1.0
        self.regime_score: float = 0.0
        self.regime_state = None  # HMM regime state (full probabilities)

        # Portfolio limits (set via set_portfolio_limits from PortfolioConfig)
        self._max_gross_leverage: float = 1.6
        self._max_net_leverage: float = 0.10

        # Initialize HMM regime detector
        self.hmm_detector = None
        hmm_enabled = getattr(cfg, "hmm_enabled", True)
        if hmm_enabled:
            try:
                from hmm_regime import HMMRegimeDetector
                self.hmm_detector = HMMRegimeDetector(
                    n_states=getattr(cfg, "hmm_n_states", 3),
                    refit_every=getattr(cfg, "hmm_refit_every", 63),
                    min_train_days=getattr(cfg, "hmm_min_train_days", 504),
                )
            except ImportError:
                pass  # hmmlearn not installed

    def estimate(
        self, prices: pd.DataFrame, fundamentals: Dict[str, dict],
        date: pd.Timestamp, lookback: int = 252,
    ):
        tickers = list(prices.columns)
        end_idx = prices.index.get_loc(date) if date in prices.index else len(prices) - 1
        start_idx = max(0, end_idx - lookback)
        price_window = prices.iloc[start_idx:end_idx + 1]
        if len(price_window) < 63:
            return

        returns = np.log(price_window / price_window.shift(1)).dropna()
        exposures = self._compute_exposures(returns, fundamentals, tickers)
        self.factor_exposures = exposures

        factor_returns = self._estimate_factor_returns(returns, exposures)
        if len(factor_returns) > 20:
            halflife = 63
            w = np.exp(-np.log(2) * np.arange(len(factor_returns))[::-1] / halflife)
            w /= w.sum()
            centered = factor_returns - factor_returns.mean()
            self.factor_covariance = pd.DataFrame(
                (centered.values.T * w) @ centered.values * 252,
                index=self.FACTOR_NAMES, columns=self.FACTOR_NAMES,
            )
        else:
            self.factor_covariance = factor_returns.cov() * 252

        self.specific_variance = (returns ** 2).mean() * 252

    def _compute_exposures(self, returns, fundamentals, tickers):
        exposures = pd.DataFrame(0.0, index=tickers, columns=self.FACTOR_NAMES)
        mkt_ret = returns.mean(axis=1)
        for t in tickers:
            if t in returns.columns:
                cov = returns[t].cov(mkt_ret)
                var = mkt_ret.var()
                exposures.loc[t, "market"] = cov / var if var > 0 else 1.0
            mcap = fundamentals.get(t, {}).get("marketCap")
            if mcap and mcap > 0:
                exposures.loc[t, "size"] = np.log(mcap)
            pe = fundamentals.get(t, {}).get("trailingPE")
            if pe and pe > 0:
                exposures.loc[t, "value"] = 1.0 / pe
            roe = fundamentals.get(t, {}).get("returnOnEquity")
            if roe is not None:
                exposures.loc[t, "quality"] = roe
        if len(returns) > 21:
            mom = returns.iloc[:-21].sum()
            for t in tickers:
                if t in mom.index:
                    exposures.loc[t, "momentum"] = mom[t]
        vol = returns.std() * np.sqrt(252)
        for t in tickers:
            if t in vol.index:
                exposures.loc[t, "volatility"] = vol[t]
        # Standardize
        for col in self.FACTOR_NAMES:
            vals = exposures[col]
            mean, std = vals.mean(), vals.std()
            if std > 0:
                exposures[col] = (vals - mean) / std
        return exposures.fillna(0)

    def _estimate_factor_returns(self, returns, exposures):
        common = returns.columns.intersection(exposures.index)
        B = exposures.loc[common].values
        R = returns[common].values
        try:
            BtB_inv = np.linalg.pinv(B.T @ B)
            fr = R @ B @ BtB_inv.T
            return pd.DataFrame(fr, index=returns.index, columns=self.FACTOR_NAMES)
        except np.linalg.LinAlgError:
            return pd.DataFrame(columns=self.FACTOR_NAMES)

    def get_portfolio_risk(self, weights: pd.Series) -> dict:
        if self.factor_exposures is None or self.factor_covariance is None:
            return {"total_vol": np.nan}
        tickers = weights.index.intersection(self.factor_exposures.index)
        if len(tickers) == 0:
            return {"total_vol": np.nan}
        w = weights.reindex(tickers, fill_value=0).values
        B = self.factor_exposures.loc[tickers].values
        F = self.factor_covariance.values
        port_exp = w @ B
        factor_var = port_exp @ F @ port_exp.T
        spec = 0
        if self.specific_variance is not None:
            sv = self.specific_variance.reindex(tickers, fill_value=0).values
            spec = np.sum(w ** 2 * sv)
        return {
            "total_vol": float(np.sqrt(max(factor_var + spec, 0))),
            "factor_vol": float(np.sqrt(max(factor_var, 0))),
            "specific_vol": float(np.sqrt(max(spec, 0))),
            "factor_exposures": {f: float(port_exp[i]) for i, f in enumerate(self.FACTOR_NAMES)},
        }

    def neutralize_factors(
        self, weights: pd.Series, max_exposure: float = 0.3,
    ) -> pd.Series:
        """Iteratively reduce factor exposures toward zero."""
        if self.factor_exposures is None:
            return weights
        common = weights.index.intersection(self.factor_exposures.index)
        if len(common) < 5:
            return weights
        w = weights.reindex(common, fill_value=0).copy()
        B = self.factor_exposures.loc[common]

        for _ in range(5):
            port_exp = w.values @ B.values
            for i, factor in enumerate(self.FACTOR_NAMES):
                if abs(port_exp[i]) > max_exposure:
                    excess = port_exp[i]
                    loadings = B[factor]
                    adjustment = -excess * loadings / ((loadings ** 2).sum() + 1e-8) * 0.3
                    w += adjustment
            w = w.clip(-0.10, 0.10)

        return w[w.abs() > 0.005]

    # ---- Risk management interface (vol targeting, drawdown, sector) ----

    def apply_sector_neutrality(self, weights, sector_map):
        if not self.cfg.sector_neutral or not sector_map:
            return weights
        sectors = pd.Series({t: sector_map.get(t, "Unknown") for t in weights.index})
        adjusted = weights.copy()
        for sector in sectors.unique():
            mask = sectors == sector
            net = weights[mask].sum()
            if abs(net) > self.cfg.max_sector_net_pct:
                n = mask.sum()
                if n > 0:
                    adjusted[mask] -= (net - np.sign(net) * self.cfg.max_sector_net_pct) / n
        return adjusted

    def compute_vol_scale(self, portfolio_returns):
        if len(portfolio_returns) < self.cfg.vol_lookback:
            return 1.0
        rv = portfolio_returns.iloc[-self.cfg.vol_lookback:].std() * np.sqrt(252)
        if rv <= 0:
            return 1.0
        return float(np.clip(self.cfg.target_annual_vol / rv, self.cfg.vol_scale_floor, self.cfg.vol_scale_cap))

    def compute_drawdown_scale(self):
        if not self.cum_returns:
            return 1.0
        cum = np.prod([1 + r for r in self.cum_returns])
        self.peak_equity = max(self.peak_equity, cum)
        dd = cum / self.peak_equity - 1
        if dd < self.cfg.max_drawdown_threshold:
            severity = dd / self.cfg.max_drawdown_threshold
            return max(self.cfg.drawdown_scale_factor, 1.0 - (severity - 1.0) * 0.5)
        return 1.0

    def update(self, daily_return):
        self.cum_returns.append(daily_return)

    def update_regime(self, prices: pd.DataFrame, cross_asset_prices: pd.DataFrame = None):
        """
        Regime detection with HMM (primary) and MA crossover (fallback).

        HMM: 3-state Gaussian model trained on macro observables.
        Provides probability distribution over bull/sideways/bear.

        MA fallback: multi-speed blend (5/20, 20/50, 50/200) when HMM
        is unavailable or undertrained.
        """
        # Try HMM first (institutional standard)
        if self.hmm_detector is not None and cross_asset_prices is not None:
            obs = self.hmm_detector.prepare_observations(cross_asset_prices)
            if len(obs) >= self.hmm_detector.min_train_days:
                if self.hmm_detector.should_refit(len(obs)):
                    self.hmm_detector.fit(obs)
                if self.hmm_detector.model is not None:
                    state = self.hmm_detector.predict(obs)
                    self.regime_score = state.regime_score
                    self.regime_state = state
                    return

        # Fallback: multi-speed MA crossover
        self._update_regime_ma(prices)

    def _update_regime_ma(self, prices: pd.DataFrame):
        """MA-based regime detection fallback."""
        mkt = prices.mean(axis=1)

        def _ma_score(fast_period, slow_period, min_fast, min_slow):
            ma_fast = mkt.rolling(fast_period, min_periods=min_fast).mean()
            ma_slow = mkt.rolling(slow_period, min_periods=min_slow).mean()
            if ma_slow.iloc[-1] > 0:
                return float(np.clip(ma_fast.iloc[-1] / ma_slow.iloc[-1] - 1, -0.1, 0.1))
            return 0.0

        fast_score = _ma_score(5, 20, 5, 15)
        medium_score = _ma_score(20, 50, 15, 30)
        slow_score = _ma_score(50, 200, 30, 100)

        self.regime_score = float(np.clip(
            fast_score * 0.2 + medium_score * 0.3 + slow_score * 0.5,
            -0.1, 0.1,
        ))

    def apply_regime_overlay(self, weights: pd.Series) -> pd.Series:
        """Apply weak regime bias to portfolio weights."""
        if abs(self.regime_score) < 0.01:
            return weights

        bias = min(abs(self.regime_score) * 2, 0.15)  # Max 15%

        if self.regime_score > 0:
            # Bullish: scale up longs, scale down shorts
            weights[weights > 0] *= (1 + bias)
            weights[weights < 0] *= (1 - bias)
        else:
            # Bearish: scale down longs, scale up shorts
            weights[weights > 0] *= (1 - bias)
            weights[weights < 0] *= (1 + bias)

        return weights

    def compute_tail_risk_scale(self, portfolio_returns) -> float:
        """
        Tail risk protection: detect market stress via recent return distribution.

        Checks for:
        - Gap-down days (daily return < -3%)
        - Elevated recent volatility (5d vol > 2x 63d vol)
        - Consecutive down days (3+ in a row)

        Returns a scaling factor (0.3 to 1.0) to reduce exposure during stress.
        """
        if not hasattr(portfolio_returns, '__len__') or len(portfolio_returns) < 10:
            return 1.0

        recent = portfolio_returns[-5:]  # last 5 days
        lookback = portfolio_returns[-63:] if len(portfolio_returns) >= 63 else portfolio_returns

        # Check for recent gap-down (any day < -3%)
        if any(r < -0.03 for r in recent):
            return 0.5  # cut exposure by half after a gap-down

        # Check for vol spike (5d realized vol > 2x 63d realized vol)
        if len(recent) >= 5 and len(lookback) >= 20:
            recent_vol = float(np.std(recent)) * np.sqrt(252)
            long_vol = float(np.std(lookback)) * np.sqrt(252)
            if long_vol > 0 and recent_vol / long_vol > 2.0:
                return 0.6  # reduce by 40% during vol spikes

        # Check for 3+ consecutive down days
        if len(recent) >= 3 and all(r < 0 for r in recent[-3:]):
            return 0.7  # reduce by 30% during persistent selling

        return 1.0

    def apply_risk_scaling(self, weights, portfolio_returns, sector_map,
                           n_long: int = 50, n_short: int = 50):
        """
        Institutional risk pipeline:
        1. Sector neutralization
        2. Factor neutralization
        3. Vol targeting
        4. Drawdown control
        5. Tail risk protection
        6. Gross leverage hard cap
        7. Dollar neutrality enforcement
        8. Regime overlay
        """
        weights = self.apply_sector_neutrality(weights, sector_map)
        weights = self.neutralize_factors(weights)
        weights *= self.compute_vol_scale(portfolio_returns)
        weights *= self.compute_drawdown_scale()
        weights *= self.compute_tail_risk_scale(portfolio_returns)

        # Remove tiny positions
        weights = weights[weights.abs() > 0.003]

        # Hard cap on gross leverage (prevents leverage explosion)
        # Uses PortfolioConfig values passed via set_portfolio_limits()
        gross = weights.abs().sum()
        max_gross = self._max_gross_leverage
        if gross > max_gross:
            weights *= max_gross / gross

        # Enforce dollar neutrality + net exposure cap
        # Risk adjustments (vol, drawdown, tail) can break the balance
        long_sum = weights[weights > 0].sum()
        short_sum = weights[weights < 0].abs().sum()
        if long_sum > 0 and short_sum > 0:
            # Scale both sides to equal dollar value (strict dollar neutral)
            target_per_side = min(long_sum, short_sum)
            target_per_side = min(target_per_side, max_gross / 2)
            weights[weights > 0] *= target_per_side / long_sum
            weights[weights < 0] *= target_per_side / short_sum

            # Hard cap on net exposure
            net = weights.sum()
            max_net = self._max_net_leverage
            if abs(net) > max_net:
                # Reduce the larger side to bring net within limit
                if net > 0:
                    excess = net - max_net
                    weights[weights > 0] *= (long_sum - excess) / long_sum
                else:
                    excess = abs(net) - max_net
                    weights[weights < 0] *= (short_sum - excess) / short_sum

        # Regime overlay (after neutrality — scales weights, doesn't add positions)
        weights = self.apply_regime_overlay(weights)

        return weights

    def set_portfolio_limits(self, max_gross: float, max_net: float):
        """Set portfolio limits from PortfolioConfig (called by model_comparison)."""
        self._max_gross_leverage = max_gross
        self._max_net_leverage = max_net

    def reset(self):
        self.cum_returns = []
        self.peak_equity = 1.0
        self.factor_exposures = None
        self.factor_covariance = None
        self.specific_variance = None
        self.regime_score = 0.0
