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
        self.regime_score: float = 0.0  # Current market regime

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

    def update_regime(self, prices: pd.DataFrame):
        """
        Weak regime overlay: 50d/200d MA trend filter.
        
        Positive score = bullish (scale up longs, scale down shorts)
        Negative score = bearish (scale down longs, scale up shorts)
        Max effect: 15% bias either direction.
        """
        mkt = prices.mean(axis=1)
        ma200 = mkt.rolling(200, min_periods=100).mean()
        ma50 = mkt.rolling(50, min_periods=30).mean()
        if ma200.iloc[-1] > 0:
            self.regime_score = float(
                np.clip(ma50.iloc[-1] / ma200.iloc[-1] - 1, -0.1, 0.1)
            )
        else:
            self.regime_score = 0.0

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

    def apply_risk_scaling(self, weights, portfolio_returns, sector_map,
                           n_long: int = 15, n_short: int = 7):
        """
        Full risk pipeline, used by backtest AND live trading:
        1. Sector neutralization
        2. Factor neutralization
        3. Vol targeting
        4. Drawdown control
        5. Re-clip to n_long/n_short (prevents position drift)
        6. Regime overlay (after clip so it gets full effect)
        """
        weights = self.apply_sector_neutrality(weights, sector_map)
        weights = self.neutralize_factors(weights)
        weights *= self.compute_vol_scale(portfolio_returns)
        weights *= self.compute_drawdown_scale()

        # Re-clip positions (factor/sector neutralization can expand beyond target)
        if len(weights[weights > 0]) > n_long:
            long_w = weights[weights > 0].nlargest(n_long)
        else:
            long_w = weights[weights > 0]

        if n_short > 0 and len(weights[weights < 0]) > n_short:
            short_w = weights[weights < 0].nsmallest(n_short)
        else:
            short_w = weights[weights < 0]

        weights = pd.concat([long_w, short_w])
        weights = weights[weights.abs() > 0.005]

        # Regime overlay (after clip — scales weights, doesn't add positions)
        weights = self.apply_regime_overlay(weights)

        return weights

    def reset(self):
        self.cum_returns = []
        self.peak_equity = 1.0
        self.factor_exposures = None
        self.factor_covariance = None
        self.specific_variance = None
        self.regime_score = 0.0
