"""
Portfolio construction: concentrated, sector-neutral, risk-parity weighted.

Key improvements:
- Risk-parity weighting (inverse vol) instead of score-weighted
- Turnover penalty to reduce unnecessary trading
- Integration with RiskModel for sector neutrality and vol targeting
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from config import PortfolioConfig

logger = logging.getLogger(__name__)


class PortfolioConstructor:

    def __init__(self, cfg: PortfolioConfig):
        self.cfg = cfg

    def construct_portfolio(
        self,
        predictions: pd.Series,
        date: pd.Timestamp,
        prev_weights: Optional[pd.Series] = None,
        vol_estimates: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Build target portfolio from model predictions.

        Args:
            predictions: ticker -> predicted score
            prev_weights: previous period weights (for turnover penalty)
            vol_estimates: ticker -> annualized volatility (for risk parity)
        """
        predictions = predictions.dropna()
        if len(predictions) < 20:
            return pd.Series(dtype=float)

        # Apply turnover penalty
        if prev_weights is not None and self.cfg.turnover_penalty > 0:
            predictions = self._apply_turnover_penalty(predictions, prev_weights)

        # Z-score normalize predictions cross-sectionally
        z_scores = (predictions - predictions.mean()) / (predictions.std() + 1e-8)

        # Dynamic position selection: top/bottom 8% by percentile
        # Robust to any prediction distribution (LightGBM, TST, CrossMamba)
        n_universe = len(predictions)
        n_per_side = max(15, min(40, int(n_universe * 0.08)))

        sorted_z = z_scores.sort_values(ascending=False)
        long_tickers = sorted_z.head(n_per_side).index.tolist()
        short_tickers = sorted_z.tail(n_per_side).index.tolist() if self.cfg.long_short else []

        # Signal-weighted: |z-score| × inverse volatility
        weights = pd.Series(0.0, index=predictions.index)

        long_z = z_scores[long_tickers].abs()
        if vol_estimates is not None:
            long_vols = vol_estimates.reindex(long_tickers).fillna(vol_estimates.median()).clip(lower=0.05)
            long_signal = long_z / long_vols
        else:
            long_signal = long_z
        if len(long_signal) > 0:
            weights[long_tickers] = long_signal / long_signal.sum()

        if short_tickers:
            short_z = z_scores[short_tickers].abs()
            if vol_estimates is not None:
                short_vols = vol_estimates.reindex(short_tickers).fillna(vol_estimates.median()).clip(lower=0.05)
                short_signal = short_z / short_vols
            else:
                short_signal = short_z
            if len(short_signal) > 0:
                weights[short_tickers] = -(short_signal / short_signal.sum())

        # Dollar-neutral leverage: equal dollars per side, capped at max_gross/2
        target_per_side = self.cfg.max_gross_leverage / 2
        long_sum = weights[weights > 0].sum()
        short_sum = weights[weights < 0].abs().sum()
        if long_sum > 0:
            weights[weights > 0] *= target_per_side / long_sum
        if short_sum > 0:
            weights[weights < 0] *= target_per_side / short_sum

        # Position limits
        weights = weights.clip(-self.cfg.max_position_pct, self.cfg.max_position_pct)

        # Re-balance after clipping to maintain dollar neutrality
        long_sum = weights[weights > 0].sum()
        short_sum = weights[weights < 0].abs().sum()
        if long_sum > 0 and short_sum > 0:
            avg_side = min(long_sum, short_sum, target_per_side)
            weights[weights > 0] *= avg_side / long_sum
            weights[weights < 0] *= avg_side / short_sum

        # Turnover limit
        if prev_weights is not None:
            weights = self._apply_turnover_limit(weights, prev_weights)

        # Remove tiny positions
        weights = weights[weights.abs() > 0.003]

        return weights

    def _apply_turnover_penalty(
        self, predictions: pd.Series, prev_weights: pd.Series,
    ) -> pd.Series:
        """
        Boost scores for currently held positions to reduce turnover.
        This is a simple but effective heuristic.
        """
        penalty = self.cfg.turnover_penalty
        held = prev_weights[prev_weights.abs() > 0.005].index

        adjusted = predictions.copy()
        for ticker in held:
            if ticker in adjusted.index:
                # Boost score of held positions proportional to current weight
                sign = np.sign(prev_weights.get(ticker, 0))
                adjusted[ticker] += sign * penalty * 10  # Scale penalty

        return adjusted

    def _equal_weights(
        self, long_tickers: list, short_tickers: list,
    ) -> pd.Series:
        weights = pd.Series(0.0, index=long_tickers + short_tickers)
        if long_tickers:
            weights[long_tickers] = 1.0 / len(long_tickers)
        if short_tickers:
            weights[short_tickers] = -1.0 / len(short_tickers)
        return weights

    def _score_weights(
        self, long_tickers: list, short_tickers: list, predictions: pd.Series,
    ) -> pd.Series:
        weights = pd.Series(0.0, index=predictions.index)
        if long_tickers:
            lp = predictions[long_tickers]
            lp = lp - lp.min() + 1e-6
            weights[long_tickers] = lp / lp.sum()
        if short_tickers:
            sp = -predictions[short_tickers]
            sp = sp - sp.min() + 1e-6
            weights[short_tickers] = -(sp / sp.sum())
        return weights

    def _risk_parity_weights(
        self, long_tickers: list, short_tickers: list,
        vol_estimates: pd.Series,
    ) -> pd.Series:
        """
        Risk-parity: weight inversely proportional to volatility.
        Each position contributes equal risk to the portfolio.
        """
        weights = pd.Series(0.0, index=list(set(long_tickers + short_tickers)))

        if long_tickers:
            vols = vol_estimates.reindex(long_tickers).fillna(vol_estimates.median())
            vols = vols.clip(lower=0.05)  # Floor at 5% vol
            inv_vol = 1.0 / vols
            weights[long_tickers] = inv_vol / inv_vol.sum()

        if short_tickers:
            vols = vol_estimates.reindex(short_tickers).fillna(vol_estimates.median())
            vols = vols.clip(lower=0.05)
            inv_vol = 1.0 / vols
            weights[short_tickers] = -(inv_vol / inv_vol.sum())

        return weights

    def _normalize_leverage(self, weights: pd.Series) -> pd.Series:
        long_sum = weights[weights > 0].sum()
        short_sum = weights[weights < 0].abs().sum()

        if not self.cfg.long_short:
            if long_sum > 0:
                weights[weights > 0] *= 1.0 / long_sum
            return weights

        n_long = self.cfg.max_positions_long
        n_short = self.cfg.max_positions_short
        total = n_long + n_short

        if total > 0:
            # Allocate gross leverage proportional to number of positions
            # e.g., 15L/5S → 75% long, 25% short
            long_frac = n_long / total
            short_frac = n_short / total
            target_long = self.cfg.max_gross_leverage * long_frac
            target_short = self.cfg.max_gross_leverage * short_frac
        else:
            target_long = self.cfg.max_gross_leverage / 2
            target_short = self.cfg.max_gross_leverage / 2

        if long_sum > 0:
            weights[weights > 0] *= target_long / long_sum
        if short_sum > 0:
            weights[weights < 0] *= target_short / short_sum

        return weights

    def _apply_turnover_limit(
        self, target: pd.Series, prev: pd.Series,
    ) -> pd.Series:
        # Skip turnover limit for initial portfolio (no previous positions)
        if prev.empty or prev.abs().sum() < 0.01:
            return target[target.abs() > 0.005]

        all_tickers = target.index.union(prev.index)
        target = target.reindex(all_tickers, fill_value=0)
        prev = prev.reindex(all_tickers, fill_value=0)
        trades = target - prev
        turnover = trades.abs().sum()

        if turnover > self.cfg.max_daily_turnover:
            scale = self.cfg.max_daily_turnover / turnover
            target = prev + trades * scale
        return target[target.abs() > 0.005]


def compute_portfolio_returns(
    weights_history: Dict, prices: pd.DataFrame, cfg: PortfolioConfig,
) -> pd.DataFrame:
    dates = sorted(weights_history.keys())
    results = []
    prev_weights = pd.Series(dtype=float)

    for date in dates:
        target_weights = weights_history[date]
        if date not in prices.index:
            continue
        if date == dates[0]:
            prev_weights = target_weights
            continue

        prev_idx = prices.index.get_loc(date) - 1
        if prev_idx < 0:
            continue
        prev_date = prices.index[prev_idx]
        tickers = prev_weights.index.intersection(prices.columns)
        tickers = tickers[prices.loc[date, tickers].notna() & prices.loc[prev_date, tickers].notna()]

        if len(tickers) == 0:
            continue

        stock_rets = prices.loc[date, tickers] / prices.loc[prev_date, tickers] - 1
        port_ret = (prev_weights.reindex(tickers, fill_value=0) * stock_rets).sum()

        trades = target_weights.reindex(tickers, fill_value=0) - prev_weights.reindex(tickers, fill_value=0)
        turnover = trades.abs().sum()

        # Volatility-dependent slippage: higher vol = wider spreads + more slippage
        # Base cost from config, scaled by recent realized vol vs long-run avg
        base_cost_bps = cfg.commission_bps + cfg.slippage_bps + cfg.spread_bps
        vol_scale = 1.0
        if len(results) >= 20:
            recent_rets = [r["gross_return"] for r in results[-5:]]
            long_rets = [r["gross_return"] for r in results[-63:]]
            recent_vol = np.std(recent_rets) if len(recent_rets) > 1 else 0
            long_vol = np.std(long_rets) if len(long_rets) > 1 else recent_vol
            if long_vol > 0:
                vol_scale = max(1.0, min(3.0, recent_vol / long_vol))
        tc = turnover * base_cost_bps * vol_scale / 10000

        results.append({
            "date": date,
            "gross_return": port_ret,
            "tc_cost": tc,
            "net_return": port_ret - tc,
            "turnover": turnover,
            "n_long": (prev_weights > 0).sum(),
            "n_short": (prev_weights < 0).sum(),
            "gross_exposure": prev_weights.abs().sum(),
            "net_exposure": prev_weights.sum(),
        })
        prev_weights = target_weights

    return pd.DataFrame(results).set_index("date") if results else pd.DataFrame()


def compute_performance_metrics(returns: pd.Series, annual_factor: int = 252) -> dict:
    if len(returns) < 2:
        return {}
    total = (1 + returns).prod() - 1
    n_years = len(returns) / annual_factor
    ann_ret = (1 + total) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = returns.std() * np.sqrt(annual_factor)
    sharpe = ann_ret / (ann_vol + 1e-8)
    cum = (1 + returns).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / (abs(max_dd) + 1e-8)
    downside = returns[returns < 0].std() * np.sqrt(annual_factor)
    sortino = ann_ret / (downside + 1e-8)
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    return {
        "total_return": total,
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": (returns > 0).mean(),
        "profit_factor": gains / (losses + 1e-8),
        "n_days": len(returns),
        "n_years": n_years,
    }


# Need numpy import at module level for compute_performance_metrics
import numpy as np
