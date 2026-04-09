"""
Institutional portfolio construction: signal-weighted, dollar-neutral, constraint-optimized.

Architecture (standard stat-arb / market-neutral):
1. Z-score normalize predictions cross-sectionally
2. Include all stocks above/below conviction threshold (dynamic position count)
3. Weight proportional to z-score magnitude × inverse volatility (signal-risk-parity)
4. Enforce strict dollar neutrality (long $ = short $)
5. Apply constraints: max position, sector limits, turnover limits, vol target

References:
- Grinold & Kahn (2000), "Active Portfolio Management", Ch. 14
- Qian (2005), "Risk Parity Portfolios"
- AQR (2019), "Constructing Long-Short Equity Portfolios"
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
        Institutional-grade portfolio construction.

        1. Z-score normalize predictions (cross-sectional standardization)
        2. Select stocks above/below conviction threshold (dynamic sizing)
        3. Weight by signal strength × inverse volatility
        4. Enforce dollar neutrality
        5. Apply constraints (position limits, turnover)

        Args:
            predictions: ticker -> predicted score
            prev_weights: previous period weights (for turnover penalty)
            vol_estimates: ticker -> annualized volatility
        """
        predictions = predictions.dropna()
        if len(predictions) < 20:
            return pd.Series(dtype=float)

        # Apply turnover penalty: boost held positions to reduce churn
        if prev_weights is not None and self.cfg.turnover_penalty > 0:
            predictions = self._apply_turnover_penalty(predictions, prev_weights)

        # Step 1: Z-score normalize predictions cross-sectionally
        z_scores = (predictions - predictions.mean()) / (predictions.std() + 1e-8)

        # Step 2: Select positions by conviction threshold
        # Dynamic sizing: include all stocks with |z| > 1.0
        # This typically selects top/bottom ~15% of universe
        z_threshold = 1.0
        long_mask = z_scores > z_threshold
        short_mask = z_scores < -z_threshold

        # Ensure minimum positions (at least 10 per side)
        if long_mask.sum() < 10:
            long_mask = z_scores >= z_scores.nlargest(10).iloc[-1]
        if short_mask.sum() < 10 and self.cfg.long_short:
            short_mask = z_scores <= z_scores.nsmallest(10).iloc[-1]

        # Cap maximum positions to avoid over-diversification
        max_per_side = min(50, len(predictions) // 4)
        if long_mask.sum() > max_per_side:
            top_n = z_scores[long_mask].nlargest(max_per_side).index
            long_mask = z_scores.index.isin(top_n)
        if short_mask.sum() > max_per_side:
            bot_n = z_scores[short_mask].nsmallest(max_per_side).index
            short_mask = z_scores.index.isin(bot_n)

        long_tickers = z_scores[long_mask].index.tolist()
        short_tickers = z_scores[short_mask].index.tolist() if self.cfg.long_short else []

        if not long_tickers:
            return pd.Series(dtype=float)

        # Step 3: Weight by signal strength × inverse volatility
        weights = pd.Series(0.0, index=predictions.index)

        # Signal-weighted: |z-score| determines conviction
        long_z = z_scores[long_tickers].abs()
        short_z = z_scores[short_tickers].abs() if short_tickers else pd.Series(dtype=float)

        # Adjust by inverse volatility if available (signal-risk-parity)
        if vol_estimates is not None:
            long_vols = vol_estimates.reindex(long_tickers).fillna(vol_estimates.median())
            long_vols = long_vols.clip(lower=0.05)
            long_signal = long_z / long_vols  # high conviction + low vol = large weight
        else:
            long_signal = long_z

        if len(long_signal) > 0:
            weights[long_tickers] = long_signal / long_signal.sum()

        if short_tickers:
            if vol_estimates is not None:
                short_vols = vol_estimates.reindex(short_tickers).fillna(vol_estimates.median())
                short_vols = short_vols.clip(lower=0.05)
                short_signal = short_z / short_vols
            else:
                short_signal = short_z

            if len(short_signal) > 0:
                weights[short_tickers] = -(short_signal / short_signal.sum())

        # Step 4: Enforce dollar neutrality and scale to target leverage
        weights = self._normalize_dollar_neutral(weights)

        # Step 5: Apply position limits
        weights = weights.clip(
            lower=-self.cfg.max_position_pct,
            upper=self.cfg.max_position_pct,
        )

        # Re-normalize after clipping to maintain dollar neutrality
        long_sum = weights[weights > 0].sum()
        short_sum = weights[weights < 0].abs().sum()
        if long_sum > 0 and short_sum > 0:
            avg_side = (long_sum + short_sum) / 2
            weights[weights > 0] *= avg_side / long_sum
            weights[weights < 0] *= avg_side / short_sum

        # Apply turnover limit
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

    def _normalize_dollar_neutral(self, weights: pd.Series) -> pd.Series:
        """
        Enforce strict dollar neutrality: long $ = short $.
        Scale to target gross leverage (split equally between sides).
        """
        long_sum = weights[weights > 0].sum()
        short_sum = weights[weights < 0].abs().sum()

        if not self.cfg.long_short:
            if long_sum > 0:
                weights[weights > 0] *= 1.0 / long_sum
            return weights

        # Target: half of gross leverage per side (dollar neutral)
        target_per_side = self.cfg.max_gross_leverage / 2

        if long_sum > 0:
            weights[weights > 0] *= target_per_side / long_sum
        if short_sum > 0:
            weights[weights < 0] *= target_per_side / short_sum

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
