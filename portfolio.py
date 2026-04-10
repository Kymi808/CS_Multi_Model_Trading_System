"""
Portfolio construction: signal-weighted, risk-parity, dollar-neutral.

Design principles (informed by OOS prediction audit):
- Predictions have ranking half-life of 16.6 days → low natural turnover
- Top quintile beats bottom by ~8 bps/day → signal is in the tails
- 547 stocks/day with tight score distribution → use ranks, not raw scores
- Risk controls should PRESERVE alpha, not destroy it

Key changes from prior version:
- Removed factor neutralization (was destroying alpha via 5 additive iterations)
- Removed vol/drawdown/tail risk scaling (procyclical, cut exposure when alpha highest)
- Increased breadth to 20L/20S (Grinold: Sharpe scales with √breadth)
- Sector cap relaxed to 15% (was 3%, too tight)
- Turnover cap relaxed to 60% (was 40%, caused stale positions)
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
        sector_map: Optional[Dict[str, str]] = None,
    ) -> pd.Series:
        """
        Build target portfolio from alpha model predictions.

        Pipeline:
        1. Cross-sectional rank → select top/bottom N
        2. Signal-weight by rank distance from median × inverse vol
        3. Normalize to target leverage per side
        4. Apply sector cap
        5. Apply position cap
        6. Enforce dollar neutrality
        7. Apply turnover constraint
        """
        predictions = predictions.dropna()
        if len(predictions) < self.cfg.max_positions_long + self.cfg.max_positions_short:
            return pd.Series(dtype=float)

        n_long = self.cfg.max_positions_long
        n_short = self.cfg.max_positions_short

        # 1. Cross-sectional rank (0 to 1)
        ranks = predictions.rank(pct=True)

        # Dispersion-based leverage: when model spreads predictions wider,
        # signal is stronger (correlation 0.48 between dispersion and alpha).
        # Scale leverage between 0.5x and 1.0x of target based on prediction std.
        pred_std = predictions.std()
        dispersion_scale = np.clip(pred_std / 0.02, 0.5, 1.0)  # 0.02 is typical std

        # Hysteresis: stocks enter at rank ≤ N, hold until rank > exit_N
        # This reduces turnover by 60%+ while preserving alpha (audit showed +0.37 Sharpe)
        exit_n_long = int(n_long * self.cfg.hysteresis_exit_mult)
        exit_n_short = int(n_short * self.cfg.hysteresis_exit_mult)

        prev_long = set()
        prev_short = set()
        if prev_weights is not None and not prev_weights.empty:
            prev_long = set(prev_weights[prev_weights > 0].index)
            prev_short = set(prev_weights[prev_weights < 0].index)

        # Long selection with hysteresis
        ranks_desc = ranks.sort_values(ascending=False)
        long_tickers = []
        # Keep existing positions that are still in the hold zone (top exit_N)
        hold_zone_long = set(ranks_desc.head(exit_n_long).index)
        for t in prev_long:
            if t in hold_zone_long and len(long_tickers) < n_long:
                long_tickers.append(t)
        # Fill remaining slots with new top-N entries
        for t in ranks_desc.index:
            if len(long_tickers) >= n_long:
                break
            if t not in long_tickers:
                long_tickers.append(t)

        # Short selection with hysteresis
        short_tickers = []
        if self.cfg.long_short:
            ranks_asc = ranks.sort_values(ascending=True)
            hold_zone_short = set(ranks_asc.head(exit_n_short).index)
            for t in prev_short:
                if t in hold_zone_short and len(short_tickers) < n_short:
                    short_tickers.append(t)
            for t in ranks_asc.index:
                if len(short_tickers) >= n_short:
                    break
                if t not in short_tickers:
                    short_tickers.append(t)

        if not long_tickers:
            return pd.Series(dtype=float)

        # 2. Signal-weighted × inverse vol
        weights = pd.Series(0.0, index=predictions.index)

        # Long side: weight by distance from median rank (stronger signal = more weight)
        long_signal = ranks[long_tickers] - 0.5  # distance from median
        if vol_estimates is not None:
            long_vols = vol_estimates.reindex(long_tickers).fillna(vol_estimates.median()).clip(lower=0.05)
            long_signal = long_signal / long_vols  # risk-parity adjustment
        long_signal = long_signal.clip(lower=0)
        if long_signal.sum() > 0:
            weights[long_tickers] = long_signal / long_signal.sum()

        # Short side: weight by distance below median
        if short_tickers:
            short_signal = 0.5 - ranks[short_tickers]  # distance below median
            if vol_estimates is not None:
                short_vols = vol_estimates.reindex(short_tickers).fillna(vol_estimates.median()).clip(lower=0.05)
                short_signal = short_signal / short_vols
            short_signal = short_signal.clip(lower=0)
            if short_signal.sum() > 0:
                weights[short_tickers] = -(short_signal / short_signal.sum())

        # 3. Scale to target leverage, adjusted by signal dispersion
        target_per_side = self.cfg.max_gross_leverage / 2 * dispersion_scale
        long_sum = weights[weights > 0].sum()
        short_sum = weights[weights < 0].abs().sum()
        if long_sum > 0:
            weights[weights > 0] *= target_per_side / long_sum
        if short_sum > 0:
            weights[weights < 0] *= target_per_side / short_sum

        # 4. Sector cap (max % of gross in any single sector)
        if sector_map:
            weights = self._apply_sector_cap(weights, sector_map)

        # 5. Position cap
        weights = weights.clip(-self.cfg.max_position_pct, self.cfg.max_position_pct)

        # 6. Re-enforce dollar neutrality after clipping
        long_sum = weights[weights > 0].sum()
        short_sum = weights[weights < 0].abs().sum()
        if long_sum > 0 and short_sum > 0:
            target = min(long_sum, short_sum, target_per_side)
            weights[weights > 0] *= target / long_sum
            weights[weights < 0] *= target / short_sum

        # 7. Turnover constraint
        if prev_weights is not None:
            weights = self._apply_turnover_limit(weights, prev_weights)

        # Remove dust
        weights = weights[weights.abs() > 0.002]

        return weights

    def _apply_sector_cap(
        self, weights: pd.Series, sector_map: Dict[str, str],
        max_sector_gross_pct: float = 0.15,
    ) -> pd.Series:
        """Cap gross exposure in any single sector to max_sector_gross_pct of total gross."""
        sectors = pd.Series({t: sector_map.get(t, "Unknown") for t in weights.index})
        gross = weights.abs().sum()
        if gross == 0:
            return weights

        adjusted = weights.copy()
        for sector in sectors.unique():
            if sector == "Unknown":
                continue  # Don't cap unmapped stocks as a single "sector"
            mask = sectors == sector
            sector_gross = weights[mask].abs().sum()
            if sector_gross > max_sector_gross_pct * gross:
                scale = (max_sector_gross_pct * gross) / sector_gross
                adjusted[mask] *= scale

        return adjusted

    def _apply_turnover_limit(
        self, target: pd.Series, prev: pd.Series,
    ) -> pd.Series:
        """Limit daily turnover to prevent excessive trading."""
        if prev.empty or prev.abs().sum() < 0.01:
            return target[target.abs() > 0.002]

        all_tickers = target.index.union(prev.index)
        target = target.reindex(all_tickers, fill_value=0)
        prev = prev.reindex(all_tickers, fill_value=0)
        trades = target - prev
        turnover = trades.abs().sum()

        if turnover > self.cfg.max_daily_turnover:
            scale = self.cfg.max_daily_turnover / turnover
            target = prev + trades * scale

        return target[target.abs() > 0.002]


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

        # Transaction costs: fixed per-unit cost (no vol scaling — simpler, more honest)
        base_cost_bps = cfg.commission_bps + cfg.slippage_bps + cfg.spread_bps
        tc = turnover * base_cost_bps / 10000

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
