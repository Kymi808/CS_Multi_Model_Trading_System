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
from typing import Dict, Optional
from config import PortfolioConfig

logger = logging.getLogger(__name__)


class PortfolioConstructor:

    def __init__(self, cfg: PortfolioConfig):
        self.cfg = cfg
        # State for stop-loss tracking (persistent across construct_portfolio calls)
        self._short_entry_price: Dict[str, float] = {}  # ticker -> entry price
        self._short_blacklist: Dict[str, pd.Timestamp] = {}  # ticker -> blacklist expiry date
        # Drawdown circuit breaker state (updated externally via update_cum_return)
        self._cum_return: float = 1.0
        self._peak_return: float = 1.0

    def update_cum_return(self, daily_return: float) -> None:
        """
        Advance the cumulative-return tracker after a realized daily return.
        Called by the backtester (or live runner) at the end of each day.
        The next construct_portfolio call will see the resulting drawdown state
        and apply the DD circuit breaker if needed.
        """
        self._cum_return *= (1 + daily_return)
        if self._cum_return > self._peak_return:
            self._peak_return = self._cum_return

    def get_current_dd(self) -> float:
        """Return current drawdown (0 to -1)."""
        if self._peak_return <= 0:
            return 0.0
        return self._cum_return / self._peak_return - 1

    def construct_portfolio(
        self,
        predictions: pd.Series,
        date: pd.Timestamp,
        prev_weights: Optional[pd.Series] = None,
        vol_estimates: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
        momentum_6m: Optional[pd.Series] = None,
        dist_from_52w_high: Optional[pd.Series] = None,
        current_prices: Optional[pd.Series] = None,
        vix_current: Optional[float] = None,
        mcap_current: Optional[pd.Series] = None,
        earnings_yield_current: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Build target portfolio from alpha model predictions.

        Pipeline:
        1. Apply SHORT-SIDE filters (momentum veto, 52w high proximity)
           to avoid shorting high-momentum squeeze candidates
        2. Cross-sectional rank → select top/bottom N
        3. Signal-weight by rank distance from median × inverse vol
        4. Normalize to target leverage per side
        5. Apply sector cap, position cap, dollar neutrality, turnover cap
        """
        predictions = predictions.dropna()
        if len(predictions) < self.cfg.max_positions_long + self.cfg.max_positions_short:
            return pd.Series(dtype=float)

        n_long = self.cfg.max_positions_long
        n_short = self.cfg.max_positions_short

        # LONG-SIDE FILTER: momentum crash protection (Daniel-Moskowitz 2016)
        # Trade audit: losing longs have 9.6% mean 6M momentum vs winners 5.3%.
        # Cap at long_max_mom126 blocks buying stocks that already rallied too much.
        long_eligible = set(predictions.index)
        long_max_mom = getattr(self.cfg, "long_max_mom126", 0.0)
        if long_max_mom > 0 and momentum_6m is not None:
            high_mom_longs = set(momentum_6m[momentum_6m > long_max_mom].index)
            long_eligible -= high_mom_longs

        # Short-side filter: build pool of stocks eligible for shorting
        # Based on audit: F4 (vol filter) is the best market-neutral filter (+0.10 Sharpe).
        # F1 (momentum/52w) was disabled after audit showed it hurt total Sharpe by 0.05.
        # Net exposure stays 0 (we still pick n_short from filtered pool).
        short_eligible = set(predictions.index)

        # VIX gate for shorts: trade audit showed shorts lose in high-VIX regimes.
        # Skip shorts entirely when VIX > threshold.
        short_max_vix = getattr(self.cfg, "short_max_vix", 0.0)
        short_enabled = True
        if short_max_vix > 0 and vix_current is not None and not pd.isna(vix_current):
            if vix_current > short_max_vix:
                short_enabled = False
                short_eligible = set()

        # Sector blacklist for shorts: trade audit identified sectors where shorts
        # systematically lose (defensive/yield). Configurable via avoid_short_sectors.
        avoid_sectors = getattr(self.cfg, "avoid_short_sectors", None)
        if avoid_sectors and sector_map:
            bad_sector_tickers = {
                t for t, s in sector_map.items() if s in avoid_sectors
            }
            short_eligible -= bad_sector_tickers

        # F1: momentum filter (disabled by default — hurts total Sharpe)
        if momentum_6m is not None:
            max_mom = self.cfg.short_max_6m_momentum
            if max_mom > 0:
                bad_mom = set(momentum_6m[momentum_6m > max_mom].index)
                short_eligible -= bad_mom
        if dist_from_52w_high is not None:
            max_dist = -self.cfg.short_min_dist_from_high
            if max_dist < 0:
                near_high = set(dist_from_52w_high[dist_from_52w_high >= max_dist].index)
                short_eligible -= near_high

        # F4: realized volatility filter (enabled — best market-neutral fix)
        # Exclude high-vol shorts that could squeeze. Filters 2-7% of universe per year.
        if vol_estimates is not None:
            max_vol = getattr(self.cfg, "short_max_63d_vol", 0.0)
            if max_vol > 0:
                high_vol = set(vol_estimates[vol_estimates > max_vol].index)
                short_eligible -= high_vol

        # UNIVERSAL QUALITY FILTERS for shorts (Citadel-style, not sector-hardcoded)
        # Blocks the "IonQ pattern" — small-cap, unprofitable, speculative stocks with
        # asymmetric upside from breakthroughs/catalysts. TP/FP analysis showed:
        #   - Unprofitable shorts: -16.6 bps alpha (losing)
        #   - Small-cap shorts: -17.1 bps alpha
        #   - Stacking mature+profitable+low-vol on shorts: +13.4 bps alpha (winning)
        min_mcap = getattr(self.cfg, "short_min_mcap", 0.0)
        if min_mcap > 0 and mcap_current is not None:
            small_caps = set(mcap_current[mcap_current < min_mcap].index)
            short_eligible -= small_caps

        min_ey = getattr(self.cfg, "short_min_earnings_yield", 0.0)
        if min_ey is not None and earnings_yield_current is not None:
            # EY <= 0 means unprofitable or negative-earnings
            unprofitable = set(earnings_yield_current[earnings_yield_current <= min_ey].index)
            short_eligible -= unprofitable

        # STOP-LOSS: force exit from existing shorts that have rallied > threshold
        # Uses entry price tracked in self._short_entry_price.
        # Blacklists the stock for N days to prevent immediate re-entry.
        forced_exits = set()
        stop_pct = getattr(self.cfg, "short_stop_loss_pct", 0.0)
        if stop_pct > 0 and current_prices is not None and self._short_entry_price:
            prev_shorts = set()
            if prev_weights is not None and not prev_weights.empty:
                prev_shorts = set(prev_weights[prev_weights < 0].index)
            for t in list(prev_shorts):
                if t in self._short_entry_price and t in current_prices.index:
                    curr = current_prices[t]
                    entry = self._short_entry_price[t]
                    if pd.notna(curr) and entry > 0:
                        # Stock rallied from entry: (curr - entry) / entry > stop_pct
                        rally = (curr / entry) - 1
                        if rally > stop_pct:
                            forced_exits.add(t)

        # Clear expired blacklist entries
        blacklist_days = getattr(self.cfg, "short_stop_blacklist_days", 0)
        if blacklist_days > 0:
            self._short_blacklist = {
                k: v for k, v in self._short_blacklist.items() if v > date
            }
            # Add new forced exits to blacklist
            if forced_exits and blacklist_days > 0:
                expiry = date + pd.Timedelta(days=blacklist_days)
                for t in forced_exits:
                    self._short_blacklist[t] = expiry
                    if t in self._short_entry_price:
                        del self._short_entry_price[t]

            # Remove blacklisted tickers from short pool
            short_eligible -= set(self._short_blacklist.keys())

        # Long side: uses long_eligible pool (momentum cap applied); short side: short_eligible.
        # 1. Cross-sectional rank (0 to 1) — use full universe for long ranking context
        ranks = predictions.rank(pct=True)

        # Dispersion scaling REMOVED. Diagnostic audit showed pred_std has only
        # +0.032 correlation with realized return — essentially noise. The dispersion
        # claim of "0.48 correlation" was from contaminated pre-PIT-fix data.
        # Removing simplifies the pipeline with no performance impact.
        dispersion_scale = 1.0

        # DRAWDOWN CIRCUIT BREAKER — smooth linear ramp (was binary sticky cut)
        # Old design: when DD < -3%, dd_scale = 0.25 (binary cut). Problem: at 25%
        # gross, recovery to a fresh peak is ~16x slower in calendar time, so the
        # breaker stayed permanently active. Run 7: breaker fired Nov 2021, never
        # released → strategy ran at 4% avg gross for 4 years.
        #
        # New design: linear interpolation from 1.0× (at threshold) to floor (at 2x threshold).
        # - DD ≥ -3% → scale = 1.0 (no cut)
        # - DD = -4.5% → scale = 0.625 (half-cut)
        # - DD ≤ -6% → scale = floor (e.g. 0.25)
        # This way the breaker partially releases as conditions improve.
        dd_threshold = getattr(self.cfg, "dd_circuit_breaker_threshold", 0.0)
        dd_floor = getattr(self.cfg, "dd_circuit_breaker_scale", 1.0)
        current_dd = self.get_current_dd()
        dd_scale = 1.0
        if dd_threshold < 0 and current_dd < dd_threshold:
            # Linear ramp from 1.0 at threshold to floor at 2× threshold
            ramp_span = abs(dd_threshold)  # e.g. 0.03
            excess = (dd_threshold - current_dd)  # how far past threshold (positive)
            ramp_fraction = min(excess / ramp_span, 1.0)
            dd_scale = 1.0 - ramp_fraction * (1.0 - dd_floor)

        # Combined leverage scaling factor
        leverage_scale = dispersion_scale * dd_scale  # 0.02 is typical std

        # Hysteresis: stocks enter at rank ≤ N, hold until rank > exit_N
        # This reduces turnover by 60%+ while preserving alpha (audit showed +0.37 Sharpe)
        exit_n_long = int(n_long * self.cfg.hysteresis_exit_mult)
        exit_n_short = int(n_short * self.cfg.hysteresis_exit_mult)

        prev_long = set()
        prev_short = set()
        if prev_weights is not None and not prev_weights.empty:
            prev_long = set(prev_weights[prev_weights > 0].index)
            prev_short = set(prev_weights[prev_weights < 0].index)

        # Long selection with hysteresis — restricted to long_eligible pool
        # (excludes stocks exceeding mom126 cap)
        eligible_long_ranks = ranks[ranks.index.isin(long_eligible)]
        ranks_desc = eligible_long_ranks.sort_values(ascending=False)
        long_tickers = []
        # Keep existing positions that are still in the hold zone (top exit_N of eligible pool)
        hold_zone_long = set(ranks_desc.head(exit_n_long).index)
        for t in prev_long:
            if t in hold_zone_long and len(long_tickers) < n_long:
                long_tickers.append(t)
        # Fill remaining slots with new top-N entries from eligible pool
        for t in ranks_desc.index:
            if len(long_tickers) >= n_long:
                break
            if t not in long_tickers:
                long_tickers.append(t)

        # Short selection with hysteresis — restricted to short_eligible pool
        # (excludes vol filter + sector blacklist + VIX gate + forced_exits)
        short_tickers = []
        if self.cfg.long_short and short_enabled:
            # Only rank stocks that are eligible for shorting
            eligible_ranks = ranks[ranks.index.isin(short_eligible)]
            ranks_asc = eligible_ranks.sort_values(ascending=True)
            hold_zone_short = set(ranks_asc.head(exit_n_short).index)
            # Keep existing shorts that are still in hold zone AND not forced-exited
            for t in prev_short:
                if t in forced_exits:
                    continue  # stop-loss kicked in — exit this position
                if t in hold_zone_short and len(short_tickers) < n_short:
                    short_tickers.append(t)
            # Fill remaining slots from eligible pool
            for t in ranks_asc.index:
                if len(short_tickers) >= n_short:
                    break
                if t not in short_tickers and t not in forced_exits:
                    short_tickers.append(t)

        # Update entry prices for NEW shorts (not in prev_short)
        if current_prices is not None:
            new_shorts = set(short_tickers) - prev_short
            for t in new_shorts:
                if t in current_prices.index and pd.notna(current_prices[t]) and current_prices[t] > 0:
                    self._short_entry_price[t] = float(current_prices[t])
            # Remove entry prices for shorts that exited
            exited_shorts = prev_short - set(short_tickers)
            for t in exited_shorts:
                if t in self._short_entry_price:
                    del self._short_entry_price[t]

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

        # 3. Scale to asymmetric long/short targets, adjusted by dispersion + DD circuit breaker.
        # long_gross_target and short_gross_target allow long-tilted portfolios
        # (audit showed shorts are net destroyers — size them smaller than longs).
        long_gross_cfg = getattr(self.cfg, "long_gross_target", 0.0)
        short_gross_cfg = getattr(self.cfg, "short_gross_target", 0.0)
        if long_gross_cfg <= 0:
            long_gross_cfg = self.cfg.max_gross_leverage / 2
        if short_gross_cfg <= 0:
            short_gross_cfg = self.cfg.max_gross_leverage / 2
        long_target = long_gross_cfg * leverage_scale
        short_target = short_gross_cfg * leverage_scale

        long_sum = weights[weights > 0].sum()
        short_sum = weights[weights < 0].abs().sum()
        if long_sum > 0:
            weights[weights > 0] *= long_target / long_sum
        if short_sum > 0:
            weights[weights < 0] *= short_target / short_sum

        # 4. Sector cap (max % of gross in any single sector)
        if sector_map:
            weights = self._apply_sector_cap(weights, sector_map)

        # 5. Position cap
        weights = weights.clip(-self.cfg.max_position_pct, self.cfg.max_position_pct)

        # 6. Re-enforce asymmetric gross targets after clipping (long != short is OK now)
        long_sum = weights[weights > 0].sum()
        short_sum = weights[weights < 0].abs().sum()
        if long_sum > 0:
            weights[weights > 0] *= long_target / long_sum
        if short_sum > 0:
            weights[weights < 0] *= short_target / short_sum

        # 7. Turnover constraint
        if prev_weights is not None:
            weights = self._apply_turnover_limit(weights, prev_weights)

        # Remove dust — threshold is adaptive to the effective target gross, which
        # matters for asymmetric large-N portfolios under the DD circuit breaker.
        # Fixed-0.002 dust killed positions when (gross × DD_scale × dispersion_scale)
        # / n_positions fell below 0.002 — causing target_weights to collapse to empty
        # and prev_weights to stay empty forever (death spiral).
        # New threshold: 10% of the average position size at current leverage_scale.
        n_positions = max(1, n_long + n_short)
        avg_pos = (long_target + short_target) / n_positions
        dust_threshold = max(1e-5, min(0.002, 0.1 * avg_pos))
        weights = weights[weights.abs() > dust_threshold]

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
        """
        Limit daily turnover to prevent excessive trading.

        Note: dust filtering is intentionally NOT done here. The hardcoded 0.002
        threshold previously here was the second source of the death-spiral bug:
        when DD circuit breaker fires and target weights drop to ~0.001 each,
        this filter would wipe every position to zero, leaving an empty target
        that the rest of the pipeline can't recover from. Dust filtering is now
        only done once at the very end of construct_portfolio with an adaptive
        threshold based on effective gross.
        """
        if prev.empty or prev.abs().sum() < 0.01:
            return target

        all_tickers = target.index.union(prev.index)
        target = target.reindex(all_tickers, fill_value=0)
        prev = prev.reindex(all_tickers, fill_value=0)
        trades = target - prev
        turnover = trades.abs().sum()

        if turnover > self.cfg.max_daily_turnover:
            scale = self.cfg.max_daily_turnover / turnover
            target = prev + trades * scale

        return target


def compute_portfolio_returns(
    weights_history: Dict, prices: pd.DataFrame, cfg: PortfolioConfig,
) -> pd.DataFrame:
    """
    Replay weights_history into a daily P&L DataFrame.

    Critical: prev_weights MUST be updated on every iteration that completes,
    not just successful ones. Otherwise, if prev_weights ever becomes empty
    (because target_weights from a prior day was empty for any reason),
    the loop never recovers — len(tickers) == 0 forever, all subsequent
    days produce no rows. This was the death-spiral bug that capped the
    backtest at 172 days when 100L/30S + DD circuit breaker fired.

    Now: any iteration where the realized return CAN'T be computed (no
    overlap, missing prices, empty prev_weights) records a zero-return row
    and STILL advances prev_weights to today's target. The portfolio
    "sat in cash" that day but is rebuilt the next day.
    """
    dates = sorted(weights_history.keys())
    results = []
    prev_weights = pd.Series(dtype=float)

    for date in dates:
        target_weights = weights_history[date]
        if date not in prices.index:
            # Can't even index this date — skip but advance state
            prev_weights = target_weights
            continue

        if date == dates[0]:
            # First day: just record initial state, no return yet
            prev_weights = target_weights
            continue

        prev_idx = prices.index.get_loc(date) - 1
        if prev_idx < 0:
            prev_weights = target_weights
            continue
        prev_date = prices.index[prev_idx]

        # Try to compute realized P&L from prev_weights
        port_ret = 0.0
        tc = 0.0
        turnover = 0.0
        if not prev_weights.empty:
            tickers = prev_weights.index.intersection(prices.columns)
            tickers = tickers[
                prices.loc[date, tickers].notna() & prices.loc[prev_date, tickers].notna()
            ]
            if len(tickers) > 0:
                stock_rets = prices.loc[date, tickers] / prices.loc[prev_date, tickers] - 1
                port_ret = float((prev_weights.reindex(tickers, fill_value=0) * stock_rets).sum())

                trades = (
                    target_weights.reindex(tickers, fill_value=0)
                    - prev_weights.reindex(tickers, fill_value=0)
                )
                turnover = float(trades.abs().sum())
                base_cost_bps = cfg.commission_bps + cfg.slippage_bps + cfg.spread_bps
                tc = turnover * base_cost_bps / 10000

        # Always record a row — even if it's a "cash day" with zero return.
        # This prevents the loop from invisibly skipping huge stretches.
        results.append({
            "date": date,
            "gross_return": port_ret,
            "tc_cost": tc,
            "net_return": port_ret - tc,
            "turnover": turnover,
            "n_long": int((prev_weights > 0).sum()),
            "n_short": int((prev_weights < 0).sum()),
            "gross_exposure": float(prev_weights.abs().sum()),
            "net_exposure": float(prev_weights.sum()),
        })
        # ALWAYS advance prev_weights, even after a "cash day". This is the
        # death-spiral fix — without this line, the loop is stuck once
        # prev_weights becomes empty.
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
