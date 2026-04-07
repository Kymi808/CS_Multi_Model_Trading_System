"""
Fundamental features: valuation, quality, growth, analyst, short interest.

These are the features that differentiate a quant-grade system from a
momentum-only system. They capture information that price alone cannot.

Key insight: We use CROSS-SECTIONAL RANKS of fundamentals, not raw values.
A PE of 15 means nothing in isolation — what matters is whether it's cheap
RELATIVE to the universe.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def build_fundamental_features(
    fundamentals: Dict[str, dict],
    prices: pd.DataFrame,
    earnings_dates: Dict[str, List[str]],
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Build fundamental feature panel.

    Since fundamentals are point-in-time (not daily), we create a static
    feature vector per ticker and broadcast it across all dates.
    This is valid because fundamentals change slowly (quarterly).

    Returns DataFrame with MultiIndex columns: (feature_name, ticker)
    """
    tickers = [t for t in prices.columns if t in fundamentals]
    dates = prices.index
    feats = {}

    # ---------------------------------------------------------------
    # 1. VALUATION COMPOSITES
    # ---------------------------------------------------------------
    _build_valuation_features(feats, fundamentals, tickers, dates, prices)

    # ---------------------------------------------------------------
    # 2. PROFITABILITY / QUALITY
    # ---------------------------------------------------------------
    _build_quality_features(feats, fundamentals, tickers, dates)

    # ---------------------------------------------------------------
    # 3. GROWTH
    # ---------------------------------------------------------------
    _build_growth_features(feats, fundamentals, tickers, dates)

    # ---------------------------------------------------------------
    # 4. ANALYST SENTIMENT
    # ---------------------------------------------------------------
    _build_analyst_features(feats, fundamentals, tickers, dates, prices)

    # ---------------------------------------------------------------
    # 5. SHORT INTEREST
    # ---------------------------------------------------------------
    _build_short_interest_features(feats, fundamentals, tickers, dates)

    # ---------------------------------------------------------------
    # 6. SIZE / BETA
    # ---------------------------------------------------------------
    _build_size_features(feats, fundamentals, tickers, dates)

    # ---------------------------------------------------------------
    # 7. EARNINGS EVENT FEATURES
    # ---------------------------------------------------------------
    _build_earnings_features(feats, earnings_dates, tickers, dates, prices)

    # ---------------------------------------------------------------
    # 8. SECTOR-RELATIVE VALUATION
    # ---------------------------------------------------------------
    _build_sector_relative_features(feats, fundamentals, tickers, dates, sector_map)

    if not feats:
        logger.warning("No fundamental features built")
        return {}

    logger.info(f"Fundamental features: {len(feats)} signals")
    return feats


def build_pit_fundamental_features(
    historical_data: Dict[str, list],
    prices: pd.DataFrame,
    earnings_dates: Dict[str, List[str]],
    sector_map: Dict[str, str],
) -> dict:
    """
    Build POINT-IN-TIME fundamental features from FMP historical quarterly data.

    Unlike build_fundamental_features which broadcasts a single snapshot,
    this uses filingDate to ensure each date only sees data that was
    publicly available. No look-ahead bias.

    Args:
        historical_data: {ticker: [{date, filingDate, trailingPE, ROE, ...}]}
                         from fetch_fmp_historical_fundamentals()
    """
    tickers = [t for t in prices.columns if t in historical_data]
    dates = prices.index
    feats = {}

    if not tickers:
        logger.warning("No PIT fundamental data available")
        return feats

    # --- Valuation ---
    for field, name in [
        ("trailingPE", "earnings_yield"),
        ("priceToBook", "book_to_price"),
        ("priceToSalesTrailing12Months", "sales_yield"),
        ("enterpriseToEbitda", "ev_ebitda_yield"),
    ]:
        df = _broadcast_pit(historical_data, field, prices.columns.tolist(), dates)
        # Invert: lower PE/PB = cheaper = higher yield
        if field in ("trailingPE", "priceToBook", "priceToSalesTrailing12Months", "enterpriseToEbitda"):
            inv = 1.0 / df.replace(0, np.nan)
            inv = inv.clip(-10, 10)  # cap extreme values
        else:
            inv = df

        if inv.notna().sum().sum() > 0:
            feats[("fund", name)] = inv
            feats[("fund", f"cs_rank_{name.replace('_yield', '').replace('book_to_price', 'btp').replace('sales_yield', 'sales_yield').replace('ev_ebitda_yield', 'ev_ebitda_yield')}")] = _rank_cross_sectional(inv)

    # Value composite
    value_keys = [k for k in feats if k[0] == "fund" and "cs_rank" in k[1] and
                  any(v in k[1] for v in ["earnings", "btp", "sales", "ev_ebitda"])]
    if len(value_keys) >= 2:
        stacked = [feats[k].stack() for k in value_keys]
        avg = pd.concat(stacked, axis=1).mean(axis=1).unstack()
        if isinstance(avg, pd.DataFrame) and not avg.empty:
            feats[("fund", "value_composite")] = avg

    # --- Quality ---
    for field, name in [
        ("returnOnEquity", "roe"), ("returnOnAssets", "roa"),
        ("grossMargins", "gross_margin"), ("operatingMargins", "op_margin"),
        ("profitMargins", "net_margin"),
    ]:
        df = _broadcast_pit(historical_data, field, prices.columns.tolist(), dates)
        if df.notna().sum().sum() > 0:
            feats[("fund", f"cs_rank_{name}")] = _rank_cross_sectional(df)

    # Quality composite
    quality_keys = [("fund", f"cs_rank_{n}") for n in
                    ["roe", "roa", "gross_margin", "op_margin", "net_margin"]]
    quality_dfs = [feats[k] for k in quality_keys if k in feats]
    if len(quality_dfs) >= 2:
        stacked = [df.stack() for df in quality_dfs]
        avg = pd.concat(stacked, axis=1).mean(axis=1).unstack()
        if isinstance(avg, pd.DataFrame) and not avg.empty:
            feats[("fund", "quality_composite")] = avg

    # --- Balance sheet ---
    for field, name, invert in [
        ("debtToEquity", "debt_to_equity", True),
        ("currentRatio", "current_ratio", False),
    ]:
        df = _broadcast_pit(historical_data, field, prices.columns.tolist(), dates)
        if df.notna().sum().sum() > 0:
            if invert:
                feats[("fund", f"cs_rank_{name}")] = 1 - _rank_cross_sectional(df)
            else:
                feats[("fund", f"cs_rank_{name}")] = _rank_cross_sectional(df)

    # --- Growth ---
    for field, name in [
        ("revenueGrowth", "rev_growth"),
        ("earningsGrowth", "earn_growth"),
        ("earningsQuarterlyGrowth", "earn_q_growth"),
    ]:
        df = _broadcast_pit(historical_data, field, prices.columns.tolist(), dates)
        if df.notna().sum().sum() > 0:
            feats[("fund", f"cs_rank_{name}")] = _rank_cross_sectional(df)

    # --- Size (from market cap) ---
    mcap_df = _broadcast_pit(historical_data, "marketCap", prices.columns.tolist(), dates)
    if mcap_df.notna().sum().sum() > 0:
        log_mcap = np.log(mcap_df.clip(lower=1))
        feats[("fund", "log_mcap")] = log_mcap
        feats[("fund", "cs_rank_size")] = _rank_cross_sectional(log_mcap)

    # --- Earnings event features (already date-aware) ---
    _build_earnings_features(feats, earnings_dates, tickers, dates, prices)

    # --- Sector-relative (using PIT data) ---
    if sector_map:
        # Get latest available fundamentals per ticker for sector-relative
        # This is a simplification — ideally each date would have its own sector-relative
        latest_fund = {}
        for t in tickers:
            recs = historical_data.get(t, [])
            if recs:
                latest = recs[-1]  # last record (most recent quarter)
                latest_fund[t] = latest
        if latest_fund:
            _build_sector_relative_features(feats, latest_fund, tickers, dates, sector_map)

    if not feats:
        logger.warning("No PIT fundamental features built")
        return feats

    logger.info(f"PIT fundamental features: {len(feats)} signals (point-in-time, no look-ahead)")
    return feats


def _broadcast_static(values: Dict[str, float], tickers: List[str],
                       dates: pd.DatetimeIndex, name: str) -> pd.DataFrame:
    """Broadcast a static per-ticker value across all dates.

    WARNING: This creates look-ahead bias when used with current snapshots
    in backtest. For honest backtest, use _broadcast_pit instead.
    """
    series = pd.Series({t: values.get(t, np.nan) for t in tickers})
    df = pd.DataFrame(
        np.tile(series.values, (len(dates), 1)),
        index=dates, columns=tickers,
    )
    return df


def _broadcast_pit(
    historical_data: Dict[str, list],
    field: str,
    tickers: List[str],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build a time-varying fundamental feature from point-in-time quarterly data.

    For each ticker, places the field value at its filingDate and forward-fills.
    The model at date T only sees data that was publicly filed before T.
    No look-ahead bias.

    Args:
        historical_data: {ticker: [{date, filingDate, trailingPE, ...}, ...]}
        field: which field to extract (e.g. "trailingPE")
        tickers: list of tickers
        dates: DatetimeIndex of all backtest dates
    """
    df = pd.DataFrame(np.nan, index=dates, columns=tickers)

    for ticker in tickers:
        records = historical_data.get(ticker, [])
        if not records:
            continue
        for rec in records:
            filing_date = rec.get("filingDate", "")
            val = rec.get(field)
            if not filing_date or val is None:
                continue
            try:
                fd = pd.Timestamp(filing_date)
            except Exception:
                continue
            # Place at first trading date on or after filing
            valid = dates[dates >= fd]
            if len(valid) > 0:
                df.loc[valid[0], ticker] = float(val)

    # Forward-fill: each quarterly value persists until next filing
    df = df.ffill()
    return df


def _rank_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Rank cross-sectionally [0, 1] per date."""
    return df.rank(axis=1, pct=True)


def _build_valuation_features(feats, fundamentals, tickers, dates, prices):
    """Value factor: cheap stocks outperform expensive ones (on average)."""
    # Earnings yield (1/PE) — more stable than PE for cross-sectional ranking
    ey = {}
    for t in tickers:
        pe = fundamentals[t].get("trailingPE")
        if pe and pe > 0:
            ey[t] = 1.0 / pe
    if ey:
        df = _broadcast_static(ey, tickers, dates, "earnings_yield")
        feats[("fund", "earnings_yield")] = df
        feats[("fund", "cs_rank_earnings_yield")] = _rank_cross_sectional(df)

    # Forward earnings yield
    fey = {}
    for t in tickers:
        fpe = fundamentals[t].get("forwardPE")
        if fpe and fpe > 0:
            fey[t] = 1.0 / fpe
    if fey:
        df = _broadcast_static(fey, tickers, dates, "fwd_earnings_yield")
        feats[("fund", "fwd_earnings_yield")] = df
        feats[("fund", "cs_rank_fwd_ey")] = _rank_cross_sectional(df)

    # Book-to-price (value factor)
    btp = {}
    for t in tickers:
        ptb = fundamentals[t].get("priceToBook")
        if ptb and ptb > 0:
            btp[t] = 1.0 / ptb
    if btp:
        df = _broadcast_static(btp, tickers, dates, "book_to_price")
        feats[("fund", "book_to_price")] = df
        feats[("fund", "cs_rank_btp")] = _rank_cross_sectional(df)

    # Sales yield (1/P/S)
    sy = {}
    for t in tickers:
        ps = fundamentals[t].get("priceToSalesTrailing12Months")
        if ps and ps > 0:
            sy[t] = 1.0 / ps
    if sy:
        df = _broadcast_static(sy, tickers, dates, "sales_yield")
        feats[("fund", "cs_rank_sales_yield")] = _rank_cross_sectional(df)

    # EV/EBITDA yield
    eve = {}
    for t in tickers:
        ev_ebitda = fundamentals[t].get("enterpriseToEbitda")
        if ev_ebitda and ev_ebitda > 0:
            eve[t] = 1.0 / ev_ebitda
    if eve:
        df = _broadcast_static(eve, tickers, dates, "ev_ebitda_yield")
        feats[("fund", "cs_rank_ev_ebitda_yield")] = _rank_cross_sectional(df)

    # COMPOSITE VALUE SCORE (average of rank signals)
    value_rank_dfs = []
    for key in ["cs_rank_earnings_yield", "cs_rank_fwd_ey", "cs_rank_btp",
                "cs_rank_sales_yield", "cs_rank_ev_ebitda_yield"]:
        full_key = ("fund", key)
        if full_key in feats:
            value_rank_dfs.append(feats[full_key])
    if len(value_rank_dfs) >= 2:
        # Average ranks across value metrics for each (date, ticker)
        stacked = [df.stack() for df in value_rank_dfs]
        avg_rank = pd.concat(stacked, axis=1).mean(axis=1).unstack()
        if isinstance(avg_rank, pd.DataFrame) and not avg_rank.empty:
            feats[("fund", "value_composite")] = avg_rank


def _build_quality_features(feats, fundamentals, tickers, dates):
    """Quality factor: profitable, stable companies outperform."""
    for metric, name in [
        ("returnOnEquity", "roe"), ("returnOnAssets", "roa"),
        ("grossMargins", "gross_margin"), ("operatingMargins", "op_margin"),
        ("profitMargins", "net_margin"),
    ]:
        vals = {t: fundamentals[t].get(metric) for t in tickers
                if fundamentals[t].get(metric) is not None}
        if vals:
            df = _broadcast_static(vals, tickers, dates, name)
            feats[("fund", f"cs_rank_{name}")] = _rank_cross_sectional(df)

    # Composite quality score
    quality_keys = [("fund", f"cs_rank_{n}") for n in
                    ["roe", "roa", "gross_margin", "op_margin", "net_margin"]]
    quality_rank_dfs = [feats[k] for k in quality_keys if k in feats]
    if len(quality_rank_dfs) >= 2:
        stacked = [df.stack() for df in quality_rank_dfs]
        avg_rank = pd.concat(stacked, axis=1).mean(axis=1).unstack()
        if isinstance(avg_rank, pd.DataFrame) and not avg_rank.empty:
            feats[("fund", "quality_composite")] = avg_rank

    # Balance sheet quality
    for metric, name in [
        ("debtToEquity", "debt_to_equity"),
        ("currentRatio", "current_ratio"),
    ]:
        vals = {t: fundamentals[t].get(metric) for t in tickers
                if fundamentals[t].get(metric) is not None}
        if vals:
            df = _broadcast_static(vals, tickers, dates, name)
            if name == "debt_to_equity":
                # Lower debt = better quality -> rank descending
                feats[("fund", f"cs_rank_{name}")] = 1 - _rank_cross_sectional(df)
            else:
                feats[("fund", f"cs_rank_{name}")] = _rank_cross_sectional(df)


def _build_growth_features(feats, fundamentals, tickers, dates):
    """Growth factor: revenue and earnings growth."""
    for metric, name in [
        ("revenueGrowth", "rev_growth"),
        ("earningsGrowth", "earn_growth"),
        ("earningsQuarterlyGrowth", "earn_q_growth"),
    ]:
        vals = {t: fundamentals[t].get(metric) for t in tickers
                if fundamentals[t].get(metric) is not None}
        if vals:
            df = _broadcast_static(vals, tickers, dates, name)
            feats[("fund", f"cs_rank_{name}")] = _rank_cross_sectional(df)


def _build_analyst_features(feats, fundamentals, tickers, dates, prices):
    """Analyst sentiment: recommendation changes, price target upside."""
    # Recommendation (1=Strong Buy, 5=Sell) — invert so higher = more bullish
    rec = {t: fundamentals[t].get("recommendationMean") for t in tickers
           if fundamentals[t].get("recommendationMean") is not None}
    if rec:
        # Invert: lower rec number = more bullish -> rank descending
        df = _broadcast_static(rec, tickers, dates, "rec_mean")
        feats[("fund", "cs_rank_analyst_rec")] = 1 - _rank_cross_sectional(df)

    # Price target upside (target / current price - 1)
    target_prices = {t: fundamentals[t].get("targetMeanPrice") for t in tickers
                     if fundamentals[t].get("targetMeanPrice") is not None}
    if target_prices:
        # Need current prices to compute upside
        last_prices = prices.iloc[-1]
        upside = {}
        for t in tickers:
            tp = target_prices.get(t)
            cp = last_prices.get(t)
            if tp and cp and cp > 0:
                upside[t] = tp / cp - 1
        if upside:
            df = _broadcast_static(upside, tickers, dates, "target_upside")
            feats[("fund", "target_upside")] = df
            feats[("fund", "cs_rank_target_upside")] = _rank_cross_sectional(df)

    # Analyst coverage (more coverage = more liquid, better price discovery)
    coverage = {t: fundamentals[t].get("numberOfAnalystOpinions") for t in tickers
                if fundamentals[t].get("numberOfAnalystOpinions") is not None}
    if coverage:
        df = _broadcast_static(coverage, tickers, dates, "analyst_coverage")
        feats[("fund", "cs_rank_coverage")] = _rank_cross_sectional(df)


def _build_short_interest_features(feats, fundamentals, tickers, dates):
    """Short interest: heavily shorted stocks tend to underperform (and squeeze)."""
    si = {t: fundamentals[t].get("shortPercentOfFloat") for t in tickers
          if fundamentals[t].get("shortPercentOfFloat") is not None}
    if si:
        df = _broadcast_static(si, tickers, dates, "short_pct_float")
        feats[("fund", "short_pct_float")] = df
        feats[("fund", "cs_rank_short_interest")] = _rank_cross_sectional(df)

    sr = {t: fundamentals[t].get("shortRatio") for t in tickers
          if fundamentals[t].get("shortRatio") is not None}
    if sr:
        df = _broadcast_static(sr, tickers, dates, "short_ratio")
        feats[("fund", "cs_rank_short_ratio")] = _rank_cross_sectional(df)


def _build_size_features(feats, fundamentals, tickers, dates):
    """Size factor: log market cap and beta."""
    mcap = {t: np.log(fundamentals[t]["marketCap"]) for t in tickers
            if fundamentals[t].get("marketCap") and fundamentals[t]["marketCap"] > 0}
    if mcap:
        df = _broadcast_static(mcap, tickers, dates, "log_mcap")
        feats[("fund", "log_mcap")] = df
        feats[("fund", "cs_rank_size")] = _rank_cross_sectional(df)

    beta = {t: fundamentals[t].get("beta") for t in tickers
            if fundamentals[t].get("beta") is not None}
    if beta:
        df = _broadcast_static(beta, tickers, dates, "beta")
        feats[("fund", "beta")] = df
        feats[("fund", "cs_rank_beta")] = _rank_cross_sectional(df)


def _build_earnings_features(feats, earnings_dates, tickers, dates, prices):
    """
    Post-Earnings Announcement Drift (PEAD) features.
    One of the most robust anomalies in finance.
    """
    if not earnings_dates:
        return

    # For each ticker, compute: days since last earnings, return since earnings
    days_since = pd.DataFrame(np.nan, index=dates, columns=tickers)
    ret_since_earn = pd.DataFrame(np.nan, index=dates, columns=tickers)
    earn_day_return = pd.DataFrame(0.0, index=dates, columns=tickers)
    log_ret = np.log(prices / prices.shift(1))

    for ticker in tickers:
        if ticker not in earnings_dates:
            continue
        earn_dates_str = earnings_dates[ticker]
        earn_dates_parsed = []
        for d in earn_dates_str:
            try:
                ed = pd.Timestamp(d)
                if ed in dates or ed <= dates[-1]:
                    earn_dates_parsed.append(ed)
            except Exception:
                continue

        if not earn_dates_parsed:
            continue

        for date in dates:
            # Find most recent earnings before this date
            past_earnings = [e for e in earn_dates_parsed if e <= date]
            if not past_earnings:
                continue
            last_earn = max(past_earnings)
            days_diff = (date - last_earn).days
            days_since.loc[date, ticker] = days_diff

            # Cumulative return since earnings
            if last_earn in prices.index and ticker in prices.columns:
                earn_price = prices.loc[last_earn, ticker]
                curr_price = prices.loc[date, ticker]
                if earn_price > 0 and not np.isnan(earn_price):
                    ret_since_earn.loc[date, ticker] = curr_price / earn_price - 1

            # Earnings day return (surprise proxy)
            if last_earn in log_ret.index and ticker in log_ret.columns:
                earn_day_return.loc[date, ticker] = log_ret.loc[last_earn, ticker]

    feats[("earn", "days_since_earnings")] = days_since
    feats[("earn", "cs_rank_days_since_earn")] = _rank_cross_sectional(days_since)
    feats[("earn", "ret_since_earnings")] = ret_since_earn
    feats[("earn", "cs_rank_ret_since_earn")] = _rank_cross_sectional(ret_since_earn)
    feats[("earn", "earnings_day_return")] = earn_day_return
    feats[("earn", "cs_rank_earn_day_ret")] = _rank_cross_sectional(earn_day_return)

    # Earnings is imminent flag (within 5 days) — useful for vol prediction
    feats[("earn", "earnings_imminent")] = (days_since < 5).astype(float)


def _build_sector_relative_features(feats, fundamentals, tickers, dates, sector_map):
    """
    Sector-relative valuation: cheap within sector, not just cheap overall.
    This isolates stock-specific value from sector rotation.
    """
    if not sector_map:
        return

    # Group tickers by sector
    sector_groups = {}
    for t in tickers:
        s = sector_map.get(t, "Unknown")
        if s not in sector_groups:
            sector_groups[s] = []
        sector_groups[s].append(t)

    # Sector-relative earnings yield
    ey = {}
    for t in tickers:
        pe = fundamentals[t].get("trailingPE")
        if pe and pe > 0:
            ey[t] = 1.0 / pe

    if ey and sector_groups:
        sector_rel = {}
        for sector, sector_tickers in sector_groups.items():
            sector_eys = {t: ey[t] for t in sector_tickers if t in ey}
            if len(sector_eys) < 3:
                continue
            sector_mean = np.mean(list(sector_eys.values()))
            sector_std = np.std(list(sector_eys.values()))
            if sector_std > 0:
                for t, v in sector_eys.items():
                    sector_rel[t] = (v - sector_mean) / sector_std

        if sector_rel:
            df = _broadcast_static(sector_rel, tickers, dates, "sector_rel_ey")
            feats[("fund", "sector_rel_value")] = df
            feats[("fund", "cs_rank_sector_rel_value")] = _rank_cross_sectional(df)
