"""
Cross-asset features: macro regime detection and cross-asset signals.

Key insight: Individual stock returns are driven ~40% by the market,
~20% by sector, and only ~40% by idiosyncratic factors. Cross-asset
signals capture the macro component that pure price/volume features miss.

Features:
1. VIX level and term structure → risk regime
2. Yield curve slope → economic cycle
3. Credit spreads → financial stress
4. Dollar, gold, oil → macro factor exposure
5. Sector momentum → rotation signals
6. Market breadth → trend strength
7. Cross-asset correlations → regime detection
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def build_cross_asset_features(
    cross_asset_prices: pd.DataFrame,
    stock_prices: pd.DataFrame,
    sector_etf_prices: pd.DataFrame,
    sector_map: Dict[str, str],
    windows: List[int] = [5, 21, 63],
) -> pd.DataFrame:
    """
    Build cross-asset feature panel.

    These features are the SAME for all tickers on a given date
    (they're market-level signals), but we interact them with
    stock-level features to create conditional signals.

    Returns DataFrame with MultiIndex columns: (feature_name, ticker)
    """
    tickers = stock_prices.columns.tolist()
    dates = stock_prices.index
    feats = {}

    if cross_asset_prices.empty:
        logger.warning("No cross-asset data available")
        return pd.DataFrame(index=dates)

    ca = cross_asset_prices.reindex(dates).ffill()

    # ---------------------------------------------------------------
    # 1. VIX REGIME
    # ---------------------------------------------------------------
    if "^VIX" in ca.columns:
        vix = ca["^VIX"]
        # VIX level
        feats[("macro", "vix_level")] = _broadcast(vix, tickers)
        # VIX percentile (where is VIX relative to history)
        for w in [63, 252]:
            vix_pct = vix.rolling(w).apply(
                lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else np.nan
            )
            feats[("macro", f"vix_pctile_{w}d")] = _broadcast(vix_pct, tickers)
        # VIX change (spike = risk-off)
        for w in windows:
            feats[("macro", f"vix_chg_{w}d")] = _broadcast(
                vix.pct_change(w), tickers
            )

    # ---------------------------------------------------------------
    # 2. YIELD CURVE
    # ---------------------------------------------------------------
    if "^TNX" in ca.columns and "^IRX" in ca.columns:
        tnx = ca["^TNX"]  # 10Y yield
        irx = ca["^IRX"]  # 3M yield
        # Yield curve slope (10Y - 3M)
        slope = tnx - irx
        feats[("macro", "yield_curve_slope")] = _broadcast(slope, tickers)
        # Change in slope
        for w in [21, 63]:
            feats[("macro", f"yield_slope_chg_{w}d")] = _broadcast(
                slope.diff(w), tickers
            )
        # Rate level change
        for w in [21, 63]:
            feats[("macro", f"rate_10y_chg_{w}d")] = _broadcast(
                tnx.diff(w), tickers
            )

    # ---------------------------------------------------------------
    # 3. CREDIT SPREADS
    # ---------------------------------------------------------------
    if "HYG" in ca.columns and "LQD" in ca.columns:
        # HY-IG spread proxy (using ETF ratio)
        credit_spread = np.log(ca["LQD"] / ca["HYG"])
        feats[("macro", "credit_spread")] = _broadcast(credit_spread, tickers)
        for w in [21, 63]:
            feats[("macro", f"credit_spread_chg_{w}d")] = _broadcast(
                credit_spread.diff(w), tickers
            )

    # ---------------------------------------------------------------
    # 4. DOLLAR, GOLD, OIL
    # ---------------------------------------------------------------
    for asset, name in [("UUP", "dollar"), ("GLD", "gold"), ("USO", "oil")]:
        if asset in ca.columns:
            log_ret = np.log(ca[asset] / ca[asset].shift(1))
            for w in windows:
                feats[("macro", f"{name}_mom_{w}d")] = _broadcast(
                    log_ret.rolling(w).sum(), tickers
                )

    # ---------------------------------------------------------------
    # 5. MARKET BREADTH
    # ---------------------------------------------------------------
    # What % of stocks are above their 50-day MA?
    stock_ret = np.log(stock_prices / stock_prices.shift(1))
    for w in [21, 50]:
        ma = stock_prices.rolling(w).mean()
        above_ma = (stock_prices > ma).mean(axis=1)
        feats[("macro", f"breadth_{w}d")] = _broadcast(above_ma, tickers)

    # Market average momentum
    for w in windows:
        mkt_mom = stock_ret.rolling(w).sum().mean(axis=1)
        feats[("macro", f"mkt_avg_mom_{w}d")] = _broadcast(mkt_mom, tickers)

    # Market dispersion (cross-sectional vol of returns)
    for w in [21, 63]:
        dispersion = stock_ret.rolling(w).sum().std(axis=1)
        feats[("macro", f"mkt_dispersion_{w}d")] = _broadcast(dispersion, tickers)

    # ---------------------------------------------------------------
    # 6. SECTOR MOMENTUM → STOCK-LEVEL
    # ---------------------------------------------------------------
    if not sector_etf_prices.empty and sector_map:
        sector_etf_prices = sector_etf_prices.reindex(dates).ffill()
        _build_sector_momentum_features(
            feats, sector_etf_prices, stock_prices, sector_map, tickers, windows
        )

    # ---------------------------------------------------------------
    # 7. RISK-ON / RISK-OFF INDICATOR
    # ---------------------------------------------------------------
    _build_risk_regime_features(feats, ca, tickers, windows)

    if not feats:
        return {}

    logger.info(f"Cross-asset features: {len(feats)} signals")
    return feats


def _broadcast(series: pd.Series, tickers: List[str]) -> pd.DataFrame:
    """Broadcast a single time series to all tickers."""
    return pd.DataFrame(
        np.tile(series.values.reshape(-1, 1), (1, len(tickers))),
        index=series.index, columns=tickers,
    )


def _build_sector_momentum_features(
    feats, sector_etf_prices, stock_prices, sector_map, tickers, windows
):
    """
    Map sector ETF momentum to individual stocks.
    If Tech sector is hot, tech stocks get a positive signal.
    """
    # Build reverse map: sector name -> ETF ticker
    from config import DataConfig
    etf_map = DataConfig().sector_etf_map
    name_to_etf = {v: k for k, v in etf_map.items()}

    sector_ret = np.log(sector_etf_prices / sector_etf_prices.shift(1))

    for w in windows:
        # Sector momentum
        sector_mom = sector_ret.rolling(w).sum()

        # Map to individual stocks
        stock_sector_mom = pd.DataFrame(np.nan, index=stock_prices.index, columns=tickers)
        for ticker in tickers:
            sector = sector_map.get(ticker)
            if sector:
                etf = name_to_etf.get(sector)
                if etf and etf in sector_mom.columns:
                    stock_sector_mom[ticker] = sector_mom[etf]

        feats[("sector", f"own_sector_mom_{w}d")] = stock_sector_mom

    # Sector rotation signal: is this stock's sector gaining vs losing momentum?
    if len(windows) >= 2:
        for etf in sector_etf_prices.columns:
            if etf in sector_ret.columns:
                pass  # Could add more here

    # Relative sector strength (stock's sector rank among all sectors)
    for w in [21, 63]:
        if w not in windows:
            continue
        sector_mom = sector_ret.rolling(w).sum()
        sector_rank = sector_mom.rank(axis=1, pct=True)

        stock_sector_rank = pd.DataFrame(np.nan, index=stock_prices.index, columns=tickers)
        for ticker in tickers:
            sector = sector_map.get(ticker)
            if sector:
                etf = name_to_etf.get(sector)
                if etf and etf in sector_rank.columns:
                    stock_sector_rank[ticker] = sector_rank[etf]

        feats[("sector", f"sector_rank_{w}d")] = stock_sector_rank


def _rolling_pct_rank(series, window=252, min_periods=63):
    """
    Rolling percentile rank — PIT version of .rank(pct=True).

    At each date, returns the fraction of the last `window` values
    that are <= current value. Only looks at past/current data (no look-ahead).
    """
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: (x <= x[-1]).mean(), raw=True
    )


def _build_risk_regime_features(feats, ca, tickers, windows):
    """
    Composite risk-on/risk-off indicator.
    Risk-on: stocks up, VIX down, credit tight, yields up
    Risk-off: stocks down, VIX up, credit wide, yields down

    PIT: uses rolling 252-day percentile rank (not global rank).
    """
    signals = []

    if "^GSPC" in ca.columns:
        sp_ret = np.log(ca["^GSPC"] / ca["^GSPC"].shift(21))
        signals.append(_rolling_pct_rank(sp_ret))

    if "^VIX" in ca.columns:
        vix_chg = ca["^VIX"].pct_change(21)
        signals.append(1 - _rolling_pct_rank(vix_chg))  # Invert: VIX up = risk-off

    if "HYG" in ca.columns:
        hyg_ret = np.log(ca["HYG"] / ca["HYG"].shift(21))
        signals.append(_rolling_pct_rank(hyg_ret))

    if "TLT" in ca.columns:
        tlt_ret = np.log(ca["TLT"] / ca["TLT"].shift(21))
        signals.append(1 - _rolling_pct_rank(tlt_ret))  # Bonds up = risk-off

    if signals:
        risk_indicator = pd.concat(signals, axis=1).mean(axis=1)
        feats[("macro", "risk_on_off")] = _broadcast(risk_indicator, tickers)
