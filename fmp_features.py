"""
Point-in-time fundamental features from Financial Modeling Prep.

BACKTEST-SAFE: Each data point is only visible on or after its publication
date. No future data leaks into historical training windows.

In production (live trading), this naturally uses the latest available data.

Key features:
1. Earnings estimate revisions (strongest known alpha factor)
2. Forward PE vs trailing PE (growth expectations)
3. Earnings surprise (post-earnings drift anomaly)
4. Earnings beat streak (consistency premium)

Reference:
- Chan, Jegadeesh & Lakonishok (1996), "Momentum Strategies"
- Bernard & Thomas (1989), "Post-Earnings-Announcement Drift"
"""
import json
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def fetch_fmp_fundamentals(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
) -> Dict[str, list]:
    """
    Fetch HISTORICAL point-in-time fundamental data from FMP.

    Returns dict of {ticker: [list of dated records]} so that each
    record can be placed at its publication date during backtest.

    In production, only the latest record matters.
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")

    cache_file = os.path.join(cache_dir, "fmp_fundamentals_hist.json")

    # Check cache (daily)
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_date") == datetime.now().strftime("%Y-%m-%d"):
                logger.info("Loading cached FMP fundamentals (historical)")
                return {k: v for k, v in cached.items() if k != "_date"}
        except Exception:
            pass

    if not api_key or api_key in ("", "xxxxx"):
        logger.info("No FMP API key — generating synthetic fundamental features")
        return _generate_synthetic_fmp(tickers)

    fmp_data = {}
    import httpx

    for ticker in tickers:
        try:
            records = []

            # Earnings surprises — fetch full history (up to 20 quarters)
            resp = httpx.get(
                f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}",
                params={"apikey": api_key},
                timeout=10.0,
            )
            surprises = []
            if resp.status_code == 200:
                surprises = resp.json() or []

            # Analyst estimates — fetch history
            resp = httpx.get(
                f"https://financialmodelingprep.com/api/v3/analyst-estimates/{ticker}",
                params={"apikey": api_key, "limit": 20},
                timeout=10.0,
            )
            estimates = []
            if resp.status_code == 200:
                estimates = resp.json() or []

            # Key metrics — fetch quarterly history
            resp = httpx.get(
                f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}",
                params={"apikey": api_key, "period": "quarter", "limit": 20},
                timeout=10.0,
            )
            metrics = []
            if resp.status_code == 200:
                metrics = resp.json() or []

            # Build dated records from earnings surprises
            # Each surprise has a date — that's when the market learned it
            for i, s in enumerate(surprises):
                pub_date = s.get("date", "")
                if not pub_date:
                    continue

                actual = float(s.get("actualEarningResult", 0) or 0)
                estimated = float(s.get("estimatedEarning", 0) or 0)

                surprise_pct = 0.0
                if estimated != 0:
                    surprise_pct = (actual - estimated) / abs(estimated)

                # Beat streak: count consecutive beats from this point backward
                streak = 0
                for past in surprises[i:i + 8]:
                    a = float(past.get("actualEarningResult", 0) or 0)
                    e = float(past.get("estimatedEarning", 0) or 0)
                    if a > e:
                        streak += 1
                    else:
                        break

                record = {
                    "date": pub_date,
                    "earnings_surprise_pct": surprise_pct,
                    "earnings_beat_streak": streak,
                }

                # Find matching estimate revision (closest date <= pub_date)
                for j, est in enumerate(estimates):
                    est_date = est.get("date", "")
                    if est_date and est_date <= pub_date and j + 1 < len(estimates):
                        current_eps = float(est.get("estimatedEpsAvg", 0) or 0)
                        prior_eps = float(estimates[j + 1].get("estimatedEpsAvg", 0) or 0)
                        if prior_eps != 0:
                            record["eps_revision_pct"] = (current_eps - prior_eps) / abs(prior_eps)
                        else:
                            record["eps_revision_pct"] = 0.0
                        break

                # Find matching PE metrics (closest date <= pub_date)
                for m in metrics:
                    m_date = m.get("date", "")
                    if m_date and m_date <= pub_date:
                        pe = float(m.get("peRatio", 0) or 0)
                        fwd_pe = float(m.get("forwardPeRatio", 0) or m.get("peRatio", 0) or 0)
                        if pe > 0 and fwd_pe > 0:
                            record["fwd_trail_pe_ratio"] = fwd_pe / pe
                        else:
                            record["fwd_trail_pe_ratio"] = 1.0
                        break

                records.append(record)

            if records:
                fmp_data[ticker] = records

        except Exception as e:
            logger.debug(f"FMP fetch failed for {ticker}: {e}")
            continue

    # Cache
    os.makedirs(cache_dir, exist_ok=True)
    to_cache = dict(fmp_data)
    to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
    with open(cache_file, "w") as f:
        json.dump(to_cache, f)

    logger.info(f"FMP fundamentals: {len(fmp_data)} tickers (historical point-in-time)")
    return fmp_data


def _generate_synthetic_fmp(tickers: List[str], seed: int = 42) -> Dict[str, list]:
    """Generate synthetic FMP data with quarterly dates for offline testing."""
    rng = np.random.RandomState(seed)
    data = {}
    # Generate 5 years of quarterly records
    dates = pd.date_range("2020-01-15", periods=20, freq="QS").strftime("%Y-%m-%d").tolist()

    for ticker in tickers:
        records = []
        streak = 0
        for dt in dates:
            surprise = float(rng.normal(0.02, 0.10))
            if surprise > 0:
                streak += 1
            else:
                streak = 0
            records.append({
                "date": dt,
                "earnings_surprise_pct": surprise,
                "earnings_beat_streak": min(streak, 8),
                "eps_revision_pct": float(rng.normal(0.01, 0.05)),
                "fwd_trail_pe_ratio": float(rng.normal(0.95, 0.15)),
            })
        data[ticker] = records
    return data


def build_fmp_features(
    fmp_data: Dict[str, list],
    prices: pd.DataFrame,
) -> Dict[tuple, pd.DataFrame]:
    """
    Build FMP fundamental features for the ML model.

    BACKTEST-SAFE: Each feature value is forward-filled from its publication
    date. The model at date T only sees data published on or before T.

    In production, this naturally uses the latest available record.
    """
    if not fmp_data:
        return {}

    tickers = list(prices.columns)
    dates = prices.index
    feature_names = [
        "earnings_surprise_pct", "earnings_beat_streak",
        "eps_revision_pct", "fwd_trail_pe_ratio",
    ]

    # Build time-indexed DataFrames per feature
    raw = {f: pd.DataFrame(np.nan, index=dates, columns=tickers) for f in feature_names}

    for ticker in tickers:
        records = fmp_data.get(ticker, [])
        if not records:
            continue

        # Handle both old format (single dict) and new format (list of dated records)
        if isinstance(records, dict):
            # Legacy format: single record, no date — broadcast (production only)
            for feat in feature_names:
                if feat in records:
                    raw[feat][ticker] = records[feat]
            continue

        # New format: list of dated records — place at publication dates
        for record in records:
            pub_date = record.get("date", "")
            if not pub_date:
                continue
            try:
                pd_date = pd.Timestamp(pub_date)
            except Exception:
                continue

            # Find the first trading date on or after publication
            valid_dates = dates[dates >= pd_date]
            if len(valid_dates) == 0:
                continue
            entry_date = valid_dates[0]

            for feat in feature_names:
                if feat in record:
                    raw[feat].loc[entry_date, ticker] = record[feat]

    # Forward-fill: each value persists until the next publication
    feats = {}
    for feat in feature_names:
        df = raw[feat].ffill()
        feats[("fmp", feat)] = df
        feats[("fmp", f"cs_rank_{feat}")] = df.rank(axis=1, pct=True)

    logger.info(f"FMP features: {len(feats)} signals (point-in-time, forward-filled)")
    return feats
