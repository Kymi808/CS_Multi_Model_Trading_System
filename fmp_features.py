"""
Point-in-time fundamental features from Financial Modeling Prep.

Two modes:
1. LIVE (fetch_fmp_fundamental_data): Uses TTM endpoints for current snapshot.
   Fast, 4 calls per ticker. Used by signal_generator.py for daily trading.

2. BACKTEST (fetch_fmp_historical_fundamentals): Uses quarterly historical
   endpoints with filingDate for true point-in-time. Each quarter's data
   is only visible after the SEC filing date. Used by backtest/retrain.

Also:
3. ALPHA (fetch_fmp_fundamentals): Earnings surprises + analyst estimates.
   Supplemental alpha features (EPS revision, beat streak, etc.)

Reference:
- Chan, Jegadeesh & Lakonishok (1996), "Momentum Strategies"
- Bernard & Thomas (1989), "Post-Earnings-Announcement Drift"
"""
import json
import os
import logging
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Suppress httpx request logging — it leaks API keys in URLs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

FMP_BASE = "https://financialmodelingprep.com/stable"


def _fmp_get(endpoint: str, api_key: str, params: dict = None, timeout: float = 10.0):
    """Make a single FMP /stable/ API call with rate-limit pause."""
    import httpx, time
    if params is None:
        params = {}
    params["apikey"] = api_key
    time.sleep(0.15)  # ~6 req/s to stay well under 750/min
    return httpx.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=timeout)


# ---------------------------------------------------------------------------
# 1. LIVE: TTM fundamentals (current snapshot, yfinance-compatible format)
# ---------------------------------------------------------------------------

# Field mappings verified against actual API responses 2026-04-07
_RATIOS_MAP = {
    "priceToEarningsRatioTTM": "trailingPE",
    "priceToBookRatioTTM": "priceToBook",
    "priceToSalesRatioTTM": "priceToSalesTrailing12Months",
    "enterpriseValueMultipleTTM": "enterpriseToEbitda",
    "grossProfitMarginTTM": "grossMargins",
    "operatingProfitMarginTTM": "operatingMargins",
    "netProfitMarginTTM": "profitMargins",
    "debtToEquityRatioTTM": "debtToEquity",
    "currentRatioTTM": "currentRatio",
    "quickRatioTTM": "quickRatio",
    "dividendYieldTTM": "dividendYield",
    "dividendPayoutRatioTTM": "payoutRatio",
}

_KEY_METRICS_MAP = {
    "returnOnEquityTTM": "returnOnEquity",
    "returnOnAssetsTTM": "returnOnAssets",
    "evToEBITDATTM": "enterpriseToEbitda",
    "evToSalesTTM": "enterpriseToRevenue",
}

_GROWTH_MAP = {
    "revenueGrowth": "revenueGrowth",
    "epsgrowth": "earningsGrowth",
    "netIncomeGrowth": "earningsQuarterlyGrowth",
}

# Historical quarterly field mappings (non-TTM endpoints)
_HIST_RATIOS_MAP = {
    "priceToEarningsRatio": "trailingPE",
    "priceToBookRatio": "priceToBook",
    "priceToSalesRatio": "priceToSalesTrailing12Months",
    "enterpriseValueMultiple": "enterpriseToEbitda",
    "grossProfitMargin": "grossMargins",
    "operatingProfitMargin": "operatingMargins",
    "netProfitMargin": "profitMargins",
    "debtToEquityRatio": "debtToEquity",
    "currentRatio": "currentRatio",
    "quickRatio": "quickRatio",
    "dividendYield": "dividendYield",
    "dividendPayoutRatio": "payoutRatio",
}

_HIST_KEY_METRICS_MAP = {
    "returnOnEquity": "returnOnEquity",
    "returnOnAssets": "returnOnAssets",
    "evToEBITDA": "enterpriseToEbitda",
    "evToSales": "enterpriseToRevenue",
    "earningsYield": "earningsYield",
    "marketCap": "marketCap",
}

_HIST_GROWTH_MAP = {
    "revenueGrowth": "revenueGrowth",
    "epsgrowth": "earningsGrowth",
    "netIncomeGrowth": "earningsQuarterlyGrowth",
}


def fetch_fmp_fundamental_data(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
) -> Dict[str, dict]:
    """
    Fetch current TTM fundamentals from FMP (drop-in yfinance replacement).

    Returns: {ticker: {trailingPE, returnOnEquity, grossMargins, ...}}
    Used for LIVE trading where current data is correct by definition.
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")

    cache_file = os.path.join(cache_dir, "fmp_fundamentals_v2.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_date") == datetime.now().strftime("%Y-%m-%d"):
                logger.info("Loading cached FMP fundamentals (v2)")
                return {k: v for k, v in cached.items() if k != "_date"}
        except Exception:
            pass

    if not api_key or api_key in ("", "xxxxx"):
        return {}

    def _fetch_one_ttm(ticker: str) -> Optional[dict]:
        """Fetch TTM fundamentals for one ticker (3-4 API calls)."""
        fund = {}
        # 1. TTM Ratios
        resp = _fmp_get("ratios-ttm", api_key, {"symbol": ticker})
        if resp.status_code in (403, 402):
            return None
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict):
                for fmp_field, yf_field in _RATIOS_MAP.items():
                    val = data.get(fmp_field)
                    if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                        fund[yf_field] = float(val)

        # 2. Key Metrics TTM
        resp = _fmp_get("key-metrics-ttm", api_key, {"symbol": ticker})
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict):
                for fmp_field, yf_field in _KEY_METRICS_MAP.items():
                    if yf_field not in fund:
                        val = data.get(fmp_field)
                        if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                            fund[yf_field] = float(val)
                mc = data.get("marketCap")
                if mc and "marketCap" not in fund:
                    fund["marketCap"] = float(mc)

        # 3. Financial Growth
        resp = _fmp_get("financial-growth", api_key, {"symbol": ticker, "period": "quarter", "limit": 1})
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict):
                for fmp_field, yf_field in _GROWTH_MAP.items():
                    val = data.get(fmp_field)
                    if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                        fund[yf_field] = float(val)

        # 4. Profile (beta, market cap fallback)
        if "beta" not in fund or "marketCap" not in fund:
            resp = _fmp_get("profile", api_key, {"symbol": ticker})
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    data = data[0]
                if isinstance(data, dict):
                    for field in ("beta", "marketCap"):
                        if field not in fund:
                            val = data.get(field)
                            if val is not None and isinstance(val, (int, float)):
                                fund[field] = float(val)

        return fund if fund else None

    # Fetch all tickers concurrently (15 threads, each does 3-4 API calls)
    MAX_WORKERS = 3
    fundamentals = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_ticker = {pool.submit(_fetch_one_ttm, t): t for t in tickers}
        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            if (i + 1) % 50 == 0:
                logger.info(f"  FMP TTM: {i + 1}/{len(tickers)} "
                            f"({len(fundamentals)} OK, {errors} errors)")
            try:
                data = future.result()
                if data:
                    fundamentals[ticker] = data
            except Exception as e:
                errors += 1
                logger.debug(f"FMP TTM failed for {ticker}: {e}")
                if errors > len(tickers) * 0.2 and errors > 10:
                    logger.error("FMP TTM: >20% failures, aborting")
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    if fundamentals:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(fundamentals)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    logger.info(f"FMP fundamentals: {len(fundamentals)}/{len(tickers)} tickers")
    return fundamentals


# ---------------------------------------------------------------------------
# 2. BACKTEST: Historical quarterly fundamentals with filing dates
# ---------------------------------------------------------------------------

def fetch_fmp_historical_fundamentals(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
    n_quarters: int = 40,  # 10 years of quarterly data (matches 8-year lookback + 2yr buffer)
) -> Dict[str, List[dict]]:
    """
    Fetch HISTORICAL quarterly fundamentals with SEC filing dates.

    Returns: {ticker: [{date, filingDate, trailingPE, ROE, ...}, ...]}

    Each record has a filingDate — the date the data became PUBLIC.
    In backtest, only use records where filingDate <= current_date.
    This eliminates look-ahead bias.

    Uses 3 endpoints per ticker: income-statement (for filingDate),
    ratios (for PE/margins/etc), key-metrics (for ROE/ROA/mcap).
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")

    cache_file = os.path.join(cache_dir, "fmp_historical_quarterly.json")

    cached_data = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached_data = json.load(f)
            # Date-agnostic: historical fundamentals don't change, always use cache if present
            cached_tickers = {k for k in cached_data.keys() if not k.startswith("_")}
            if cached_tickers:
                missing = [t for t in tickers if t not in cached_tickers]
                if not missing:
                    logger.info(f"Loading cached FMP historical fundamentals ({len(cached_tickers)} tickers)")
                    return {k: v for k, v in cached_data.items() if not k.startswith("_")}
                else:
                    logger.info(f"Cache has {len(cached_tickers)} tickers, fetching {len(missing)} missing...")
                    tickers = missing  # only fetch the missing ones
        except Exception:
            cached_data = {}

    if not api_key or api_key in ("", "xxxxx"):
        return {k: v for k, v in cached_data.items() if k != "_date"}

    def _fetch_one_historical(ticker: str) -> Optional[List[dict]]:
        """Fetch all historical quarterly data for one ticker (4 API calls)."""
        # 1. Income statement for filingDate
        resp = _fmp_get("income-statement", api_key, {
            "symbol": ticker, "period": "quarter", "limit": n_quarters,
        })
        if resp.status_code in (403, 402):
            return None

        filing_dates = {}
        if resp.status_code == 200:
            for rec in resp.json() or []:
                d = rec.get("date")
                fd = rec.get("filingDate")
                if d and fd:
                    filing_dates[d] = fd
        if not filing_dates:
            return None

        # 2. Historical ratios
        resp = _fmp_get("ratios", api_key, {
            "symbol": ticker, "period": "quarter", "limit": n_quarters,
        })
        ratios_by_date = {}
        if resp.status_code == 200:
            for rec in resp.json() or []:
                ratios_by_date[rec.get("date", "")] = rec

        # 3. Historical key-metrics
        resp = _fmp_get("key-metrics", api_key, {
            "symbol": ticker, "period": "quarter", "limit": n_quarters,
        })
        metrics_by_date = {}
        if resp.status_code == 200:
            for rec in resp.json() or []:
                metrics_by_date[rec.get("date", "")] = rec

        # 4. Financial growth
        resp = _fmp_get("financial-growth", api_key, {
            "symbol": ticker, "period": "quarter", "limit": n_quarters,
        })
        growth_by_date = {}
        if resp.status_code == 200:
            for rec in resp.json() or []:
                growth_by_date[rec.get("date", "")] = rec

        # Build dated records
        records = []
        for date, filing_date in sorted(filing_dates.items()):
            fund = {"date": date, "filingDate": filing_date}

            r = ratios_by_date.get(date, {})
            for fmp_field, yf_field in _HIST_RATIOS_MAP.items():
                val = r.get(fmp_field)
                if val is not None and isinstance(val, (int, float)):
                    fund[yf_field] = float(val)

            m = metrics_by_date.get(date, {})
            for fmp_field, yf_field in _HIST_KEY_METRICS_MAP.items():
                if yf_field not in fund:
                    val = m.get(fmp_field)
                    if val is not None and isinstance(val, (int, float)):
                        fund[yf_field] = float(val)

            g = growth_by_date.get(date, {})
            for fmp_field, yf_field in _HIST_GROWTH_MAP.items():
                val = g.get(fmp_field)
                if val is not None and isinstance(val, (int, float)):
                    fund[yf_field] = float(val)

            if len(fund) > 3:
                records.append(fund)

        return records if records else None

    # Fetch concurrently (5 threads — each does 4 API calls = ~20 concurrent)
    # Conservative to stay under 750 req/min FMP Premium limit
    MAX_WORKERS = 3
    all_data = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_ticker = {pool.submit(_fetch_one_historical, t): t for t in tickers}
        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            if (i + 1) % 50 == 0:
                logger.info(f"  FMP historical: {i + 1}/{len(tickers)} "
                            f"({len(all_data)} OK, {errors} errors)")
            try:
                records = future.result()
                if records:
                    all_data[ticker] = records
            except Exception as e:
                errors += 1
                logger.debug(f"FMP historical failed for {ticker}: {e}")
                if errors > len(tickers) * 0.2 and errors > 10:
                    logger.error("FMP historical: >20% failures, aborting")
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    # Merge newly fetched data with existing cache (incremental cache)
    merged = {k: v for k, v in cached_data.items() if not k.startswith("_")}
    merged.update(all_data)

    if merged:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(merged)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    logger.info(f"FMP historical: {len(merged)} total tickers in cache "
                f"({len(all_data)} new, {len(merged) - len(all_data)} from cache), "
                f"~{n_quarters} quarters each")

    return merged


def get_pit_fundamentals(
    historical_data: Dict[str, List[dict]],
    as_of_date: str,
) -> Dict[str, dict]:
    """
    Get point-in-time fundamentals as of a specific date.

    For each ticker, returns the latest quarterly record whose filingDate
    is on or before as_of_date. This ensures no look-ahead bias.

    Args:
        historical_data: output of fetch_fmp_historical_fundamentals()
        as_of_date: "YYYY-MM-DD" string

    Returns: {ticker: {trailingPE, ROE, ...}} — same format as yfinance
    """
    result = {}
    for ticker, records in historical_data.items():
        # Find latest record filed on or before as_of_date
        best = None
        for rec in records:
            fd = rec.get("filingDate", "")
            if fd and fd <= as_of_date:
                if best is None or fd > best.get("filingDate", ""):
                    best = rec
        if best:
            # Strip metadata, return only fundamental fields
            fund = {k: v for k, v in best.items() if k not in ("date", "filingDate")}
            if fund:
                result[ticker] = fund
    return result


# ---------------------------------------------------------------------------
# 3. ALPHA: Earnings surprises + analyst estimates (supplemental)
# ---------------------------------------------------------------------------

def fetch_fmp_fundamentals(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
) -> Dict[str, list]:
    """
    Fetch earnings surprise + analyst estimate alpha features.

    Returns: {ticker: [{date, earnings_surprise_pct, beat_streak, ...}]}
    Uses /stable/earnings endpoint (available on Premium).
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")

    cache_file = os.path.join(cache_dir, "fmp_fundamentals_hist.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            # Date-agnostic: earnings surprises are historical, always use cache
            non_meta = {k: v for k, v in cached.items() if not k.startswith("_")}
            if non_meta:
                logger.info(f"Loading cached FMP alpha features ({len(non_meta)} tickers)")
                return non_meta
        except Exception:
            pass

    if not api_key or api_key in ("", "xxxxx"):
        logger.info("No FMP API key — skipping earnings alpha features")
        return {}

    def _fetch_one_earnings(ticker: str) -> Optional[list]:
        try:
            resp = _fmp_get("earnings", api_key, {"symbol": ticker})
            if resp.status_code in (403, 402, 404):
                return None

            earnings = (resp.json() or []) if resp.status_code == 200 else []
            if not earnings:
                return None

            records = []
            for i, e in enumerate(earnings):
                pub_date = e.get("date", "")
                if not pub_date:
                    continue

                actual = e.get("epsActual")
                estimated = e.get("epsEstimated")
                if actual is None or estimated is None:
                    continue

                actual = float(actual)
                estimated = float(estimated)

                surprise_pct = 0.0
                if estimated != 0:
                    surprise_pct = (actual - estimated) / abs(estimated)

                streak = 0
                for past in earnings[i:i + 8]:
                    a = past.get("epsActual")
                    est = past.get("epsEstimated")
                    if a is not None and est is not None and float(a) > float(est):
                        streak += 1
                    else:
                        break

                record = {
                    "date": pub_date,
                    "earnings_surprise_pct": surprise_pct,
                    "earnings_beat_streak": streak,
                }

                if i + 1 < len(earnings):
                    prior_est = earnings[i + 1].get("epsEstimated")
                    if prior_est is not None and estimated != 0 and float(prior_est) != 0:
                        record["eps_revision_pct"] = (estimated - float(prior_est)) / abs(float(prior_est))

                records.append(record)

            return records if records else None
        except Exception:
            return None

    # Parallel fetch
    fmp_data = {}
    errors = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        future_to_ticker = {pool.submit(_fetch_one_earnings, t): t for t in tickers}
        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            if (i + 1) % 50 == 0:
                logger.info(f"  FMP earnings: {i + 1}/{len(tickers)} ({len(fmp_data)} OK)")
            try:
                records = future.result()
                if records:
                    fmp_data[ticker] = records
            except Exception:
                errors += 1
                if errors > len(tickers) * 0.2 and errors > 10:
                    logger.error("FMP earnings: >20% failures, aborting")
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    if fmp_data:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(fmp_data)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    logger.info(f"FMP alpha features: {len(fmp_data)} tickers")
    return fmp_data


def _generate_synthetic_fmp(tickers: List[str], seed: int = 42) -> Dict[str, list]:
    """Generate synthetic FMP data for offline testing."""
    rng = np.random.RandomState(seed)
    data = {}
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
            })
        data[ticker] = records
    return data


def build_fmp_features(
    fmp_data: Dict[str, list],
    prices: pd.DataFrame,
) -> Dict[tuple, pd.DataFrame]:
    """Build FMP alpha features (earnings surprise, beat streak, EPS revision)."""
    if not fmp_data:
        return {}

    tickers = list(prices.columns)
    dates = prices.index
    feature_names = ["earnings_surprise_pct", "earnings_beat_streak", "eps_revision_pct"]

    raw = {f: pd.DataFrame(np.nan, index=dates, columns=tickers) for f in feature_names}

    for ticker in tickers:
        records = fmp_data.get(ticker, [])
        if not records:
            continue
        if isinstance(records, dict):
            for feat in feature_names:
                if feat in records:
                    raw[feat][ticker] = records[feat]
            continue

        for record in records:
            pub_date = record.get("date", "")
            if not pub_date:
                continue
            try:
                pd_date = pd.Timestamp(pub_date)
            except Exception:
                continue
            valid_dates = dates[dates >= pd_date]
            if len(valid_dates) == 0:
                continue
            entry_date = valid_dates[0]
            for feat in feature_names:
                if feat in record:
                    raw[feat].loc[entry_date, ticker] = record[feat]

    feats = {}
    for feat in feature_names:
        df = raw[feat].ffill()
        feats[("fmp", feat)] = df
        feats[("fmp", f"cs_rank_{feat}")] = df.rank(axis=1, pct=True)

    logger.info(f"FMP alpha features: {len(feats)} signals (point-in-time)")
    return feats
