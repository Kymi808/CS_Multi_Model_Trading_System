"""
Point-in-time premium features from FMP.

Unlike the old fmp_data_provider approach that fetched current snapshots
(unusable for backtest due to look-ahead), this module uses historical
endpoints with proper publication dates:

1. Insider trades — uses SEC Form 4 filingDate (when publicly known)
2. Analyst grades/ratings — uses date (when rating change was published)

All features are computed with proper PIT logic: for each backtest date,
only data with publication_date <= current_date is used. This eliminates
look-ahead bias and produces backtest features identical to live features.
"""
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/stable"
MAX_WORKERS = 3


def _fmp_get(endpoint: str, api_key: str, params: dict = None, timeout: float = 15.0):
    """Make a single FMP API call with rate limiting."""
    import httpx
    if params is None:
        params = {}
    params["apikey"] = api_key
    time.sleep(0.15)
    return httpx.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=timeout)


# =============================================================================
# 1. INSIDER TRADES (PIT via filingDate)
# =============================================================================

def fetch_insider_trades_pit(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
    limit_per_ticker: int = 1000,
) -> Dict[str, List[dict]]:
    """
    Fetch insider transactions with SEC filing dates.

    Returns: {ticker: [{filingDate, transactionDate, price, shares,
                        transactionType, isBuy, reportingName, ...}, ...]}

    Uses filingDate (SEC Form 4 public disclosure date) for PIT correctness.
    Insider knows immediately on transactionDate, but market only knows
    on filingDate (typically 1-3 days later).
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")

    cache_file = os.path.join(cache_dir, "fmp_insider_trades.json")
    cached_data = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached_data = json.load(f)
            cached_tickers = {k for k in cached_data.keys() if not k.startswith("_")}
            missing = [t for t in tickers if t not in cached_tickers]
            if not missing:
                logger.info(f"Loading cached insider trades ({len(cached_tickers)} tickers)")
                return {k: v for k, v in cached_data.items() if not k.startswith("_")}
            else:
                logger.info(f"Insider cache has {len(cached_tickers)}, fetching {len(missing)} missing")
                tickers = missing
        except Exception:
            cached_data = {}

    if not api_key or api_key in ("", "xxxxx"):
        return {k: v for k, v in cached_data.items() if not k.startswith("_")}

    def _fetch_one(ticker: str) -> Optional[List[dict]]:
        """Fetch insider trades for one ticker, paginated."""
        all_records = []
        page = 0
        while page < 5:  # max 5 pages × 100 = 500 records per ticker
            resp = _fmp_get("insider-trading/search", api_key, {
                "symbol": ticker, "limit": 100, "page": page,
            })
            if resp.status_code != 200:
                break
            data = resp.json()
            if not isinstance(data, list) or not data:
                break
            all_records.extend(data)
            if len(data) < 100:
                break
            page += 1

        # Normalize records
        cleaned = []
        for rec in all_records:
            filing_date = rec.get("filingDate", "")
            if not filing_date:
                continue
            price = rec.get("price", 0) or 0
            shares = rec.get("securitiesTransacted", 0) or 0
            acquisition = rec.get("acquisitionOrDisposition", "")
            # Only include actual market transactions (not option exercises, etc.)
            # "A" = acquired (bought), "D" = disposed (sold)
            if acquisition not in ("A", "D"):
                continue
            # Skip zero-price transactions (stock grants, option exercises)
            if price == 0:
                continue
            # Skip non-common-stock
            security = rec.get("securityName", "")
            if security and "common" not in security.lower() and "class" not in security.lower():
                continue

            cleaned.append({
                "filingDate": filing_date[:10],
                "price": float(price),
                "shares": int(shares),
                "usd_value": float(price * shares),
                "is_buy": acquisition == "A",
                "reporter": rec.get("reportingName", ""),
                "owner_type": rec.get("typeOfOwner", "").lower(),
            })
        return cleaned if cleaned else None

    all_data = {}
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            if (i + 1) % 50 == 0:
                logger.info(f"  Insider trades: {i + 1}/{len(tickers)} "
                            f"({len(all_data)} OK, {errors} errors)")
            try:
                records = future.result()
                if records:
                    all_data[ticker] = records
            except Exception:
                errors += 1

    # Merge with cache
    merged = {k: v for k, v in cached_data.items() if not k.startswith("_")}
    merged.update(all_data)
    if merged:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(merged)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    logger.info(f"Insider trades: {len(merged)} total tickers "
                f"({len(all_data)} new, {len(merged) - len(all_data)} cached)")
    return merged


def _is_exec_owner(owner_type: str) -> bool:
    """Check if owner is C-suite/senior officer."""
    ol = (owner_type or "").lower()
    return any(t in ol for t in ["chief", "ceo", "cfo", "coo", "president", "chairman"])


def build_insider_pit_features(
    insider_data: Dict[str, List[dict]],
    prices: pd.DataFrame,
    sector_map: Optional[Dict[str, str]] = None,
) -> Dict[tuple, pd.DataFrame]:
    """
    Build point-in-time insider trading features (vectorized).

    Approach:
    1. Build daily per-ticker aggregates (sum USD, count distinct reporters)
       indexed by filingDate.
    2. Reindex to prices.index with 0-fill.
    3. Rolling sum for windowed features.

    Uses rolling(21) ≈ 30 calendar days on business day index.
    """
    feats = {}
    dates = prices.index
    tickers = list(prices.columns)

    # Daily aggregates - one row per (filingDate, ticker)
    buy_usd_daily = pd.DataFrame(0.0, index=dates, columns=tickers)
    sell_usd_daily = pd.DataFrame(0.0, index=dates, columns=tickers)
    buy_count_daily = pd.DataFrame(0.0, index=dates, columns=tickers)
    sell_count_daily = pd.DataFrame(0.0, index=dates, columns=tickers)
    exec_buy_daily = pd.DataFrame(0.0, index=dates, columns=tickers)

    for ticker in tickers:
        if ticker not in insider_data:
            continue
        records = insider_data[ticker]
        if not records:
            continue
        df = pd.DataFrame(records)
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        df["is_exec"] = df["owner_type"].apply(_is_exec_owner) if "owner_type" in df.columns else False

        buys = df[df["is_buy"]]
        sells = df[~df["is_buy"]]

        # Aggregate per filing date
        if not buys.empty:
            buy_usd = buys.groupby("filingDate")["usd_value"].sum()
            buy_cnt = buys.groupby("filingDate")["reporter"].nunique()
            exec_buy = buys[buys["is_exec"]].groupby("filingDate").size() if "is_exec" in buys.columns else pd.Series(dtype=float)

            buy_usd_daily[ticker] = buy_usd.reindex(dates, fill_value=0).values
            buy_count_daily[ticker] = buy_cnt.reindex(dates, fill_value=0).values
            if not exec_buy.empty:
                exec_buy_daily[ticker] = exec_buy.reindex(dates, fill_value=0).clip(upper=1).values

        if not sells.empty:
            sell_usd = sells.groupby("filingDate")["usd_value"].sum()
            sell_cnt = sells.groupby("filingDate")["reporter"].nunique()
            sell_usd_daily[ticker] = sell_usd.reindex(dates, fill_value=0).values
            sell_count_daily[ticker] = sell_cnt.reindex(dates, fill_value=0).values

    # Rolling windows (21 business days ≈ 30 calendar days, 63 ≈ 90)
    net_buy_usd_30d = (buy_usd_daily - sell_usd_daily).rolling(21, min_periods=1).sum()
    buy_count_30d = buy_count_daily.rolling(21, min_periods=1).sum()
    sell_count_30d = sell_count_daily.rolling(21, min_periods=1).sum()
    exec_buy_90d = exec_buy_daily.rolling(63, min_periods=1).max().clip(upper=1)

    # Log-transform USD (compress heavy tails while preserving sign)
    net_buy_usd_30d_log = np.sign(net_buy_usd_30d) * np.log1p(net_buy_usd_30d.abs())
    buy_minus_sell_30d = buy_count_30d - sell_count_30d

    feats[("insider", "net_buy_usd_30d")] = net_buy_usd_30d_log
    feats[("insider", "cs_rank_net_buy_usd_30d")] = net_buy_usd_30d_log.rank(axis=1, pct=True)
    feats[("insider", "buy_count_30d")] = buy_count_30d
    feats[("insider", "sell_count_30d")] = sell_count_30d
    feats[("insider", "buy_minus_sell_count_30d")] = buy_minus_sell_30d
    feats[("insider", "cs_rank_buy_minus_sell_30d")] = buy_minus_sell_30d.rank(axis=1, pct=True)
    feats[("insider", "exec_buy_flag_90d")] = exec_buy_90d

    logger.info(f"Insider PIT features: {len(feats)} signals")
    return feats


# =============================================================================
# 2. ANALYST GRADES / RATINGS (PIT via date)
# =============================================================================

# Map grade strings to numeric scores (1=sell, 5=strong buy)
GRADE_SCORES = {
    # Strong buy (5)
    "strong buy": 5, "strongbuy": 5, "buy": 5,
    "outperform": 5, "overweight": 5,
    # Buy / Accumulate (4)
    "moderate buy": 4, "moderatebuy": 4, "accumulate": 4,
    "market outperform": 4, "sector outperform": 4, "add": 4,
    "top pick": 5,
    # Hold (3)
    "hold": 3, "neutral": 3, "market perform": 3, "sector perform": 3,
    "equal-weight": 3, "equalweight": 3, "equal weight": 3,
    "perform": 3, "peer perform": 3, "in-line": 3, "inline": 3,
    # Moderate sell (2)
    "moderate sell": 2, "moderatesell": 2, "reduce": 2,
    "underweight": 2, "underperform": 2,
    # Sell (1)
    "sell": 1, "strong sell": 1, "strongsell": 1,
}


def _grade_to_score(grade: str) -> Optional[int]:
    """Convert rating string to 1-5 score. Returns None if unrecognized."""
    if not grade:
        return None
    g = grade.lower().strip()
    return GRADE_SCORES.get(g)


def fetch_analyst_grades_pit(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
) -> Dict[str, List[dict]]:
    """
    Fetch historical analyst rating changes from FMP.

    Returns: {ticker: [{date, company, prev_grade, new_grade,
                        prev_score, new_score, is_upgrade}, ...]}

    All records have date = when the rating change was published.
    PIT-compatible by construction.
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")

    cache_file = os.path.join(cache_dir, "fmp_analyst_grades.json")
    cached_data = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached_data = json.load(f)
            cached_tickers = {k for k in cached_data.keys() if not k.startswith("_")}
            missing = [t for t in tickers if t not in cached_tickers]
            if not missing:
                logger.info(f"Loading cached analyst grades ({len(cached_tickers)} tickers)")
                return {k: v for k, v in cached_data.items() if not k.startswith("_")}
            else:
                logger.info(f"Grades cache has {len(cached_tickers)}, fetching {len(missing)} missing")
                tickers = missing
        except Exception:
            cached_data = {}

    if not api_key or api_key in ("", "xxxxx"):
        return {k: v for k, v in cached_data.items() if not k.startswith("_")}

    def _fetch_one(ticker: str) -> Optional[List[dict]]:
        """Fetch all historical grades for one ticker."""
        resp = _fmp_get("grades", api_key, {"symbol": ticker, "limit": 1000})
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None

        cleaned = []
        for rec in data:
            date = rec.get("date", "")
            prev_grade = rec.get("previousGrade", "")
            new_grade = rec.get("newGrade", "")
            if not date or not new_grade:
                continue

            prev_score = _grade_to_score(prev_grade)
            new_score = _grade_to_score(new_grade)
            if new_score is None:
                continue

            # Determine upgrade/downgrade
            is_upgrade = False
            is_downgrade = False
            if prev_score is not None:
                is_upgrade = new_score > prev_score
                is_downgrade = new_score < prev_score
            else:
                # New coverage initiation: treat as upgrade if new_score >= 4
                is_upgrade = new_score >= 4
                is_downgrade = new_score <= 2

            cleaned.append({
                "date": date[:10],
                "company": rec.get("gradingCompany", ""),
                "prev_score": prev_score,
                "new_score": new_score,
                "is_upgrade": is_upgrade,
                "is_downgrade": is_downgrade,
            })
        return cleaned if cleaned else None

    all_data = {}
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            if (i + 1) % 50 == 0:
                logger.info(f"  Analyst grades: {i + 1}/{len(tickers)} "
                            f"({len(all_data)} OK, {errors} errors)")
            try:
                records = future.result()
                if records:
                    all_data[ticker] = records
            except Exception:
                errors += 1

    merged = {k: v for k, v in cached_data.items() if not k.startswith("_")}
    merged.update(all_data)
    if merged:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(merged)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    logger.info(f"Analyst grades: {len(merged)} total tickers "
                f"({len(all_data)} new, {len(merged) - len(all_data)} cached)")
    return merged


def build_grades_pit_features(
    grades_data: Dict[str, List[dict]],
    prices: pd.DataFrame,
) -> Dict[tuple, pd.DataFrame]:
    """
    Build point-in-time analyst grades features (vectorized).

    Features:
    - upgrades_30d: count of upgrades in last 30 days
    - downgrades_30d: count of downgrades
    - net_upgrades_60d: upgrades - downgrades in last 60 days
    - consensus_score_90d: mean rating score (1-5) over last 90 days
    """
    feats = {}
    dates = prices.index
    tickers = list(prices.columns)

    # Build daily aggregates
    upgrades_daily = pd.DataFrame(0.0, index=dates, columns=tickers)
    downgrades_daily = pd.DataFrame(0.0, index=dates, columns=tickers)
    score_sum_daily = pd.DataFrame(0.0, index=dates, columns=tickers)
    score_count_daily = pd.DataFrame(0.0, index=dates, columns=tickers)

    for ticker in tickers:
        if ticker not in grades_data:
            continue
        records = grades_data[ticker]
        if not records:
            continue
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        up = df[df["is_upgrade"]]
        down = df[df["is_downgrade"]]

        if not up.empty:
            up_daily = up.groupby("date").size()
            upgrades_daily[ticker] = up_daily.reindex(dates, fill_value=0).values
        if not down.empty:
            down_daily = down.groupby("date").size()
            downgrades_daily[ticker] = down_daily.reindex(dates, fill_value=0).values

        # Score aggregates (for rolling mean)
        score_sum = df.groupby("date")["new_score"].sum()
        score_cnt = df.groupby("date").size()
        score_sum_daily[ticker] = score_sum.reindex(dates, fill_value=0).values
        score_count_daily[ticker] = score_cnt.reindex(dates, fill_value=0).values

    # Rolling windows (business days)
    upgrades_30d = upgrades_daily.rolling(21, min_periods=1).sum()
    downgrades_30d = downgrades_daily.rolling(21, min_periods=1).sum()
    net_upgrades_60d = (upgrades_daily - downgrades_daily).rolling(42, min_periods=1).sum()

    # Consensus score over 90 days = sum / count of grades in window
    score_sum_90d = score_sum_daily.rolling(63, min_periods=1).sum()
    score_count_90d = score_count_daily.rolling(63, min_periods=1).sum()
    consensus_score_90d = score_sum_90d / score_count_90d.replace(0, np.nan)

    feats[("grades", "upgrades_30d")] = upgrades_30d
    feats[("grades", "downgrades_30d")] = downgrades_30d
    feats[("grades", "net_upgrades_60d")] = net_upgrades_60d
    feats[("grades", "cs_rank_net_upgrades_60d")] = net_upgrades_60d.rank(axis=1, pct=True)
    feats[("grades", "consensus_score_90d")] = consensus_score_90d
    feats[("grades", "cs_rank_consensus_score_90d")] = consensus_score_90d.rank(axis=1, pct=True)

    logger.info(f"Analyst grades PIT features: {len(feats)} signals")
    return feats
