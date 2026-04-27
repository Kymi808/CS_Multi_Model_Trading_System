"""
Point-in-time earnings quality features from FMP.

Implements well-documented academic anomalies:

1. ACCRUALS (Sloan 1996) — HIGH IMPACT
   (net_income - operating_cash_flow) / total_assets
   Low accruals = earnings backed by cash = better future returns
   One of the most robust anomalies in equity markets

2. ASSET GROWTH (Cooper/Gulen/Schill 2008, Jensen 2020)
   total_assets_{t} / total_assets_{t-4q} - 1
   Low asset growth = better future returns (firms over-investing underperform)

3. SHAREHOLDER YIELD (Boudoukh et al 2007)
   -(dividends_paid + stock_repurchases) / market_cap
   High shareholder yield (returns of capital) = better future returns
   Note: dividendsPaid and stockRepurchased are negative in cash flow → we negate

All features use filingDate for PIT correctness (no look-ahead).
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
    import httpx
    if params is None:
        params = {}
    params["apikey"] = api_key
    time.sleep(0.15)
    return httpx.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=timeout)


def fetch_earnings_quality_raw(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
    n_quarters: int = 40,
) -> Dict[str, List[dict]]:
    """
    Fetch raw cash flow + balance sheet + income statement data with filingDate.

    Returns: {ticker: [{filingDate, total_assets, net_income, operating_cash_flow,
                        dividends_paid, stock_repurchases, market_cap, revenue}, ...]}

    Uses 3 endpoints per ticker (cash-flow, balance-sheet, income-statement).
    Merges them by date.
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")

    cache_file = os.path.join(cache_dir, "fmp_earnings_quality_raw.json")

    # Load cache (date-agnostic — historical data doesn't change)
    cached_data = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached_data = json.load(f)
            cached_tickers = {k for k in cached_data.keys() if not k.startswith("_")}
            missing = [t for t in tickers if t not in cached_tickers]
            if not missing:
                logger.info(f"Loading cached earnings quality data ({len(cached_tickers)} tickers)")
                return {k: v for k, v in cached_data.items() if not k.startswith("_")}
            logger.info(f"Earnings quality cache has {len(cached_tickers)}, fetching {len(missing)} missing")
            tickers = missing
        except Exception:
            cached_data = {}

    if not api_key or api_key in ("", "xxxxx"):
        return {k: v for k, v in cached_data.items() if not k.startswith("_")}

    def _fetch_one(ticker: str) -> Optional[List[dict]]:
        """Fetch and merge 3 statements for one ticker."""
        # 1. Cash flow statement
        resp = _fmp_get("cash-flow-statement", api_key, {
            "symbol": ticker, "period": "quarter", "limit": n_quarters,
        })
        if resp.status_code != 200:
            return None
        cf_data = resp.json() or []
        cf_by_date = {}
        for rec in cf_data:
            date = rec.get("date", "")
            if date:
                cf_by_date[date] = {
                    "filingDate": rec.get("filingDate", date),
                    "operatingCashFlow": rec.get("operatingCashFlow"),
                    "netIncome_cf": rec.get("netIncome"),  # from CF statement
                    "commonStockRepurchased": rec.get("commonStockRepurchased"),
                    "commonDividendsPaid": rec.get("commonDividendsPaid"),
                }

        # 2. Balance sheet
        resp = _fmp_get("balance-sheet-statement", api_key, {
            "symbol": ticker, "period": "quarter", "limit": n_quarters,
        })
        if resp.status_code != 200:
            return None
        bs_data = resp.json() or []
        bs_by_date = {}
        for rec in bs_data:
            date = rec.get("date", "")
            if date:
                bs_by_date[date] = {
                    "totalAssets": rec.get("totalAssets"),
                    "totalStockholdersEquity": rec.get("totalStockholdersEquity"),
                }

        # 3. Income statement (for marketCap proxy via shares × current price later)
        resp = _fmp_get("income-statement", api_key, {
            "symbol": ticker, "period": "quarter", "limit": n_quarters,
        })
        is_data = []
        if resp.status_code == 200:
            is_data = resp.json() or []
        is_by_date = {}
        for rec in is_data:
            date = rec.get("date", "")
            if date:
                is_by_date[date] = {
                    "revenue": rec.get("revenue"),
                    "netIncome_is": rec.get("netIncome"),
                    "weightedAverageShsOut": rec.get("weightedAverageShsOut"),
                }

        # Merge by date
        all_dates = sorted(set(cf_by_date.keys()) | set(bs_by_date.keys()) | set(is_by_date.keys()),
                           reverse=True)
        merged = []
        for date in all_dates:
            cf = cf_by_date.get(date, {})
            bs = bs_by_date.get(date, {})
            is_rec = is_by_date.get(date, {})
            merged.append({
                "date": date,
                "filingDate": cf.get("filingDate", date)[:10],
                "net_income": cf.get("netIncome_cf") or is_rec.get("netIncome_is"),
                "operating_cash_flow": cf.get("operatingCashFlow"),
                "total_assets": bs.get("totalAssets"),
                "total_equity": bs.get("totalStockholdersEquity"),
                "stock_repurchases": cf.get("commonStockRepurchased"),  # negative when buying back
                "dividends_paid": cf.get("commonDividendsPaid"),  # negative when paying
                "revenue": is_rec.get("revenue"),
                "shares_out": is_rec.get("weightedAverageShsOut"),
            })
        return merged if merged else None

    all_data = {}
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            if (i + 1) % 50 == 0:
                logger.info(f"  Earnings quality: {i + 1}/{len(tickers)} "
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

    logger.info(f"Earnings quality: {len(merged)} total tickers "
                f"({len(all_data)} new, {len(merged) - len(all_data)} cached)")
    return merged


def build_earnings_quality_features(
    raw_data: Dict[str, List[dict]],
    prices: pd.DataFrame,
) -> Dict[tuple, pd.DataFrame]:
    """
    Build PIT earnings quality features from raw fundamental data.

    For each date T, uses the latest filing with filingDate <= T (forward fill).
    """
    feats = {}
    dates = prices.index
    tickers = list(prices.columns)

    # Build per-feature (date × ticker) DataFrames
    accruals_df = pd.DataFrame(np.nan, index=dates, columns=tickers)
    asset_growth_df = pd.DataFrame(np.nan, index=dates, columns=tickers)
    shareholder_yield_df = pd.DataFrame(np.nan, index=dates, columns=tickers)

    for ticker in tickers:
        if ticker not in raw_data:
            continue
        records = raw_data[ticker]
        if not records or len(records) < 2:
            continue

        # Sort by filing date
        df = pd.DataFrame(records)
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        df = df.sort_values("filingDate").reset_index(drop=True)

        # Compute per-quarter features
        accruals_series = []
        asset_growth_series = []
        shareholder_yield_series = []

        for i, row in df.iterrows():
            fd = row["filingDate"]
            ni = row.get("net_income")
            ocf = row.get("operating_cash_flow")
            ta = row.get("total_assets")
            shares = row.get("shares_out")

            # Accruals: (net_income - operating_cash_flow) / total_assets
            if ni is not None and ocf is not None and ta is not None and ta > 0:
                accruals = (ni - ocf) / ta
                accruals_series.append((fd, accruals))

            # Asset growth (4-quarter lookback)
            if i >= 4 and ta is not None and ta > 0:
                prev_ta = df.iloc[i - 4].get("total_assets")
                if prev_ta is not None and prev_ta > 0:
                    asset_growth = (ta / prev_ta) - 1
                    asset_growth_series.append((fd, asset_growth))

            # Shareholder yield (trailing 4 quarters)
            if i >= 3:  # need 4 quarters of data
                trailing_divs = 0
                trailing_buybacks = 0
                valid = True
                for j in range(4):
                    r = df.iloc[i - j]
                    d = r.get("dividends_paid")
                    b = r.get("stock_repurchases")
                    if d is None and b is None:
                        valid = False
                        break
                    trailing_divs += (d or 0)
                    trailing_buybacks += (b or 0)
                # FMP returns these as negative numbers → flip sign for "returned to shareholders"
                returned = -(trailing_divs + trailing_buybacks)
                # Get market cap estimate: use current price × shares at filing date
                if valid and shares is not None and shares > 0 and returned > 0:
                    # Find price nearest filingDate
                    if ticker in prices.columns:
                        px_before = prices[ticker].loc[:fd].dropna()
                        if len(px_before) > 0:
                            price = px_before.iloc[-1]
                            market_cap = price * shares
                            if market_cap > 0:
                                sh_yield = returned / market_cap
                                # Clip outliers
                                if -0.5 < sh_yield < 1.0:
                                    shareholder_yield_series.append((fd, sh_yield))

        # Forward-fill each series across prices.index
        def _fill(series_list, df_out):
            if not series_list:
                return
            s = pd.Series(dict(series_list)).sort_index()
            # Reindex to prices dates with forward-fill
            aligned = s.reindex(dates, method="ffill")
            df_out[ticker] = aligned

        _fill(accruals_series, accruals_df)
        _fill(asset_growth_series, asset_growth_df)
        _fill(shareholder_yield_series, shareholder_yield_df)

    # Build features
    if accruals_df.notna().sum().sum() > 0:
        feats[("quality", "accruals")] = accruals_df  # lower = better (raw)
        feats[("quality", "cs_rank_accruals_inv")] = (-accruals_df).rank(axis=1, pct=True)  # higher rank = lower accruals = better

    if asset_growth_df.notna().sum().sum() > 0:
        feats[("quality", "asset_growth_yoy")] = asset_growth_df
        feats[("quality", "cs_rank_asset_growth_inv")] = (-asset_growth_df).rank(axis=1, pct=True)  # lower growth = better

    if shareholder_yield_df.notna().sum().sum() > 0:
        feats[("quality", "shareholder_yield")] = shareholder_yield_df
        feats[("quality", "cs_rank_shareholder_yield")] = shareholder_yield_df.rank(axis=1, pct=True)

    logger.info(f"Earnings quality PIT features: {len(feats)} signals")
    return feats
