"""
Data fetching: prices, volumes, fundamentals, cross-asset, earnings dates.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import os
import json
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


def fetch_price_data(
    tickers: List[str], cfg, end_date: Optional[str] = None, cache_dir: str = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(cache_dir, exist_ok=True)
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=cfg.lookback_years * 365 + 30)
    ).strftime("%Y-%m-%d")

    cache_px = os.path.join(cache_dir, f"prices_{len(tickers)}.csv")
    cache_vol = os.path.join(cache_dir, f"volumes_{len(tickers)}.csv")
    if os.path.exists(cache_px) and os.path.exists(cache_vol):
        logger.info("Loading cached price data")
        return pd.read_csv(cache_px, index_col=0, parse_dates=True), \
               pd.read_csv(cache_vol, index_col=0, parse_dates=True)

    logger.info(f"Downloading {len(tickers)} tickers: {start_date} to {end_date}")
    all_prices, all_volumes = [], []
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i+50]
        logger.info(f"  Batch {i//50+1}: {len(batch)} tickers")
        try:
            data = yf.download(batch, start=start_date, end=end_date,
                               auto_adjust=True, threads=True, progress=False)
            if len(batch) == 1:
                all_prices.append(data["Close"].to_frame(batch[0]))
                all_volumes.append(data["Volume"].to_frame(batch[0]))
            else:
                all_prices.append(data["Close"])
                all_volumes.append(data["Volume"])
        except Exception as e:
            logger.warning(f"  Batch failed: {e}")

    if not all_prices:
        raise ValueError("No data downloaded")
    prices = pd.concat(all_prices, axis=1).loc[:, ~pd.concat(all_prices, axis=1).columns.duplicated()]
    volumes = pd.concat(all_volumes, axis=1).loc[:, ~pd.concat(all_volumes, axis=1).columns.duplicated()]
    prices = prices.ffill(limit=3)
    volumes = volumes.ffill(limit=3)
    prices.to_csv(cache_px)
    volumes.to_csv(cache_vol)
    logger.info(f"Cached: {prices.shape}")
    return prices, volumes


def fetch_cross_asset_data(
    tickers: List[str], start_date: str, end_date: str, cache_dir: str = "data",
) -> pd.DataFrame:
    cache_file = os.path.join(cache_dir, "cross_asset.csv")
    if os.path.exists(cache_file):
        logger.info("Loading cached cross-asset data")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    logger.info(f"Downloading cross-asset data: {len(tickers)} tickers")
    try:
        data = yf.download(tickers, start=start_date, end=end_date,
                           auto_adjust=True, threads=True, progress=False)
        if len(tickers) == 1:
            ca = data["Close"].to_frame(tickers[0])
        else:
            ca = data["Close"]
        ca = ca.ffill().bfill()
        os.makedirs(cache_dir, exist_ok=True)
        ca.to_csv(cache_file)
        return ca
    except Exception as e:
        logger.warning(f"Cross-asset download failed: {e}")
        return pd.DataFrame()


def fetch_fundamental_data(
    tickers: List[str], cache_dir: str = "data",
) -> Dict[str, dict]:
    """Fetch fundamental data (valuation, profitability, growth, quality, analyst)."""
    cache_file = os.path.join(cache_dir, "fundamentals.json")
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        age_hours = (datetime.now().timestamp() - mtime) / 3600
        if age_hours < 24:  # Cache for 24 hours
            logger.info("Loading cached fundamentals")
            with open(cache_file) as f:
                return json.load(f)

    logger.info(f"Fetching fundamentals for {len(tickers)} tickers...")
    fundamentals = {}
    fields = [
        # Valuation
        "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
        "enterpriseToRevenue", "enterpriseToEbitda",
        # Profitability
        "returnOnEquity", "returnOnAssets", "grossMargins",
        "operatingMargins", "profitMargins",
        # Growth
        "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
        # Quality / Balance Sheet
        "debtToEquity", "currentRatio", "quickRatio",
        # Analyst
        "recommendationMean", "targetMeanPrice", "numberOfAnalystOpinions",
        # Size
        "marketCap", "beta",
        # Dividends
        "dividendYield", "payoutRatio",
        # Short interest
        "shortPercentOfFloat", "shortRatio",
    ]

    for i, ticker in enumerate(tickers):
        if i % 20 == 0 and i > 0:
            logger.info(f"  Fundamentals: {i}/{len(tickers)}")
        try:
            info = yf.Ticker(ticker).info
            fund = {}
            for f_name in fields:
                val = info.get(f_name)
                if val is not None and isinstance(val, (int, float)):
                    fund[f_name] = float(val)
            if fund:
                fundamentals[ticker] = fund
        except Exception:
            continue

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(fundamentals, f)
    logger.info(f"Fetched fundamentals for {len(fundamentals)} tickers")
    return fundamentals


def fetch_earnings_dates(
    tickers: List[str], cache_dir: str = "data",
) -> Dict[str, List[str]]:
    """Fetch recent earnings dates for post-earnings drift signals."""
    cache_file = os.path.join(cache_dir, "earnings_dates.json")
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if (datetime.now().timestamp() - mtime) / 3600 < 24:
            with open(cache_file) as f:
                return json.load(f)

    logger.info(f"Fetching earnings dates for {len(tickers)} tickers...")
    earnings = {}
    for i, ticker in enumerate(tickers):
        if i % 30 == 0 and i > 0:
            logger.info(f"  Earnings dates: {i}/{len(tickers)}")
        try:
            t = yf.Ticker(ticker)
            cal = t.earnings_dates
            if cal is not None and len(cal) > 0:
                dates = [str(d.date()) for d in cal.index[:8]]
                earnings[ticker] = dates
        except Exception:
            continue

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(earnings, f)
    logger.info(f"Fetched earnings dates for {len(earnings)} tickers")
    return earnings
