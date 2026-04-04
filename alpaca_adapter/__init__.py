"""
Adapter that provides yfinance-compatible function signatures using AlpacaDataProvider.

This is a drop-in replacement for CS_Multi_Model_Trading_System/data_loader.py functions:
- fetch_price_data()  -> uses Alpaca historical bars
- fetch_cross_asset_data() -> uses Alpaca ETF proxies
- fetch_news_sentiment() -> uses Alpaca news API

Functions that Alpaca does NOT cover (fundamentals, earnings dates, sectors)
are left to their original yfinance implementations for now.
A future phase will replace those with Financial Modeling Prep or similar.
"""
import asyncio
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .provider import AlpacaDataProvider, CROSS_ASSET_MAP

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync code, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
        # Already in async context — use nest_asyncio or create task
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop — create one
        return asyncio.run(coro)


def fetch_price_data(
    tickers: List[str],
    cfg,
    end_date: Optional[str] = None,
    cache_dir: str = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop-in replacement for CS system's fetch_price_data().

    Returns (prices_df, volumes_df) with DatetimeIndex and ticker columns,
    identical format to what yf.download() produces.
    """
    os.makedirs(cache_dir, exist_ok=True)
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d")
        - timedelta(days=cfg.lookback_years * 365 + 30)
    ).strftime("%Y-%m-%d")

    # Check cache first (same logic as original)
    cache_px = os.path.join(cache_dir, f"prices_{len(tickers)}.csv")
    cache_vol = os.path.join(cache_dir, f"volumes_{len(tickers)}.csv")
    if _is_cache_valid(cache_px) and _is_cache_valid(cache_vol):
        logger.info("Loading cached price data")
        return (
            pd.read_csv(cache_px, index_col=0, parse_dates=True),
            pd.read_csv(cache_vol, index_col=0, parse_dates=True),
        )

    logger.info(f"Fetching {len(tickers)} tickers via Alpaca: {start_date} to {end_date}")

    async def _fetch():
        provider = AlpacaDataProvider()
        try:
            bars = await provider.get_bars(
                symbols=tickers,
                start=start_date,
                end=end_date,
                timeframe="1Day",
                adjustment="all",
            )
            return bars
        finally:
            await provider.close()

    bars = _run_async(_fetch())

    # Convert to separate prices and volumes DataFrames
    prices_frames = {}
    volumes_frames = {}

    for symbol, bar_list in bars.items():
        if not bar_list:
            continue
        df = pd.DataFrame([
            {
                "Date": b.timestamp,
                "Close": b.close,
                "Volume": b.volume,
            }
            for b in bar_list
        ])
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        prices_frames[symbol] = df["Close"]
        volumes_frames[symbol] = df["Volume"]

    if prices_frames:
        prices = pd.DataFrame(prices_frames)
        volumes = pd.DataFrame(volumes_frames)

        if len(prices.dropna(how="all")) > 100:
            prices = prices.ffill(limit=3)
            volumes = volumes.ffill(limit=3)
            prices.to_csv(cache_px)
            volumes.to_csv(cache_vol)
            logger.info(f"Cached: {prices.shape}")
            return prices, volumes

    # Fallback: synthetic data (same as original)
    logger.warning("Alpaca fetch returned insufficient data — generating synthetic")
    from .synthetic import generate_synthetic_prices
    prices, volumes = generate_synthetic_prices(tickers, start_date, end_date)
    prices.to_csv(cache_px)
    volumes.to_csv(cache_vol)
    return prices, volumes


def fetch_cross_asset_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    Drop-in replacement for CS system's fetch_cross_asset_data().

    Fetches ETF proxies via Alpaca and maps them back to the original
    ticker names (^VIX -> VIXY, ^TNX -> TLT, etc.)
    """
    cache_file = os.path.join(cache_dir, "cross_asset.csv")
    if _is_cache_valid(cache_file):
        logger.info("Loading cached cross-asset data")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    logger.info(f"Fetching cross-asset data via Alpaca: {len(tickers)} tickers")

    # Map requested tickers to ETF equivalents
    etf_map = {}
    unmapped = []
    for t in tickers:
        if t in CROSS_ASSET_MAP:
            etf_map[t] = CROSS_ASSET_MAP[t]
        else:
            # Try as-is (might already be a valid Alpaca symbol)
            etf_map[t] = t

    etf_symbols = list(set(etf_map.values()))

    async def _fetch():
        provider = AlpacaDataProvider()
        try:
            bars = await provider.get_bars(
                symbols=etf_symbols,
                start=start_date,
                end=end_date,
                timeframe="1Day",
                adjustment="all",
            )
            return bars
        finally:
            await provider.close()

    bars = _run_async(_fetch())

    # Build DataFrame with original ticker names as columns
    frames = {}
    for original_ticker, etf_symbol in etf_map.items():
        bar_list = bars.get(etf_symbol, [])
        if bar_list:
            df = pd.DataFrame([
                {"Date": b.timestamp, "Close": b.close}
                for b in bar_list
            ])
            df.set_index("Date", inplace=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            frames[original_ticker] = df["Close"]

    if frames:
        ca = pd.DataFrame(frames)
        if len(ca.dropna(how="all")) > 100:
            ca = ca.ffill().bfill()
            os.makedirs(cache_dir, exist_ok=True)
            ca.to_csv(cache_file)
            return ca

    logger.warning("Alpaca cross-asset fetch failed — generating synthetic")
    from .synthetic import generate_synthetic_cross_asset
    ca = generate_synthetic_cross_asset(tickers, start_date, end_date)
    os.makedirs(cache_dir, exist_ok=True)
    ca.to_csv(cache_file)
    return ca


def fetch_news_sentiment(
    tickers: List[str],
    max_per_ticker: int = 10,
    cache_dir: str = "data",
) -> Dict[str, dict]:
    """
    Drop-in replacement for CS system's fetch_news_sentiment().

    Uses Alpaca news API instead of yfinance Ticker.news.
    Returns dict[ticker -> sentiment_dict] with same keys as original.
    """
    cache_file = os.path.join(cache_dir, "sentiment.json")
    if _is_json_cache_valid(cache_file):
        logger.info("Loading cached sentiment data")
        with open(cache_file) as f:
            return json.load(f)

    logger.info(f"Fetching news sentiment via Alpaca for {len(tickers)} tickers")

    # Simple financial sentiment lexicon (same approach as CS system)
    POSITIVE = {
        "beat", "exceeds", "surpass", "upgrade", "bullish", "growth", "profit",
        "gain", "rally", "surge", "boost", "strong", "record", "outperform",
        "up", "rise", "high", "positive", "optimistic", "buy",
    }
    NEGATIVE = {
        "miss", "disappoint", "downgrade", "bearish", "loss", "decline",
        "drop", "fall", "crash", "weak", "cut", "warning", "risk",
        "down", "low", "negative", "pessimistic", "sell", "layoff", "lawsuit",
    }

    async def _fetch():
        provider = AlpacaDataProvider()
        try:
            return await provider.get_news_for_symbols(
                tickers, max_per_symbol=max_per_ticker,
            )
        finally:
            await provider.close()

    news_by_symbol = _run_async(_fetch())

    sentiment = {}
    for ticker in tickers:
        articles = news_by_symbol.get(ticker, [])
        if not articles:
            sentiment[ticker] = _neutral_sentiment()
            continue

        scores = []
        for article in articles:
            text = (article.headline + " " + article.summary).lower()
            words = set(text.split())
            pos = len(words & POSITIVE)
            neg = len(words & NEGATIVE)
            total = pos + neg
            score = (pos - neg) / max(total, 1)
            scores.append(score)

        scores_arr = np.array(scores)
        sentiment[ticker] = {
            "avg_sentiment": float(scores_arr.mean()),
            "max_sentiment": float(scores_arr.max()),
            "min_sentiment": float(scores_arr.min()),
            "sentiment_std": float(scores_arr.std()) if len(scores_arr) > 1 else 0.0,
            "n_articles": len(articles),
            "positive_ratio": float((scores_arr > 0).mean()),
            "negative_ratio": float((scores_arr < 0).mean()),
        }

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(sentiment, f)
    logger.info(f"Computed sentiment for {len(sentiment)} tickers")
    return sentiment


def _neutral_sentiment() -> dict:
    return {
        "avg_sentiment": 0.0,
        "max_sentiment": 0.0,
        "min_sentiment": 0.0,
        "sentiment_std": 0.0,
        "n_articles": 0,
        "positive_ratio": 0.0,
        "negative_ratio": 0.0,
    }


def _is_cache_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, index_col=0, nrows=5)
        return len(df) >= 1 and len(df.columns) >= 1
    except Exception:
        return False


def _is_json_cache_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        return isinstance(data, dict) and len(data) > 0
    except Exception:
        return False
