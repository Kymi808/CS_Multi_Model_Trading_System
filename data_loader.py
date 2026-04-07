"""
Data fetching: prices, volumes, fundamentals, cross-asset, earnings dates.
Falls back to synthetic data generation when network is unavailable.
"""
import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# In production, refuse to trade on synthetic data.
# Set TRADING_ENV=paper or TRADING_ENV=live to enable this guard.
PRODUCTION_MODE = os.environ.get("TRADING_ENV", "").lower() in ("production", "live", "paper")


def _is_cache_valid(path: str, min_rows: int = 10) -> bool:
    """Check if a CSV cache file exists and has meaningful data."""
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, index_col=0, nrows=5)
        return len(df) >= 1 and len(df.columns) >= 1
    except Exception:
        return False


def _is_json_cache_valid(path: str) -> bool:
    """Check if a JSON cache file exists and has meaningful data."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        return isinstance(data, dict) and len(data) > 0
    except Exception:
        return False


def _generate_synthetic_prices(
    tickers: List[str], start_date: str, end_date: str, seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic synthetic price and volume data for backtesting."""
    logger.info(f"Generating synthetic price data for {len(tickers)} tickers")
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)

    # Sector-like groupings for correlated returns
    n_tickers = len(tickers)
    n_days = len(dates)

    # Market factor + sector factors + idiosyncratic
    market_returns = rng.normal(0.0004, 0.01, n_days)  # ~10% annual, ~16% vol

    # Generate correlated returns per ticker
    all_returns = np.zeros((n_days, n_tickers))
    for i in range(n_tickers):
        beta = 0.5 + rng.random() * 1.0  # beta between 0.5 and 1.5
        idio_vol = 0.005 + rng.random() * 0.015  # 0.5% to 2% daily idio vol
        drift = rng.normal(0.0002, 0.0003)  # slight positive drift
        idio = rng.normal(drift, idio_vol, n_days)
        all_returns[:, i] = beta * market_returns + idio

    # Convert to prices (start at realistic levels)
    start_prices = 50 + rng.random(n_tickers) * 400  # $50 to $450
    prices_arr = np.zeros((n_days, n_tickers))
    prices_arr[0] = start_prices
    for t in range(1, n_days):
        prices_arr[t] = prices_arr[t-1] * (1 + all_returns[t])

    prices = pd.DataFrame(prices_arr, index=dates, columns=tickers)

    # Generate volumes (correlated with abs returns)
    base_volumes = (1e6 + rng.random(n_tickers) * 9e6)  # 1M to 10M shares
    vol_noise = rng.lognormal(0, 0.3, (n_days, n_tickers))
    abs_ret_factor = 1 + 5 * np.abs(all_returns)  # higher volume on big moves
    volumes_arr = base_volumes * vol_noise * abs_ret_factor
    volumes = pd.DataFrame(volumes_arr.astype(int), index=dates, columns=tickers)

    logger.info(f"Synthetic data: {prices.shape[0]} days, {prices.shape[1]} tickers")
    return prices, volumes


def _generate_synthetic_cross_asset(
    tickers: List[str], start_date: str, end_date: str, seed: int = 123,
) -> pd.DataFrame:
    """Generate synthetic cross-asset data (VIX, yields, commodities, ETFs)."""
    logger.info(f"Generating synthetic cross-asset data for {len(tickers)} tickers")
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    data = {}
    for ticker in tickers:
        if "VIX" in ticker:
            # VIX: mean-reverting around 18, range 10-40
            vix = np.zeros(n_days)
            vix[0] = 18.0
            for t in range(1, n_days):
                vix[t] = vix[t-1] + 0.05 * (18 - vix[t-1]) + rng.normal(0, 1.2)
                vix[t] = max(10, min(45, vix[t]))
            data[ticker] = vix
        elif "TNX" in ticker or "IRX" in ticker:
            # Yields: mean-reverting
            level = 4.0 if "TNX" in ticker else 5.0
            y = np.zeros(n_days)
            y[0] = level
            for t in range(1, n_days):
                y[t] = y[t-1] + 0.02 * (level - y[t-1]) + rng.normal(0, 0.05)
                y[t] = max(0.5, min(8.0, y[t]))
            data[ticker] = y
        elif ticker in ("GLD", "USO", "UUP"):
            # Commodities / dollar: random walk with drift
            start = {"GLD": 180, "USO": 70, "UUP": 28}.get(ticker, 100)
            returns = rng.normal(0.0002, 0.008, n_days)
            px = start * np.cumprod(1 + returns)
            data[ticker] = px
        elif ticker in ("HYG", "LQD", "TLT"):
            # Bonds
            start = {"HYG": 75, "LQD": 110, "TLT": 100}.get(ticker, 100)
            returns = rng.normal(0.0001, 0.005, n_days)
            px = start * np.cumprod(1 + returns)
            data[ticker] = px
        elif ticker == "^GSPC":
            returns = rng.normal(0.0004, 0.01, n_days)
            px = 4500 * np.cumprod(1 + returns)
            data[ticker] = px
        elif ticker in ("IWM", "QQQ"):
            start = {"IWM": 200, "QQQ": 380}.get(ticker, 200)
            returns = rng.normal(0.0003, 0.012, n_days)
            px = start * np.cumprod(1 + returns)
            data[ticker] = px
        else:
            # Sector ETFs
            start = 50 + rng.random() * 100
            returns = rng.normal(0.0003, 0.009, n_days)
            px = start * np.cumprod(1 + returns)
            data[ticker] = px

    return pd.DataFrame(data, index=dates)


def _generate_synthetic_fundamentals(
    tickers: List[str], seed: int = 42,
) -> Dict[str, dict]:
    """Generate synthetic fundamental data."""
    logger.info(f"Generating synthetic fundamentals for {len(tickers)} tickers")
    rng = np.random.RandomState(seed)
    fundamentals = {}
    for ticker in tickers:
        fundamentals[ticker] = {
            "trailingPE": float(10 + rng.random() * 40),
            "forwardPE": float(8 + rng.random() * 35),
            "priceToBook": float(1 + rng.random() * 15),
            "priceToSalesTrailing12Months": float(0.5 + rng.random() * 20),
            "enterpriseToRevenue": float(1 + rng.random() * 15),
            "enterpriseToEbitda": float(5 + rng.random() * 25),
            "returnOnEquity": float(rng.normal(0.15, 0.1)),
            "returnOnAssets": float(rng.normal(0.08, 0.05)),
            "grossMargins": float(0.2 + rng.random() * 0.6),
            "operatingMargins": float(0.05 + rng.random() * 0.3),
            "profitMargins": float(0.02 + rng.random() * 0.25),
            "revenueGrowth": float(rng.normal(0.08, 0.15)),
            "earningsGrowth": float(rng.normal(0.10, 0.25)),
            "earningsQuarterlyGrowth": float(rng.normal(0.05, 0.3)),
            "debtToEquity": float(20 + rng.random() * 200),
            "currentRatio": float(0.5 + rng.random() * 3),
            "quickRatio": float(0.3 + rng.random() * 2.5),
            "recommendationMean": float(1.5 + rng.random() * 3),
            "targetMeanPrice": float(50 + rng.random() * 400),
            "numberOfAnalystOpinions": float(5 + rng.randint(0, 35)),
            "marketCap": float(1e10 + rng.random() * 2e12),
            "beta": float(0.5 + rng.random() * 1.5),
            "dividendYield": float(rng.random() * 0.04),
            "payoutRatio": float(rng.random() * 0.7),
            "shortPercentOfFloat": float(rng.random() * 0.1),
            "shortRatio": float(1 + rng.random() * 8),
        }
    return fundamentals


def _generate_synthetic_earnings(
    tickers: List[str], seed: int = 42,
) -> Dict[str, List[str]]:
    """Generate synthetic earnings dates."""
    logger.info(f"Generating synthetic earnings dates for {len(tickers)} tickers")
    rng = np.random.RandomState(seed)
    earnings = {}
    base = datetime(2025, 1, 15)
    for ticker in tickers:
        offset = rng.randint(0, 30)
        dates = []
        for q in range(8):
            d = base + timedelta(days=offset + q * 91)
            dates.append(str(d.date()))
        earnings[ticker] = dates
    return earnings


def _generate_synthetic_sectors(tickers: List[str]) -> Dict[str, str]:
    """Generate synthetic sector mappings."""
    sectors = [
        "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
        "Communication Services", "Industrials", "Consumer Defensive",
        "Energy", "Utilities", "Real Estate", "Basic Materials",
    ]
    sector_map = {}
    # Assign sectors in a deterministic rotating pattern
    for i, ticker in enumerate(tickers):
        sector_map[ticker] = sectors[i % len(sectors)]
    return sector_map


def _generate_synthetic_sentiment(
    tickers: List[str], seed: int = 42,
) -> Dict[str, dict]:
    """Generate synthetic sentiment data."""
    logger.info(f"Generating synthetic sentiment for {len(tickers)} tickers")
    rng = np.random.RandomState(seed)
    sentiment = {}
    for ticker in tickers:
        avg = float(rng.normal(0.05, 0.2))
        sentiment[ticker] = {
            "avg_sentiment": avg,
            "max_sentiment": float(avg + abs(rng.normal(0, 0.3))),
            "min_sentiment": float(avg - abs(rng.normal(0, 0.3))),
            "sentiment_std": float(abs(rng.normal(0.15, 0.05))),
            "n_articles": int(3 + rng.randint(0, 8)),
            "positive_ratio": float(0.3 + rng.random() * 0.4),
            "negative_ratio": float(0.1 + rng.random() * 0.3),
        }
    return sentiment


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
    if _is_cache_valid(cache_px) and _is_cache_valid(cache_vol):
        logger.info("Loading cached price data")
        return pd.read_csv(cache_px, index_col=0, parse_dates=True), \
               pd.read_csv(cache_vol, index_col=0, parse_dates=True)

    logger.info(f"Downloading {len(tickers)} tickers: {start_date} to {end_date}")
    all_prices, all_volumes = [], []
    try:
        import yfinance as yf
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
    except Exception as e:
        logger.warning(f"yfinance import or download failed: {e}")

    if all_prices:
        prices = pd.concat(all_prices, axis=1).loc[:, ~pd.concat(all_prices, axis=1).columns.duplicated()]
        volumes = pd.concat(all_volumes, axis=1).loc[:, ~pd.concat(all_volumes, axis=1).columns.duplicated()]
        # Check if we got actual data (not empty frames)
        if len(prices.dropna(how="all")) > 100:
            prices = prices.ffill(limit=3)
            volumes = volumes.ffill(limit=3)
            prices.to_csv(cache_px)
            volumes.to_csv(cache_vol)
            logger.info(f"Cached: {prices.shape}")
            return prices, volumes

    # Fallback: generate synthetic data
    if PRODUCTION_MODE:
        raise RuntimeError(
            "Price data fetch failed and TRADING_ENV is set to production/paper/live. "
            "Refusing to use synthetic data. Check network connectivity and API keys."
        )
    logger.warning("Network unavailable — generating synthetic price data for offline comparison")
    prices, volumes = _generate_synthetic_prices(tickers, start_date, end_date)
    prices.to_csv(cache_px)
    volumes.to_csv(cache_vol)
    return prices, volumes


def fetch_cross_asset_data(
    tickers: List[str], start_date: str, end_date: str, cache_dir: str = "data",
) -> pd.DataFrame:
    cache_file = os.path.join(cache_dir, "cross_asset.csv")
    if _is_cache_valid(cache_file):
        logger.info("Loading cached cross-asset data")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    logger.info(f"Downloading cross-asset data: {len(tickers)} tickers")
    try:
        import yfinance as yf
        data = yf.download(tickers, start=start_date, end=end_date,
                           auto_adjust=True, threads=True, progress=False)
        if len(tickers) == 1:
            ca = data["Close"].to_frame(tickers[0])
        else:
            ca = data["Close"]
        if len(ca.dropna(how="all")) > 100:
            ca = ca.ffill().bfill()
            os.makedirs(cache_dir, exist_ok=True)
            ca.to_csv(cache_file)
            return ca
    except Exception as e:
        logger.warning(f"Cross-asset download failed: {e}")

    # Fallback: synthetic
    if PRODUCTION_MODE:
        raise RuntimeError(
            "Cross-asset data fetch failed and TRADING_ENV is set to production/paper/live. "
            "Refusing to use synthetic data. Check network connectivity and API keys."
        )
    logger.warning("Network unavailable — generating synthetic cross-asset data")
    ca = _generate_synthetic_cross_asset(tickers, start_date, end_date)
    os.makedirs(cache_dir, exist_ok=True)
    ca.to_csv(cache_file)
    return ca


def fetch_fundamental_data(
    tickers: List[str], cache_dir: str = "data",
) -> Dict[str, dict]:
    """
    Fetch fundamental data (valuation, profitability, growth, quality, analyst).

    WARNING — LOOK-AHEAD BIAS:
    yfinance returns CURRENT fundamental data (today's PE, ROE, etc.) regardless
    of the backtest date. This means the model trains on information it wouldn't
    have had access to at that historical point in time. This inflates backtest
    returns by an estimated 2-5%.

    For production: replace with point-in-time fundamental data from
    Financial Modeling Prep, Sharadar, or Compustat. These provide as-reported
    financials with proper publication dates.

    Mitigation applied: fundamental features are cross-sectionally ranked
    (relative, not absolute) which partially reduces the bias since all stocks
    are equally affected by the look-ahead.
    """
    cache_file = os.path.join(cache_dir, "fundamentals.json")
    if _is_json_cache_valid(cache_file):
        logger.info("Loading cached fundamentals")
        with open(cache_file) as f:
            return json.load(f)

    logger.info(f"Fetching fundamentals for {len(tickers)} tickers...")
    fundamentals = {}
    fields = [
        "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
        "enterpriseToRevenue", "enterpriseToEbitda",
        "returnOnEquity", "returnOnAssets", "grossMargins",
        "operatingMargins", "profitMargins",
        "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
        "debtToEquity", "currentRatio", "quickRatio",
        "recommendationMean", "targetMeanPrice", "numberOfAnalystOpinions",
        "marketCap", "beta",
        "dividendYield", "payoutRatio",
        "shortPercentOfFloat", "shortRatio",
    ]

    try:
        import yfinance as yf
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
    except Exception as e:
        logger.warning(f"yfinance fundamentals failed: {e}")

    if not fundamentals:
        if PRODUCTION_MODE:
            raise RuntimeError(
                "Fundamental data fetch failed and TRADING_ENV is set to production/paper/live. "
                "Refusing to use synthetic data. Check network connectivity and API keys."
            )
        logger.warning("Network unavailable — generating synthetic fundamentals")
        fundamentals = _generate_synthetic_fundamentals(tickers)

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
    if _is_json_cache_valid(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    logger.info(f"Fetching earnings dates for {len(tickers)} tickers...")
    earnings = {}
    try:
        import yfinance as yf
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
    except Exception as e:
        logger.warning(f"yfinance earnings dates failed: {e}")

    if not earnings:
        if PRODUCTION_MODE:
            raise RuntimeError(
                "Earnings dates fetch failed and TRADING_ENV is set to production/paper/live. "
                "Refusing to use synthetic data. Check network connectivity and API keys."
            )
        logger.warning("Network unavailable — generating synthetic earnings dates")
        earnings = _generate_synthetic_earnings(tickers)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(earnings, f)
    logger.info(f"Fetched earnings dates for {len(earnings)} tickers")
    return earnings
