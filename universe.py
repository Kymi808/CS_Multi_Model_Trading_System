"""
Universe selection with sector classification.
"""
import pandas as pd
import yfinance as yf
import logging
import json
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def get_sp500_tickers() -> List[str]:
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.warning(f"Failed to scrape S&P 500: {e}. Using fallback.")
        return _fallback_tickers()


def get_sp500_sector_map() -> Dict[str, str]:
    """Get sector mapping from Wikipedia S&P 500 table."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        df = tables[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return dict(zip(df["Symbol"], df["GICS Sector"]))
    except Exception:
        return {}


def _fallback_tickers() -> List[str]:
    return [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
        "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
        "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
        "ACN", "TMO", "ABT", "DHR", "NEE", "LIN", "ADBE", "TXN", "PM",
        "CMCSA", "NKE", "RTX", "HON", "ORCL", "COP", "UPS", "LOW", "QCOM",
        "BA", "SPGI", "CAT", "INTC", "AMD", "GS", "MS", "BLK", "DE",
        "AXP", "MDLZ", "ISRG", "ADI", "SYK", "GILD", "BKNG", "VRTX",
        "REGN", "MMC", "PLD", "CB", "CI", "SO", "DUK", "EOG", "BSX",
        "LRCX", "CME", "MO", "ZTS", "SCHW", "ANET", "SNPS", "CDNS",
        "NOC", "ITW", "KLAC", "SHW", "FDX", "EMR", "APD", "MCK", "GD",
        "CL", "ORLY", "AJG", "HUM", "MCHP", "TDG", "WM", "SLB",
        "PSX", "VLO", "MPC", "FTNT", "ROP", "PCAR",
    ]


def filter_universe_by_liquidity(
    tickers: List[str], cfg, prices: pd.DataFrame, volumes: pd.DataFrame,
) -> List[str]:
    filtered = []
    for ticker in tickers:
        if ticker not in prices.columns or ticker not in volumes.columns:
            continue
        px = prices[ticker].dropna()
        vol = volumes[ticker].dropna()
        if len(px) < cfg.min_history_days:
            continue
        if 1 - len(px) / len(prices) > cfg.max_missing_pct:
            continue
        common_idx = px.index.intersection(vol.index)
        if len(common_idx) < cfg.min_history_days:
            continue
        if (px.loc[common_idx] * vol.loc[common_idx]).mean() < cfg.min_avg_dollar_volume:
            continue
        filtered.append(ticker)
    logger.info(f"Universe filtered: {len(tickers)} -> {len(filtered)} tickers")
    return filtered


def get_universe(cfg) -> List[str]:
    if cfg.universe_source == "sp500":
        return get_sp500_tickers()
    elif cfg.universe_source == "custom":
        return cfg.custom_tickers
    return get_sp500_tickers()


def load_sector_map(tickers: List[str], cache_dir: str = "data") -> Dict[str, str]:
    cache_file = os.path.join(cache_dir, "sectors.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)
        # Return cached if it covers most tickers
        if len(set(tickers) & set(cached.keys())) > len(tickers) * 0.5:
            return cached

    # Try Wikipedia first (fast)
    sectors = get_sp500_sector_map()

    # Fill missing with yfinance (slow but thorough)
    missing = [t for t in tickers if t not in sectors]
    if missing:
        logger.info(f"Fetching sector info for {len(missing)} tickers from yfinance...")
        for ticker in missing[:50]:  # Cap to avoid rate limits
            try:
                info = yf.Ticker(ticker).info
                sectors[ticker] = info.get("sector", "Unknown")
            except Exception:
                sectors[ticker] = "Unknown"

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(sectors, f)
    return sectors
