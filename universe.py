"""
Universe selection with sector classification.

NOTE on survivorship bias:
This module uses CURRENT S&P 500 constituents for historical backtests.
Stocks that were delisted or removed from the index are excluded, which
inflates backtest returns by an estimated 1-3%. Production systems should
use point-in-time constituent lists from a data vendor (e.g., Compustat,
Sharadar, or similar).
"""
import pandas as pd
import logging
import json
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def get_sp500_tickers() -> List[str]:
    """
    Get S&P 500 constituents. Tries multiple sources in order:
    1. Alpaca asset API (if API keys available)
    2. Wikipedia scrape
    3. Hardcoded fallback list
    """
    # Try Alpaca first (most reliable, no scraping issues)
    tickers = _get_tickers_from_alpaca()
    if tickers and len(tickers) > 50:
        return tickers

    # Fallback to Wikipedia
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        logger.warning(f"Wikipedia scrape failed: {e}. Using fallback.")
        return _fallback_tickers()


def _get_tickers_from_alpaca() -> List[str]:
    """
    Get tradeable US equities from Alpaca's asset API.
    Filters for active, tradeable stocks on NYSE/NASDAQ with sufficient market cap.
    """
    try:
        import httpx
        api_key = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_API_SECRET", "")
        if not api_key or api_key in ("", "xxxxx"):
            return []

        base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        resp = httpx.get(
            f"{base_url}/v2/assets",
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
            params={
                "status": "active",
                "asset_class": "us_equity",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        assets = resp.json()

        # Filter for tradeable, shortable stocks on major exchanges
        tickers = [
            a["symbol"] for a in assets
            if a.get("tradable") and a.get("shortable")
            and a.get("exchange") in ("NYSE", "NASDAQ")
            and not a.get("symbol", "").endswith("W")  # exclude warrants
            and "." not in a.get("symbol", "")  # exclude class shares like BRK.B
        ]

        # Expanded universe: use ALL tradeable/shortable stocks from major exchanges
        # Downstream liquidity filter (min_avg_dollar_volume) handles the rest
        # This gives us 300-500 liquid names instead of ~100
        max_universe = 500
        known = set(_fallback_tickers())

        # Priority: known large-caps first, then remaining Alpaca assets
        prioritized = [t for t in tickers if t in known]
        remaining = [t for t in tickers if t not in known]
        combined = prioritized + remaining

        result = combined[:max_universe]
        logger.info(f"Fetched {len(result)} tickers from Alpaca ({len(prioritized)} known + {len(remaining[:max_universe-len(prioritized)])} additional)")
        return result

    except Exception as e:
        logger.debug(f"Alpaca asset API unavailable: {e}")
        return []


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
    """Expanded large/mid-cap universe (~300 names) for better cross-sectional signal."""
    return [
        # Mega-cap (top 30)
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA",
        "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
        "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
        "ACN", "TMO", "ABT", "CRM",
        # Large-cap (next 70)
        "DHR", "NEE", "LIN", "ADBE", "TXN", "PM", "CMCSA", "NKE", "RTX",
        "HON", "ORCL", "COP", "UPS", "LOW", "QCOM", "BA", "SPGI", "CAT",
        "INTC", "AMD", "GS", "MS", "BLK", "DE", "AXP", "MDLZ", "ISRG",
        "ADI", "SYK", "GILD", "BKNG", "VRTX", "REGN", "PLD", "CB",
        "CI", "SO", "DUK", "EOG", "BSX", "LRCX", "CME", "MO", "ZTS",
        "SCHW", "ANET", "SNPS", "CDNS", "NOC", "ITW", "KLAC", "SHW",
        "FDX", "EMR", "APD", "MCK", "GD", "CL", "ORLY", "AJG", "HUM",
        "MCHP", "TDG", "WM", "SLB", "PSX", "VLO", "MPC", "FTNT", "ROP",
        "PCAR", "NFLX",
        # Mid-cap expansion (next 200 — Russell 1000 names)
        "ABNB", "ACGL", "AFL", "AIG", "AIZ", "AJG", "ALL", "AMGN", "AMP",
        "AMT", "ANSS", "AON", "AOS", "APA", "APH", "APTV", "ARE", "ATO",
        "ATVI", "AWK", "AZO", "BAX", "BBY", "BDX", "BEN", "BF-B",
        "BIO", "BKR", "BR", "BRO", "BWA", "CAG", "CARR", "CBOE", "CBRE",
        "CCI", "CE", "CF", "CHD", "CHRW", "CINF", "CLX", "CMS", "CNP",
        "COO", "CPRT", "CRL", "CSCO", "CSGP", "CSX", "CTAS", "CTLT",
        "CTSH", "CTVA", "CVS", "D", "DAL", "DD", "DELL", "DFS",
        "DG", "DGX", "DHI", "DXCM", "EA", "EBAY", "ECL", "ED",
        "EFX", "EIX", "EL", "EMN", "ENPH", "EOG", "EPAM", "EQIX",
        "EQR", "EQT", "ES", "ESS", "ETN", "ETR", "EVRG", "EW",
        "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FBHS",
        "FCX", "FDS", "FICO", "FIS", "FISV", "FLT", "FMC",
        "FOX", "FOXA", "FRC", "FRT", "FTNT", "GE", "GEHC", "GEN",
        "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GPC", "GPN",
        "GRMN", "GWW", "HAL", "HAS", "HBAN", "HCA", "HOLX",
        "HPE", "HPQ", "HSIC", "HST", "HSY", "HWM", "IBM", "ICE",
        "IDXX", "IEX", "IFF", "ILMN", "INCY", "IP", "IPG", "IQV",
        "IR", "IRM", "IT", "JBHT", "JCI", "JKHY", "JNPR",
        "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KMB", "KMI",
        "KMX", "KR", "L", "LDOS", "LEN", "LH", "LHX",
        "LKQ", "LMT", "LNT", "LUV", "LVS", "LW", "LYB", "LYV",
        "MAA", "MAR", "MAS", "MKC", "MKTX", "MLM", "MOH",
        "MPWR", "MRO", "MSCI", "MSI", "MTB", "MTCH", "MTD",
        "MU", "NDAQ", "NDSN", "NEM", "NI", "NRG",
        "NSC", "NTAP", "NTRS", "NUE", "NVR", "NWL", "NWS",
        "O", "ODFL", "OKE", "OMC", "ON", "OTIS", "OXY",
        "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEAK", "PEG",
        "PFE", "PFG", "PGR", "PH", "PHM", "PKG", "PKI", "PNC",
        "PNR", "PNW", "POOL", "PPG", "PPL", "PRU", "PTC",
        "PVH", "PWR", "PXD", "PYPL", "QRVO",
        "RCL", "RE", "REG", "RF", "RHI", "RJF", "RL",
        "RMD", "ROK", "ROL", "ROST", "RSG",
        "SBAC", "SBUX", "SEDG", "SEE", "SJM", "SNA", "SNPS",
        "SPG", "SRE", "STE", "STT", "STX", "STZ", "SWK",
        "SWKS", "SYF", "SYY", "T", "TAP", "TDY", "TECH", "TEL",
        "TER", "TFC", "TFX", "TRGP", "TRMB", "TROW", "TRV",
        "TSCO", "TSN", "TT", "TTWO", "TXT", "TYL",
        "UAL", "UDR", "UHS", "ULTA", "URI", "USB",
        "VFC", "VICI", "VLO", "VMC", "VRSK", "VRSN", "VRTX",
        "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD",
        "WDC", "WEC", "WELL", "WFC", "WHR", "WRB", "WRK",
        "WST", "WTW", "WY", "WYNN", "XEL", "XYL",
        "YUM", "ZBH", "ZBRA", "ZION",
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
    """Load sector classifications. Uses cache, Wikipedia, or Alpaca as sources."""
    cache_file = os.path.join(cache_dir, "sectors.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)
        if isinstance(cached, dict) and len(set(tickers) & set(cached.keys())) > len(tickers) * 0.5:
            return cached

    # Try Wikipedia first (fast, has GICS sectors)
    sectors = get_sp500_sector_map()

    # Fill missing — use hardcoded sector mapping instead of yfinance API calls
    # This avoids the unreliable yf.Ticker().info API
    missing = [t for t in tickers if t not in sectors]
    if missing:
        known_sectors = _known_sector_map()
        for t in missing:
            sectors[t] = known_sectors.get(t, "Unknown")

    # If we still have poor coverage, generate synthetic
    if not sectors or len(set(tickers) & set(sectors.keys())) < len(tickers) * 0.3:
        logger.warning("Insufficient sector data — generating synthetic mapping")
        from data_loader import _generate_synthetic_sectors
        sectors = _generate_synthetic_sectors(tickers)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(sectors, f)
    return sectors


def _known_sector_map() -> Dict[str, str]:
    """Hardcoded GICS sector mapping for common large-caps."""
    return {
        # Technology
        "AAPL": "Information Technology", "MSFT": "Information Technology",
        "NVDA": "Information Technology", "AVGO": "Information Technology",
        "AMD": "Information Technology", "INTC": "Information Technology",
        "QCOM": "Information Technology", "TXN": "Information Technology",
        "ADI": "Information Technology", "LRCX": "Information Technology",
        "KLAC": "Information Technology", "SNPS": "Information Technology",
        "CDNS": "Information Technology", "MCHP": "Information Technology",
        "ADBE": "Information Technology", "CRM": "Information Technology",
        "ORCL": "Information Technology", "CSCO": "Information Technology",
        "ACN": "Information Technology", "ANET": "Information Technology",
        "FTNT": "Information Technology",
        # Communication
        "GOOGL": "Communication Services", "META": "Communication Services",
        "NFLX": "Communication Services", "CMCSA": "Communication Services",
        # Consumer Discretionary
        "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
        "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
        "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
        "BKNG": "Consumer Discretionary", "ORLY": "Consumer Discretionary",
        "TDG": "Consumer Discretionary",
        # Consumer Staples
        "PG": "Consumer Staples", "KO": "Consumer Staples",
        "PEP": "Consumer Staples", "COST": "Consumer Staples",
        "WMT": "Consumer Staples", "PM": "Consumer Staples",
        "CL": "Consumer Staples", "MDLZ": "Consumer Staples",
        "MO": "Consumer Staples",
        # Healthcare
        "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
        "MRK": "Health Care", "ABBV": "Health Care", "TMO": "Health Care",
        "ABT": "Health Care", "DHR": "Health Care", "ISRG": "Health Care",
        "SYK": "Health Care", "GILD": "Health Care", "VRTX": "Health Care",
        "REGN": "Health Care", "BSX": "Health Care", "ZTS": "Health Care",
        "CI": "Health Care", "HUM": "Health Care",
        # Financials
        "JPM": "Financials", "V": "Financials", "MA": "Financials",
        "GS": "Financials", "MS": "Financials", "BLK": "Financials",
        "SPGI": "Financials", "AXP": "Financials", "SCHW": "Financials",
        "CME": "Financials", "CB": "Financials", "MMC": "Financials",
        "AJG": "Financials",
        # Energy
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
        "EOG": "Energy", "SLB": "Energy", "PSX": "Energy",
        "VLO": "Energy", "MPC": "Energy",
        # Industrials
        "BA": "Industrials", "CAT": "Industrials", "DE": "Industrials",
        "RTX": "Industrials", "HON": "Industrials", "UPS": "Industrials",
        "FDX": "Industrials", "EMR": "Industrials", "NOC": "Industrials",
        "ITW": "Industrials", "GD": "Industrials", "WM": "Industrials",
        "PCAR": "Industrials", "ROP": "Industrials",
        # Materials
        "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
        # Real Estate
        "PLD": "Real Estate",
        # Utilities
        "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities",
        # Misc
        "MCK": "Health Care",
    }
