"""
Insider trading features from SEC Form 4 filings.

Insider buying is one of the strongest known predictive signals:
- Insiders buy for ONE reason: they think the stock will go up
- Insiders sell for MANY reasons: taxes, diversification, estate planning
- Therefore: buying signal >> selling signal

Features:
1. Net insider buying ratio (buys / total transactions)
2. Dollar value of insider purchases (log-scaled)
3. Number of distinct insiders buying (cluster buying = stronger)
4. Insider buying relative to market cap

Data source: SEC EDGAR (free) or FMP (with API key).
Fallback: synthetic data for offline testing.

Reference:
- Lakonishok & Lee (2001), "Are Insider Trades Informative?"
- Seyhun (1998), "Investment Intelligence from Insider Trading"
"""
import json
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def fetch_insider_data(
    tickers: List[str],
    lookback_days: int = 90,
    cache_dir: str = "data",
) -> Dict[str, dict]:
    """
    Fetch insider transaction data for multiple tickers.

    Tries FMP first (better data), falls back to synthetic.
    SEC EDGAR Form 4 is free but harder to parse in bulk.
    """
    cache_file = os.path.join(cache_dir, "insider_data.json")

    # Check cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_date") == datetime.now().strftime("%Y-%m-%d"):
                logger.info("Loading cached insider data")
                return {k: v for k, v in cached.items() if k != "_date"}
        except Exception:
            pass

    insider_data = {}

    # Try FMP if API key available
    fmp_key = os.environ.get("FMP_API_KEY", "")
    if fmp_key and fmp_key not in ("", "xxxxx"):
        insider_data = _fetch_from_fmp(tickers, fmp_key, lookback_days)

    # Fallback to synthetic
    if not insider_data:
        logger.info("Generating synthetic insider data (no FMP key or fetch failed)")
        insider_data = _generate_synthetic_insider(tickers)

    # Cache
    os.makedirs(cache_dir, exist_ok=True)
    to_cache = dict(insider_data)
    to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
    with open(cache_file, "w") as f:
        json.dump(to_cache, f)

    logger.info(f"Insider data: {len(insider_data)} tickers")
    return insider_data


def _fetch_from_fmp(
    tickers: List[str], api_key: str, lookback_days: int,
) -> Dict[str, dict]:
    """Fetch insider transactions from Financial Modeling Prep."""
    import httpx

    data = {}
    for ticker in tickers[:50]:  # limit to avoid rate limits
        try:
            resp = httpx.get(
                f"https://financialmodelingprep.com/api/v4/insider-trading",
                params={"symbol": ticker, "apikey": api_key, "limit": 50},
                timeout=10.0,
            )
            if resp.status_code != 200:
                continue

            transactions = resp.json()
            if not transactions:
                continue

            n_buys = sum(1 for t in transactions if t.get("transactionType") == "P-Purchase")
            n_sells = sum(1 for t in transactions if t.get("transactionType") == "S-Sale")
            buy_value = sum(
                abs(float(t.get("securitiesTransacted", 0)) * float(t.get("price", 0)))
                for t in transactions if t.get("transactionType") == "P-Purchase"
            )
            distinct_buyers = len(set(
                t.get("reportingName", "") for t in transactions
                if t.get("transactionType") == "P-Purchase"
            ))

            total = n_buys + n_sells
            data[ticker] = {
                "n_buys": n_buys,
                "n_sells": n_sells,
                "net_buy_ratio": n_buys / total if total > 0 else 0.5,
                "buy_dollar_value": buy_value,
                "n_distinct_buyers": distinct_buyers,
            }
        except Exception:
            continue

    return data


def _generate_synthetic_insider(tickers: List[str], seed: int = 42) -> Dict[str, dict]:
    """Generate synthetic insider data for offline testing."""
    rng = np.random.RandomState(seed)
    data = {}
    for ticker in tickers:
        n_buys = int(rng.poisson(2))
        n_sells = int(rng.poisson(3))
        total = n_buys + n_sells
        data[ticker] = {
            "n_buys": n_buys,
            "n_sells": n_sells,
            "net_buy_ratio": n_buys / total if total > 0 else 0.5,
            "buy_dollar_value": float(rng.exponential(500_000)) if n_buys > 0 else 0,
            "n_distinct_buyers": min(n_buys, int(rng.poisson(1) + 1)) if n_buys > 0 else 0,
        }
    return data


def build_insider_features(
    insider_data: Dict[str, dict],
    prices: pd.DataFrame,
    fundamentals: Optional[Dict[str, dict]] = None,
) -> Dict[tuple, pd.DataFrame]:
    """
    Build insider trading features for the ML model.

    Returns dict of ("insider", feature_name) -> DataFrame(dates x tickers)
    Static features broadcast across all dates (like fundamentals).
    """
    if not insider_data:
        return {}

    tickers = list(prices.columns)
    dates = prices.index
    feats = {}

    def _broadcast(values: dict, name: str) -> pd.DataFrame:
        """Broadcast static per-ticker values across all dates."""
        series = pd.Series(values)
        return pd.DataFrame(
            {t: series.get(t, np.nan) for t in tickers},
            index=dates,
        )

    def _rank_cs(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)

    # 1. Net insider buying ratio (higher = more buying relative to selling)
    buy_ratios = {t: d.get("net_buy_ratio", 0.5) for t, d in insider_data.items()}
    df = _broadcast(buy_ratios, "net_buy_ratio")
    feats[("insider", "net_buy_ratio")] = df
    feats[("insider", "cs_rank_net_buy_ratio")] = _rank_cs(df)

    # 2. Dollar value of insider purchases (log-scaled)
    buy_values = {
        t: np.log1p(d.get("buy_dollar_value", 0))
        for t, d in insider_data.items()
    }
    df = _broadcast(buy_values, "log_buy_value")
    feats[("insider", "log_buy_value")] = df
    feats[("insider", "cs_rank_log_buy_value")] = _rank_cs(df)

    # 3. Number of distinct insiders buying (cluster buying = stronger signal)
    n_buyers = {t: d.get("n_distinct_buyers", 0) for t, d in insider_data.items()}
    df = _broadcast(n_buyers, "n_distinct_buyers")
    feats[("insider", "n_distinct_buyers")] = df
    feats[("insider", "cs_rank_n_distinct_buyers")] = _rank_cs(df)

    # 4. Insider buying relative to market cap
    if fundamentals:
        buy_to_mcap = {}
        for t, d in insider_data.items():
            mcap = fundamentals.get(t, {}).get("marketCap", 0)
            if mcap > 0:
                buy_to_mcap[t] = d.get("buy_dollar_value", 0) / mcap
            else:
                buy_to_mcap[t] = 0
        df = _broadcast(buy_to_mcap, "buy_to_mcap")
        feats[("insider", "buy_to_mcap")] = df
        feats[("insider", "cs_rank_buy_to_mcap")] = _rank_cs(df)

    logger.info(f"Insider features: {len(feats)} signals")
    return feats
