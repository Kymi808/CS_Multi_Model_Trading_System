"""
OpenBB-sourced alternative data features.

IMPORTANT — BACKTEST vs PRODUCTION:
- Options data (IV skew, put-call ratio) and short interest are SNAPSHOT data.
  Free APIs do not provide historical time series.
- In BACKTEST mode: these features are EXCLUDED to prevent look-ahead bias.
  The model trains and backtests without them.
- In PRODUCTION mode: these features are included as real-time signals.
  The model sees current IV skew, put-call ratio, short interest alongside
  price-based features. This is safe because production only acts on
  current data.

If you need these in backtest, you need a paid historical provider:
- CBOE LiveVol for historical options chains
- FINRA for bimonthly historical short interest reports
- Quandl/Nasdaq Data Link for historical SI% of float

References:
- Bali & Hovakimian (2009), "Volatility Spreads and Expected Stock Returns"
- Boehmer et al. (2008), "Which Shorts Are Informed?"
- Rapach et al. (2016), "Short Interest and Aggregate Stock Returns"
"""
import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

# Try OpenBB import
OPENBB_AVAILABLE = False
try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    pass


def fetch_options_data(
    tickers: List[str],
    cache_dir: str = "data",
    live_mode: bool = False,
) -> Dict[str, dict]:
    """
    Fetch options-implied signals for each ticker.

    Args:
        tickers: List of ticker symbols.
        cache_dir: Directory for caching.
        live_mode: If False (backtest), returns empty dict to prevent
                   look-ahead bias. If True (production), fetches current data.

    Key signals (production only):
    - IV skew: difference between OTM put IV and ATM call IV
    - Put-call ratio: volume of puts / volume of calls
    - ATM implied volatility
    """
    if not live_mode:
        logger.info("OpenBB options: skipped in backtest mode (no historical data — would leak)")
        return {}

    cache_file = os.path.join(cache_dir, "options_data.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_date") == datetime.now().strftime("%Y-%m-%d"):
                logger.info("Loading cached options data")
                return {k: v for k, v in cached.items() if k != "_date"}
        except Exception:
            pass

    if not OPENBB_AVAILABLE:
        logger.info("OpenBB not installed — no options data available")
        return {}

    options_data = {}
    for ticker in tickers[:100]:  # limit to avoid rate limits
        try:
            chain = obb.derivatives.options.chains(ticker)
            if chain is None:
                continue

            df = chain.to_dataframe() if hasattr(chain, 'to_dataframe') else pd.DataFrame()
            if df.empty:
                continue

            calls = df[df["option_type"] == "call"] if "option_type" in df.columns else pd.DataFrame()
            puts = df[df["option_type"] == "put"] if "option_type" in df.columns else pd.DataFrame()

            iv_col = "implied_volatility" if "implied_volatility" in df.columns else "iv"
            if iv_col in df.columns:
                call_iv = calls[iv_col].median() if len(calls) > 0 else 0
                put_iv = puts[iv_col].median() if len(puts) > 0 else 0
                iv_skew = put_iv - call_iv
            else:
                iv_skew = 0
                call_iv = 0

            vol_col = "volume" if "volume" in df.columns else "vol"
            if vol_col in df.columns:
                call_vol = calls[vol_col].sum() if len(calls) > 0 else 1
                put_vol = puts[vol_col].sum() if len(puts) > 0 else 0
                pc_ratio = put_vol / max(call_vol, 1)
            else:
                pc_ratio = 1.0

            options_data[ticker] = {
                "iv_skew": float(iv_skew),
                "put_call_ratio": float(pc_ratio),
                "atm_iv": float(call_iv),
            }

        except Exception as e:
            logger.debug(f"Options fetch failed for {ticker}: {e}")
            continue

    # Cache
    if options_data:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(options_data)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    logger.info(f"Options data: {len(options_data)} tickers (live mode)")
    return options_data


def fetch_short_interest(
    tickers: List[str],
    cache_dir: str = "data",
    live_mode: bool = False,
) -> Dict[str, dict]:
    """
    Fetch short interest data.

    Args:
        tickers: List of ticker symbols.
        cache_dir: Directory for caching.
        live_mode: If False (backtest), returns empty dict to prevent
                   look-ahead bias. If True (production), fetches current data.
    """
    if not live_mode:
        logger.info("OpenBB short interest: skipped in backtest mode (no historical data — would leak)")
        return {}

    cache_file = os.path.join(cache_dir, "short_interest.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_date") == datetime.now().strftime("%Y-%m-%d"):
                logger.info("Loading cached short interest")
                return {k: v for k, v in cached.items() if k != "_date"}
        except Exception:
            pass

    if not OPENBB_AVAILABLE:
        logger.info("OpenBB not installed — no short interest data available")
        return {}

    short_data = {}
    for ticker in tickers[:100]:
        try:
            si = obb.equity.shorts.short_interest(ticker)
            if si is None:
                continue

            df = si.to_dataframe() if hasattr(si, 'to_dataframe') else pd.DataFrame()
            if df.empty:
                continue

            latest = df.iloc[0] if len(df) > 0 else {}
            prior = df.iloc[1] if len(df) > 1 else latest

            si_pct = float(latest.get("short_interest_pct_float",
                          latest.get("short_percent_of_float", 0)) or 0)
            dtc = float(latest.get("days_to_cover", 0) or 0)

            prior_si = float(prior.get("short_interest_pct_float",
                            prior.get("short_percent_of_float", 0)) or 0)
            si_change = si_pct - prior_si

            short_data[ticker] = {
                "si_pct_float": si_pct,
                "days_to_cover": dtc,
                "si_change": si_change,
            }

        except Exception as e:
            logger.debug(f"Short interest failed for {ticker}: {e}")
            continue

    if short_data:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(short_data)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    logger.info(f"Short interest: {len(short_data)} tickers (live mode)")
    return short_data


def build_openbb_features(
    options_data: Dict[str, dict],
    short_data: Dict[str, dict],
    prices: pd.DataFrame,
) -> Dict[tuple, pd.DataFrame]:
    """
    Build ML features from OpenBB options + short interest data.

    In backtest mode, options_data and short_data will be empty dicts
    (fetch functions return {} when live_mode=False), so this returns
    no features. No look-ahead bias.

    In production, current snapshot data is broadcast across the
    prediction window (safe — we're only predicting forward from now).
    """
    if not options_data and not short_data:
        return {}

    tickers = list(prices.columns)
    dates = prices.index
    feats = {}

    def _broadcast(values: dict, name: str) -> pd.DataFrame:
        series = pd.Series(values)
        return pd.DataFrame(
            {t: series.get(t, np.nan) for t in tickers},
            index=dates,
        )

    def _rank_cs(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)

    # Options features
    if options_data:
        iv_skew = {t: d.get("iv_skew", 0) for t, d in options_data.items()}
        df = _broadcast(iv_skew, "iv_skew")
        feats[("options", "iv_skew")] = df
        feats[("options", "cs_rank_iv_skew")] = _rank_cs(df)

        pc_ratio = {t: d.get("put_call_ratio", 1.0) for t, d in options_data.items()}
        df = _broadcast(pc_ratio, "put_call_ratio")
        feats[("options", "put_call_ratio")] = df
        feats[("options", "cs_rank_put_call_ratio")] = _rank_cs(df)

        atm_iv = {t: d.get("atm_iv", 0) for t, d in options_data.items()}
        df = _broadcast(atm_iv, "atm_iv")
        feats[("options", "atm_iv")] = df
        feats[("options", "cs_rank_atm_iv")] = _rank_cs(df)

    # Short interest features
    if short_data:
        si_pct = {t: d.get("si_pct_float", 0) for t, d in short_data.items()}
        df = _broadcast(si_pct, "si_pct_float")
        feats[("shorts", "si_pct_float")] = df
        feats[("shorts", "cs_rank_si_pct")] = _rank_cs(df)

        dtc = {t: d.get("days_to_cover", 0) for t, d in short_data.items()}
        df = _broadcast(dtc, "days_to_cover")
        feats[("shorts", "days_to_cover")] = df
        feats[("shorts", "cs_rank_dtc")] = _rank_cs(df)

        si_chg = {t: d.get("si_change", 0) for t, d in short_data.items()}
        df = _broadcast(si_chg, "si_change")
        feats[("shorts", "si_change")] = df
        feats[("shorts", "cs_rank_si_change")] = _rank_cs(df)

    logger.info(f"OpenBB features: {len(feats)} signals (production live data)")
    return feats
