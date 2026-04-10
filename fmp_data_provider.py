"""
Consolidated FMP data provider — bulk endpoints + premium features.

Provides:
1. Bulk TTM fundamentals (1-3 API calls for ALL tickers)
2. Analyst estimates → EPS revision features
3. Real insider trades (SEC Form 4)
4. Financial scores (Piotroski F-Score, Altman Z-Score)
5. Price target consensus (analyst targets)
6. Claude LLM sentiment on FMP stock news

All endpoints use /stable/ base URL.
Per-ticker endpoints use concurrent fetching (20 threads) for ~15x speedup.
"""
import json
import os
import logging
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

FMP_BASE = "https://financialmodelingprep.com/stable"

# Concurrency: 20 threads stays well under 750 req/min Premium limit
# (20 threads × ~2 req/sec each = ~40 req/sec = 2,400 req/min theoretical,
#  but FMP latency ~0.5-1s/req so actual is ~20-40 req/sec, under 750 with margin)
MAX_WORKERS = 3


def _fmp_get(endpoint: str, api_key: str, params: dict = None, timeout: float = 15.0):
    import httpx, time
    params = params or {}
    params["apikey"] = api_key
    time.sleep(0.15)  # ~6 req/s to stay under 750/min
    return httpx.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=timeout)


def _fetch_parallel(
    tickers: List[str],
    fetch_one: Callable[[str], Optional[dict]],
    label: str = "FMP",
    max_workers: int = MAX_WORKERS,
) -> Dict[str, dict]:
    """
    Fetch data for multiple tickers concurrently.

    Args:
        tickers: list of ticker symbols
        fetch_one: function(ticker) -> dict or None
        label: log label
        max_workers: thread pool size

    Returns: {ticker: data} for all successful fetches
    """
    results = {}
    errors = 0
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_ticker = {pool.submit(fetch_one, t): t for t in tickers}

        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            if (i + 1) % 50 == 0:
                logger.info(f"  {label}: {i + 1}/{total} ({len(results)} OK, {errors} errors)")
            try:
                data = future.result()
                if data:
                    results[ticker] = data
            except Exception as e:
                errors += 1
                logger.debug(f"{label} failed for {ticker}: {e}")
                # Circuit breaker: if >20% fail, likely an API issue
                if errors > total * 0.2 and errors > 10:
                    logger.error(f"{label}: >20% failures ({errors}/{i+1}), possible API issue")
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    return results


# ---------------------------------------------------------------------------
# 1. BULK TTM Fundamentals (replaces per-ticker fetching)
# ---------------------------------------------------------------------------

_BULK_RATIOS_MAP = {
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

_BULK_METRICS_MAP = {
    "returnOnEquityTTM": "returnOnEquity",
    "returnOnAssetsTTM": "returnOnAssets",
    "evToSalesTTM": "enterpriseToRevenue",
    "marketCap": "marketCap",
}


def fetch_bulk_fundamentals(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
) -> Dict[str, dict]:
    """
    Fetch TTM fundamentals for ALL tickers via bulk endpoints.
    2 API calls total instead of 424 × 4 = 1,696.
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        return {}

    cache_file = os.path.join(cache_dir, "fmp_bulk_fundamentals.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_date") == datetime.now().strftime("%Y-%m-%d"):
                logger.info("Loading cached bulk fundamentals")
                return {k: v for k, v in cached.items() if k != "_date"}
        except Exception:
            pass

    import httpx
    fundamentals = {}
    ticker_set = set(tickers)

    # Bulk ratios TTM — single call for all tickers
    logger.info("Fetching bulk ratios TTM...")
    resp = _fmp_get("ratios-ttm-bulk", api_key)
    if resp.status_code == 200:
        for rec in resp.json() or []:
            sym = rec.get("symbol", "")
            if sym not in ticker_set:
                continue
            fund = fundamentals.setdefault(sym, {})
            for fmp_field, yf_field in _BULK_RATIOS_MAP.items():
                val = rec.get(fmp_field)
                if val is not None and isinstance(val, (int, float)):
                    try:
                        if not np.isnan(val):
                            fund[yf_field] = float(val)
                    except (TypeError, ValueError):
                        fund[yf_field] = float(val)
    else:
        logger.warning(f"Bulk ratios TTM failed: {resp.status_code}")

    # Bulk key metrics TTM — single call
    logger.info("Fetching bulk key metrics TTM...")
    resp = _fmp_get("key-metrics-ttm-bulk", api_key)
    if resp.status_code == 200:
        for rec in resp.json() or []:
            sym = rec.get("symbol", "")
            if sym not in ticker_set:
                continue
            fund = fundamentals.setdefault(sym, {})
            for fmp_field, yf_field in _BULK_METRICS_MAP.items():
                if yf_field not in fund:
                    val = rec.get(fmp_field)
                    if val is not None and isinstance(val, (int, float)):
                        try:
                            if not np.isnan(val):
                                fund[yf_field] = float(val)
                        except (TypeError, ValueError):
                            fund[yf_field] = float(val)
    else:
        logger.warning(f"Bulk key metrics TTM failed: {resp.status_code}")

    # Profile bulk for beta (single call)
    logger.info("Fetching bulk profiles...")
    resp = _fmp_get("profile-bulk", api_key, {"part": 0})
    if resp.status_code == 200:
        for rec in resp.json() or []:
            sym = rec.get("symbol", "")
            if sym not in ticker_set:
                continue
            fund = fundamentals.setdefault(sym, {})
            if "beta" not in fund:
                val = rec.get("beta")
                if val is not None and isinstance(val, (int, float)):
                    fund["beta"] = float(val)
            if "marketCap" not in fund:
                val = rec.get("marketCap") or rec.get("mktCap")
                if val is not None and isinstance(val, (int, float)):
                    fund["marketCap"] = float(val)

    # If bulk failed (402), fall back to per-ticker TTM from fmp_features
    if len(fundamentals) < len(tickers) * 0.3:
        logger.info("Bulk endpoints unavailable, falling back to per-ticker TTM...")
        from fmp_features import fetch_fmp_fundamental_data
        fundamentals = fetch_fmp_fundamental_data(tickers, api_key, cache_dir)

    # Cache
    if fundamentals:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(fundamentals)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    logger.info(f"Bulk fundamentals: {len(fundamentals)}/{len(tickers)} tickers")
    return fundamentals


# ---------------------------------------------------------------------------
# 2. Analyst Estimates → EPS Revision Features
# ---------------------------------------------------------------------------

def fetch_analyst_estimates(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
) -> Dict[str, dict]:
    """
    Fetch forward analyst estimates for EPS revision signal.

    EPS revision = (current_estimate - prior_estimate) / abs(prior_estimate)
    One of the strongest cross-sectional alpha factors.

    Returns: {ticker: {eps_revision_pct, fwd_pe_ratio, estimate_dispersion, n_analysts}}
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        return {}

    cache_file = os.path.join(cache_dir, "fmp_analyst_estimates.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_date") == datetime.now().strftime("%Y-%m-%d"):
                return {k: v for k, v in cached.items() if k != "_date"}
        except Exception:
            pass

    def _fetch_one_estimate(ticker: str) -> Optional[dict]:
        resp = _fmp_get("analyst-estimates", api_key, {
            "symbol": ticker, "period": "quarter", "limit": 4,
        })
        if resp.status_code in (402, 403):
            return None
        data = (resp.json() or []) if resp.status_code == 200 else []
        if len(data) < 2:
            return None

        current = data[0]
        prior = data[1]

        cur_eps = current.get("epsAvg") or current.get("estimatedEpsAvg") or 0
        prior_eps = prior.get("epsAvg") or prior.get("estimatedEpsAvg") or 0

        result = {}
        if prior_eps != 0:
            result["eps_revision_pct"] = (cur_eps - prior_eps) / abs(prior_eps)

        eps_high = current.get("epsHigh") or current.get("estimatedEpsHigh") or 0
        eps_low = current.get("epsLow") or current.get("estimatedEpsLow") or 0
        if cur_eps != 0:
            result["estimate_dispersion"] = (eps_high - eps_low) / abs(cur_eps)

        cur_rev = current.get("revenueAvg") or current.get("estimatedRevenueAvg") or 0
        prior_rev = prior.get("revenueAvg") or prior.get("estimatedRevenueAvg") or 0
        if prior_rev != 0:
            result["revenue_revision_pct"] = (cur_rev - prior_rev) / abs(prior_rev)

        result["n_analysts"] = (
            current.get("numAnalystsEps")
            or current.get("numberAnalystsEstimatedEps")
            or 0
        )
        return result if result else None

    estimates = _fetch_parallel(tickers, _fetch_one_estimate, "Analyst estimates")

    if estimates:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(estimates)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    logger.info(f"Analyst estimates: {len(estimates)} tickers")
    return estimates


# ---------------------------------------------------------------------------
# 3. Real Insider Trades (SEC Form 4)
# ---------------------------------------------------------------------------

def fetch_insider_trades(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
) -> Dict[str, dict]:
    """
    Fetch real insider trading data from FMP /stable/insider-trading/search.

    Returns: {ticker: {net_buy_ratio, buy_dollar_value, n_distinct_buyers, ...}}
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        return {}

    cache_file = os.path.join(cache_dir, "fmp_insider_trades.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_date") == datetime.now().strftime("%Y-%m-%d"):
                return {k: v for k, v in cached.items() if k != "_date"}
        except Exception:
            pass

    def _is_buy(t: dict) -> bool:
        tt = (t.get("transactionType", "") or "").lower()
        return "purchase" in tt or tt == "p-purchase"

    def _is_sell(t: dict) -> bool:
        tt = (t.get("transactionType", "") or "").lower()
        return "sale" in tt or tt == "s-sale"

    def _fetch_one_insider(ticker: str) -> Optional[dict]:
        resp = _fmp_get("insider-trading/search", api_key, {
            "symbol": ticker, "limit": 50, "page": 0,
        })
        if resp.status_code in (402, 403):
            return None
        transactions = (resp.json() or []) if resp.status_code == 200 else []
        if not transactions:
            return None

        n_buys = sum(1 for t in transactions if _is_buy(t))
        n_sells = sum(1 for t in transactions if _is_sell(t))
        buy_value = sum(
            abs(float(t.get("securitiesTransacted", 0) or 0) * float(t.get("price", 0) or 0))
            for t in transactions if _is_buy(t)
        )
        distinct_buyers = len(set(
            t.get("reportingName", "") for t in transactions if _is_buy(t)
        ))
        total = n_buys + n_sells
        return {
            "n_buys": n_buys,
            "n_sells": n_sells,
            "net_buy_ratio": n_buys / total if total > 0 else 0.5,
            "buy_dollar_value": buy_value,
            "n_distinct_buyers": distinct_buyers,
        }

    insider_data = _fetch_parallel(tickers, _fetch_one_insider, "Insider trades")

    if insider_data:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(insider_data)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    logger.info(f"Insider trades: {len(insider_data)} tickers")
    return insider_data


# ---------------------------------------------------------------------------
# 4. Financial Scores (Piotroski F-Score, Altman Z-Score)
# ---------------------------------------------------------------------------

def fetch_financial_scores(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
) -> Dict[str, dict]:
    """
    Fetch Piotroski F-Score and Altman Z-Score from FMP.

    Piotroski: 0-9, higher = stronger fundamentals (quality factor)
    Altman Z: >2.99 = safe, 1.81-2.99 = grey zone, <1.81 = distress risk
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        return {}

    cache_file = os.path.join(cache_dir, "fmp_financial_scores.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_date") == datetime.now().strftime("%Y-%m-%d"):
                return {k: v for k, v in cached.items() if k != "_date"}
        except Exception:
            pass

    # Try bulk first
    scores_data = {}
    resp = _fmp_get("scores-bulk", api_key)
    if resp.status_code == 200:
        ticker_set = set(tickers)
        for rec in resp.json() or []:
            sym = rec.get("symbol", "")
            if sym not in ticker_set:
                continue
            piotroski = rec.get("piotroskiScore")
            altman = rec.get("altmanZScore")
            if piotroski is not None or altman is not None:
                entry = {}
                if piotroski is not None:
                    entry["piotroskiScore"] = float(piotroski)
                if altman is not None and isinstance(altman, (int, float)):
                    entry["altmanZScore"] = float(altman)
                scores_data[sym] = entry
        logger.info(f"Financial scores (bulk): {len(scores_data)} tickers")
    else:
        # Fallback to per-ticker (concurrent)
        logger.info("Bulk scores unavailable, fetching per-ticker...")

        def _fetch_one_score(ticker: str) -> Optional[dict]:
            resp = _fmp_get("financial-scores", api_key, {"symbol": ticker})
            if resp.status_code in (402, 403):
                return None
            data = resp.json()
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                return None
            entry = {}
            p = data.get("piotroskiScore")
            z = data.get("altmanZScore")
            if p is not None:
                entry["piotroskiScore"] = float(p)
            if z is not None and isinstance(z, (int, float)):
                entry["altmanZScore"] = float(z)
            return entry if entry else None

        scores_data = _fetch_parallel(tickers, _fetch_one_score, "Financial scores")

    if scores_data:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(scores_data)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    return scores_data


# ---------------------------------------------------------------------------
# 5. Price Target Consensus
# ---------------------------------------------------------------------------

def fetch_price_targets(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
) -> Dict[str, dict]:
    """
    Fetch analyst price target consensus.

    Returns: {ticker: {targetMeanPrice, targetHighPrice, targetLowPrice, n_analysts}}
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        return {}

    # Try bulk first
    cache_file = os.path.join(cache_dir, "fmp_price_targets.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_date") == datetime.now().strftime("%Y-%m-%d"):
                return {k: v for k, v in cached.items() if k != "_date"}
        except Exception:
            pass

    targets = {}
    resp = _fmp_get("price-target-summary-bulk", api_key)
    if resp.status_code == 200:
        ticker_set = set(tickers)
        for rec in resp.json() or []:
            sym = rec.get("symbol", "")
            if sym not in ticker_set:
                continue
            entry = {}
            for fmp_f, our_f in [
                ("lastMonthAvgPriceTarget", "targetMeanPrice"),
                ("lastQuarterAvgPriceTarget", "targetQuarterPrice"),
            ]:
                val = rec.get(fmp_f)
                if val and isinstance(val, (int, float)):
                    entry[our_f] = float(val)
            if entry:
                targets[sym] = entry
        logger.info(f"Price targets (bulk): {len(targets)} tickers")
    else:
        # Per-ticker fallback (concurrent)
        def _fetch_one_target(ticker: str) -> Optional[dict]:
            resp = _fmp_get("price-target-consensus", api_key, {"symbol": ticker})
            if resp.status_code in (402, 403):
                return None
            data = resp.json()
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                return None
            entry = {}
            for field in ["targetHigh", "targetLow", "targetConsensus", "targetMedian"]:
                val = data.get(field)
                if val and isinstance(val, (int, float)):
                    entry[field] = float(val)
            if not entry:
                return None
            entry["targetMeanPrice"] = entry.get("targetConsensus", entry.get("targetMedian", 0))
            return entry

        targets = _fetch_parallel(tickers, _fetch_one_target, "Price targets")

    if targets:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(targets)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    return targets


# ---------------------------------------------------------------------------
# 6. Claude LLM Sentiment on FMP News
# ---------------------------------------------------------------------------

def fetch_claude_sentiment(
    tickers: List[str],
    api_key: str = "",
    cache_dir: str = "data",
    max_articles_per_ticker: int = 5,
) -> Dict[str, dict]:
    """
    Fetch stock news from FMP and score with Claude Haiku.

    Much more accurate than keyword-based sentiment because Claude:
    - Understands context (e.g. "revenue missed but guidance raised" = mixed, not negative)
    - Handles sarcasm, hedging, conditional statements
    - Scores on continuous scale with reasoning

    Cost: ~$0.01 per 100 articles with Haiku = ~$0.50/day for 424 tickers

    Returns: {ticker: {llm_sentiment, llm_confidence, n_articles, ...}}
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        logger.info("No ANTHROPIC_API_KEY — skipping Claude sentiment")
        return {}
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        return {}

    cache_file = os.path.join(cache_dir, "fmp_claude_sentiment.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_date") == datetime.now().strftime("%Y-%m-%d"):
                logger.info("Loading cached Claude sentiment")
                return {k: v for k, v in cached.items() if k != "_date"}
        except Exception:
            pass

    import httpx

    # Step 1: Fetch news for all tickers (batch by 5 tickers per call)
    logger.info(f"Fetching FMP news for {len(tickers)} tickers...")
    ticker_articles: Dict[str, List[str]] = {}
    batch_size = 5
    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start:batch_start + batch_size]
        symbols = ",".join(batch)
        try:
            resp = _fmp_get("news/stock", api_key, {
                "symbols": symbols, "limit": max_articles_per_ticker * len(batch), "page": 0,
            })
            if resp.status_code == 200:
                for article in resp.json() or []:
                    sym = article.get("symbol", "")
                    title = article.get("title", "")
                    text = article.get("text", "")
                    snippet = title
                    if text:
                        snippet = f"{title}. {text[:200]}"
                    if sym in set(batch) and snippet:
                        articles = ticker_articles.setdefault(sym, [])
                        if len(articles) < max_articles_per_ticker:
                            articles.append(snippet)
        except Exception as e:
            logger.debug(f"News fetch failed for batch {batch}: {e}")
        time.sleep(0.15)

    if not ticker_articles:
        logger.warning("No news articles fetched from FMP")
        return {}

    logger.info(f"Fetched news for {len(ticker_articles)} tickers, scoring with Claude...")

    # Step 2: Score with Claude Haiku in batches
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
    except ImportError:
        logger.warning("anthropic package not installed — pip install anthropic")
        return {}

    sentiment_data = {}

    # Batch tickers into groups for efficiency (10 tickers per LLM call)
    ticker_list = list(ticker_articles.keys())
    llm_batch_size = 10

    for batch_start in range(0, len(ticker_list), llm_batch_size):
        batch_tickers = ticker_list[batch_start:batch_start + llm_batch_size]

        # Build prompt with all tickers' articles
        prompt_parts = []
        for ticker in batch_tickers:
            articles = ticker_articles.get(ticker, [])
            articles_text = "\n".join(f"  - {a}" for a in articles)
            prompt_parts.append(f"{ticker}:\n{articles_text}")

        prompt = (
            "Score the sentiment of these stock news articles for each ticker. "
            "For each ticker, return a JSON object with:\n"
            "- sentiment: float from -1.0 (very bearish) to +1.0 (very bullish)\n"
            "- confidence: float from 0.0 to 1.0 (how confident in the score)\n"
            "- reasoning: brief 5-word explanation\n\n"
            "Consider: Is the news about earnings beats/misses, guidance changes, "
            "analyst actions, product launches, legal issues, macro impacts?\n\n"
            "Return ONLY a JSON object mapping ticker -> {sentiment, confidence, reasoning}.\n\n"
            + "\n\n".join(prompt_parts)
        )

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Parse JSON from response
            # Handle potential markdown wrapping
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            scores = json.loads(text)

            for ticker in batch_tickers:
                score = scores.get(ticker, {})
                if isinstance(score, dict) and "sentiment" in score:
                    sentiment_data[ticker] = {
                        "llm_sentiment": float(score["sentiment"]),
                        "llm_confidence": float(score.get("confidence", 0.5)),
                        "n_articles": len(ticker_articles.get(ticker, [])),
                    }
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Claude sentiment batch failed: {e}")
            # Fallback: score individually
            for ticker in batch_tickers:
                articles = ticker_articles.get(ticker, [])
                if articles:
                    try:
                        single_resp = client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=100,
                            messages=[{"role": "user", "content": (
                                f"Rate the stock sentiment of these headlines for {ticker} "
                                f"from -1.0 (bearish) to +1.0 (bullish). "
                                f"Reply with ONLY a number.\n\n"
                                + "\n".join(f"- {a}" for a in articles)
                            )}],
                        )
                        score_text = single_resp.content[0].text.strip()
                        score_val = float(score_text.replace(",", ""))
                        score_val = max(-1.0, min(1.0, score_val))
                        sentiment_data[ticker] = {
                            "llm_sentiment": score_val,
                            "llm_confidence": 0.5,
                            "n_articles": len(articles),
                        }
                    except Exception:
                        pass

        if batch_start % 50 == 0 and batch_start > 0:
            logger.info(f"  Claude sentiment: {batch_start}/{len(ticker_list)} tickers scored")

    # Cache
    if sentiment_data:
        os.makedirs(cache_dir, exist_ok=True)
        to_cache = dict(sentiment_data)
        to_cache["_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(cache_file, "w") as f:
            json.dump(to_cache, f)

    logger.info(f"Claude sentiment: {len(sentiment_data)} tickers scored")
    return sentiment_data


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_premium_features(
    tickers: List[str],
    prices: pd.DataFrame,
    api_key: str = "",
    cache_dir: str = "data",
    live_mode: bool = False,
) -> Dict[tuple, pd.DataFrame]:
    """
    Build all premium FMP features in one call.

    Args:
        live_mode: If True, also run Claude sentiment on current news.
                   False for backtest (can't backtest on today's news).

    Returns dict of (category, feature_name) -> DataFrame features,
    same format as other feature builders.
    """
    if not api_key:
        api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        return {}

    dates = prices.index
    feats = {}

    def _broadcast(values: dict, name: str) -> pd.DataFrame:
        series = pd.Series({t: values.get(t, np.nan) for t in prices.columns})
        return pd.DataFrame(
            np.tile(series.values, (len(dates), 1)),
            index=dates, columns=prices.columns,
        )

    def _rank_cs(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)

    # --- Analyst Estimates ---
    logger.info("Fetching analyst estimates...")
    estimates = fetch_analyst_estimates(tickers, api_key, cache_dir)
    if estimates:
        for field in ["eps_revision_pct", "revenue_revision_pct", "estimate_dispersion"]:
            vals = {t: d.get(field) for t, d in estimates.items() if d.get(field) is not None}
            if vals:
                df = _broadcast(vals, field)
                feats[("analyst", field)] = df
                feats[("analyst", f"cs_rank_{field}")] = _rank_cs(df)

        n_analysts = {t: float(d.get("n_analysts", 0)) for t, d in estimates.items()}
        if n_analysts:
            feats[("analyst", "cs_rank_n_analysts")] = _rank_cs(_broadcast(n_analysts, "n"))

    # --- Financial Scores ---
    logger.info("Fetching financial scores...")
    scores = fetch_financial_scores(tickers, api_key, cache_dir)
    if scores:
        piotroski = {t: d.get("piotroskiScore") for t, d in scores.items()
                     if d.get("piotroskiScore") is not None}
        if piotroski:
            df = _broadcast(piotroski, "piotroski")
            feats[("quality", "piotroski_score")] = df
            feats[("quality", "cs_rank_piotroski")] = _rank_cs(df)

        altman = {t: d.get("altmanZScore") for t, d in scores.items()
                  if d.get("altmanZScore") is not None}
        if altman:
            df = _broadcast(altman, "altman_z")
            feats[("quality", "altman_z_score")] = df
            feats[("quality", "cs_rank_altman_z")] = _rank_cs(df)

    # --- Price Targets ---
    logger.info("Fetching price targets...")
    targets = fetch_price_targets(tickers, api_key, cache_dir)
    if targets:
        last_prices = prices.iloc[-1]
        upside = {}
        for t in tickers:
            tp = targets.get(t, {}).get("targetMeanPrice")
            cp = last_prices.get(t)
            if tp and cp and cp > 0:
                upside[t] = tp / cp - 1
        if upside:
            df = _broadcast(upside, "target_upside")
            feats[("analyst", "target_upside")] = df
            feats[("analyst", "cs_rank_target_upside")] = _rank_cs(df)

    # --- Claude Sentiment (live only — can't backtest on today's news) ---
    sentiment = fetch_claude_sentiment(tickers, api_key, cache_dir) if live_mode else {}
    if sentiment:
        llm_sent = {t: d.get("llm_sentiment") for t, d in sentiment.items()
                    if d.get("llm_sentiment") is not None}
        if llm_sent:
            df = _broadcast(llm_sent, "llm_sentiment")
            feats[("sent", "llm_sentiment")] = df
            feats[("sent", "cs_rank_llm_sentiment")] = _rank_cs(df)

        llm_conf = {t: d.get("llm_confidence") for t, d in sentiment.items()
                    if d.get("llm_confidence") is not None}
        if llm_conf:
            feats[("sent", "llm_confidence")] = _broadcast(llm_conf, "conf")

        # Confidence-weighted sentiment (high confidence amplifies signal)
        if llm_sent and llm_conf:
            weighted = {t: llm_sent.get(t, 0) * llm_conf.get(t, 0.5)
                        for t in set(llm_sent) & set(llm_conf)}
            if weighted:
                df = _broadcast(weighted, "weighted_sent")
                feats[("sent", "cs_rank_llm_weighted_sentiment")] = _rank_cs(df)

    logger.info(f"Premium features: {len(feats)} signals")
    return feats
