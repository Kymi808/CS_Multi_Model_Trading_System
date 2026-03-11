"""
News sentiment features using yfinance news + financial sentiment lexicon.

No API keys required. Uses yfinance's built-in news feed and a
purpose-built financial sentiment dictionary.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import json
import os
import re
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Financial sentiment lexicon
POSITIVE_WORDS = {
    "beat", "beats", "exceeded", "surpassed", "outperformed", "topped",
    "record", "blowout", "strong", "stellar", "robust", "solid",
    "raised", "upgraded", "boosted", "hiked", "lifted", "upped",
    "upside", "bullish", "optimistic", "confident", "accelerating",
    "growth", "growing", "expansion", "expanding", "soaring", "surging",
    "jumped", "rallied", "gained", "climbed", "advanced", "rose",
    "profit", "profitable", "revenue", "earnings", "dividend", "buyback",
    "innovation", "breakthrough", "partnership", "acquisition", "deal",
    "approved", "launch", "launched", "momentum",
    "upgrade", "buy", "overweight", "outperform", "positive", "winner",
    "opportunity", "upbeat", "recovery", "rebound", "turnaround",
}

NEGATIVE_WORDS = {
    "missed", "miss", "fell", "declined", "disappointed", "weak",
    "below", "shortfall", "plunged", "tumbled", "slumped", "dropped",
    "lowered", "downgraded", "cut", "slashed", "reduced", "warning",
    "warned", "downside", "bearish", "pessimistic", "cautious",
    "loss", "losses", "deficit", "debt", "bankruptcy", "default",
    "lawsuit", "investigation", "probe", "fraud", "scandal", "recall",
    "layoffs", "restructuring", "downturn", "recession", "crisis",
    "downgrade", "sell", "underweight", "underperform", "negative",
    "risk", "risks", "volatile", "uncertainty", "concern", "concerns",
    "headwinds", "challenged", "challenging", "pressure", "pressured",
    "crash", "correction", "selloff", "panic", "fear", "collapse",
}

INTENSIFIERS = {"very", "extremely", "significantly", "sharply", "dramatically", "massive", "huge"}
NEGATORS = {"not", "no", "never", "neither", "nor", "don't", "doesn't", "didn't", "won't", "isn't", "aren't"}


def score_headline(text: str) -> float:
    if not text:
        return 0.0
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    score = 0.0
    prev_word, prev_prev = "", ""
    for word in words:
        mult = 1.0
        if prev_word in NEGATORS or prev_prev in NEGATORS:
            mult = -0.8
        if prev_word in INTENSIFIERS:
            mult *= 1.5
        if word in POSITIVE_WORDS:
            score += 1.0 * mult
        elif word in NEGATIVE_WORDS:
            score -= 1.0 * mult
        prev_prev = prev_word
        prev_word = word
    return float(np.clip(score / (np.sqrt(len(words)) + 1e-8), -1, 1))


def _extract_titles_from_news(news_data) -> List[str]:
    """
    Extract titles from yfinance news, handling ALL known format variants:
    - v0.2.31: list of dicts with "title" key
    - v0.2.36+: list of dicts with nested "content.title"
    - Some versions: dict wrapper with "news" key
    - Some versions: dicts with "relatedTickers" instead of per-ticker
    """
    titles = []
    if news_data is None:
        return titles

    # Unwrap dict wrappers
    if isinstance(news_data, dict):
        for key in ["news", "items", "result", "data"]:
            if key in news_data:
                news_data = news_data[key]
                break
        else:
            return titles

    if not isinstance(news_data, (list, tuple)):
        return titles

    for item in news_data:
        if not isinstance(item, dict):
            continue
        title = None
        # Try direct "title" key
        if "title" in item and isinstance(item["title"], str):
            title = item["title"]
        # Try nested "content.title"
        elif "content" in item and isinstance(item["content"], dict):
            title = item["content"].get("title")
        # Try "headline"
        elif "headline" in item:
            title = item["headline"]
        if title and isinstance(title, str) and len(title) > 5:
            titles.append(title)

    return titles


def fetch_news_sentiment(
    tickers: List[str], cache_dir: str = "data", max_per_ticker: int = 10,
) -> Dict[str, dict]:
    cache_file = os.path.join(cache_dir, "news_sentiment.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if isinstance(cached, dict) and len(cached) > 0:
                age_hours = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600
                if age_hours < 6:
                    logger.info("Loading cached sentiment data")
                    return cached
        except Exception:
            pass

    logger.info(f"Fetching news sentiment for {len(tickers)} tickers...")
    sentiment_data = {}
    n_total_articles = 0

    for i, ticker in enumerate(tickers):
        if i % 20 == 0 and i > 0:
            logger.info(f"  Sentiment: {i}/{len(tickers)} ({n_total_articles} articles so far)")
        try:
            t = yf.Ticker(ticker)

            # Try multiple methods to get news
            titles = []

            # Method 1: t.news property
            try:
                news = t.news
                titles = _extract_titles_from_news(news)
            except Exception:
                pass

            # Method 2: t.get_news() method (newer yfinance)
            if not titles:
                try:
                    news = t.get_news()
                    titles = _extract_titles_from_news(news)
                except (AttributeError, Exception):
                    pass

            if not titles:
                continue

            titles = titles[:max_per_ticker]
            scores = [score_headline(t) for t in titles]
            n_total_articles += len(scores)

            if not scores:
                continue

            arr = np.array(scores)
            sentiment_data[ticker] = {
                "avg_sentiment": float(arr.mean()),
                "max_sentiment": float(arr.max()),
                "min_sentiment": float(arr.min()),
                "sentiment_std": float(arr.std()) if len(arr) > 1 else 0.0,
                "n_articles": len(scores),
                "positive_ratio": float((arr > 0).mean()),
                "negative_ratio": float((arr < 0).mean()),
            }
        except Exception:
            continue

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(sentiment_data, f)

    logger.info(f"Sentiment: {len(sentiment_data)} tickers, {n_total_articles} total articles")
    if len(sentiment_data) == 0:
        logger.warning(
            "No sentiment data extracted — generating synthetic sentiment for offline use."
        )
        from data_loader import _generate_synthetic_sentiment
        sentiment_data = _generate_synthetic_sentiment(tickers)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(sentiment_data, f)
    return sentiment_data


def build_sentiment_features(
    sentiment_data: Dict[str, dict], prices: pd.DataFrame,
) -> Dict[tuple, pd.DataFrame]:
    tickers = [t for t in prices.columns if t in sentiment_data]
    dates = prices.index
    feats = {}

    if not tickers:
        logger.warning("No sentiment data available — skipping sentiment features")
        return feats

    def _broadcast(values, name):
        series = pd.Series({t: values.get(t, np.nan) for t in prices.columns})
        return pd.DataFrame(
            np.tile(series.values, (len(dates), 1)),
            index=dates, columns=prices.columns,
        )

    def _rank_cs(df):
        return df.rank(axis=1, pct=True)

    avg_sent = {t: sentiment_data[t]["avg_sentiment"] for t in tickers}
    df = _broadcast(avg_sent, "avg_sentiment")
    feats[("sent", "avg_sentiment")] = df
    feats[("sent", "cs_rank_sentiment")] = _rank_cs(df)

    max_sent = {t: sentiment_data[t]["max_sentiment"] for t in tickers}
    feats[("sent", "cs_rank_max_sentiment")] = _rank_cs(_broadcast(max_sent, "max"))

    min_sent = {t: sentiment_data[t]["min_sentiment"] for t in tickers}
    feats[("sent", "cs_rank_min_sentiment")] = _rank_cs(_broadcast(min_sent, "min"))

    sent_std = {t: sentiment_data[t]["sentiment_std"] for t in tickers}
    feats[("sent", "sentiment_dispersion")] = _broadcast(sent_std, "std")
    feats[("sent", "cs_rank_sent_dispersion")] = _rank_cs(_broadcast(sent_std, "std"))

    n_articles = {t: float(sentiment_data[t]["n_articles"]) for t in tickers}
    feats[("sent", "cs_rank_news_volume")] = _rank_cs(_broadcast(n_articles, "n"))

    pos_ratio = {t: sentiment_data[t]["positive_ratio"] for t in tickers}
    feats[("sent", "positive_ratio")] = _broadcast(pos_ratio, "pos")
    feats[("sent", "cs_rank_positive_ratio")] = _rank_cs(_broadcast(pos_ratio, "pos"))

    # Composite
    rank_keys = ["cs_rank_sentiment", "cs_rank_positive_ratio", "cs_rank_max_sentiment"]
    rank_dfs = [feats[("sent", k)] for k in rank_keys if ("sent", k) in feats]
    if len(rank_dfs) >= 2:
        stacked = [df.stack() for df in rank_dfs]
        composite = pd.concat(stacked, axis=1).mean(axis=1).unstack()
        if isinstance(composite, pd.DataFrame) and not composite.empty:
            feats[("sent", "sentiment_composite")] = composite

    logger.info(f"Sentiment features: {len(feats)} signals from {len(tickers)} tickers")
    return feats
