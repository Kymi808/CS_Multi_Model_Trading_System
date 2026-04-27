"""
LLM-powered sentiment analysis using Claude Haiku.

Replaces word-counting with context-aware article analysis:
1. Claude reads each article headline + summary
2. Scores sentiment with reasoning (not just keyword matching)
3. Classifies event type (earnings, analyst, regulatory, insider, macro, M&A)
4. Scores magnitude (how big is the impact?)
5. Applies temporal decay (recent articles matter more)

Cost: ~$0.02/day using Haiku with selective analysis (~30 articles).

Design decisions:
- Haiku not Sonnet (12x cheaper, fast enough for sentiment scoring)
- Batch 5 articles per API call (fewer calls, more context for relative scoring)
- Only analyze articles for portfolio stocks (not full universe)
- Cache results to avoid re-analyzing the same article
"""
import json
import os
import hashlib
from datetime import datetime, timezone

import httpx

import logging
from pathlib import Path

logger_module = logging.getLogger("alpaca_adapter.news")


def safe_load_state(path, default=None):
    """Standalone safe JSON loader."""
    default = default if default is not None else {}
    if not path.exists():
        return dict(default)
    try:
        import json as _json
        data = _json.loads(path.read_text())
        return data if isinstance(data, dict) else dict(default)
    except Exception:
        return dict(default)


def safe_save_state(path, state):
    """Standalone safe JSON writer."""
    import json as _json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(state, indent=2))
    tmp.rename(path)

logger = logging.getLogger("alpaca_adapter.news.llm_sentiment")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Cache to avoid re-analyzing the same article
CACHE_FILE = Path("./data/sentiment_cache.json")
MAX_CACHE_SIZE = 2000  # max cached articles

# Temporal decay: articles lose relevance over time
DECAY_HALF_LIFE_HOURS = 6  # sentiment halves in relevance every 6 hours


def _get_api_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "")


def _article_hash(headline: str, source: str) -> str:
    """Unique ID for an article to avoid re-analysis."""
    return hashlib.md5(f"{headline}:{source}".encode()).hexdigest()[:12]


async def analyze_articles_batch(
    articles: list[dict],
    context: str = "",
) -> list[dict]:
    """
    Analyze a batch of articles using Claude Haiku.

    Sends up to 5 articles per API call for efficiency.
    Each article gets: sentiment score, event type, magnitude, reasoning.

    Args:
        articles: list of {headline, summary, symbols, source, created_at}
        context: optional market context (regime, VIX level, etc.)

    Returns:
        list of {article_hash, sentiment, event_type, magnitude, reasoning, decay_weight}
    """
    api_key = _get_api_key()
    if not api_key or api_key.startswith("sk-ant-xxx"):
        logger.debug("Anthropic API key not available — skipping LLM sentiment")
        return []

    # Load cache
    cache = safe_load_state(CACHE_FILE, {"articles": {}})
    cached_articles = cache.get("articles", {})

    results = []
    uncached = []

    # Check cache first
    for article in articles:
        h = _article_hash(article.get("headline", ""), article.get("source", ""))
        if h in cached_articles:
            cached = cached_articles[h]
            # Apply temporal decay to cached result
            cached["decay_weight"] = _compute_decay(article.get("created_at", ""))
            results.append(cached)
        else:
            uncached.append(article)

    if not uncached:
        return results

    # Batch uncached articles (5 per API call)
    for i in range(0, len(uncached), 5):
        batch = uncached[i:i + 5]
        batch_results = await _analyze_batch(batch, context, api_key)
        results.extend(batch_results)

        # Cache results
        for r in batch_results:
            cached_articles[r["article_hash"]] = r

    # Trim cache
    if len(cached_articles) > MAX_CACHE_SIZE:
        # Keep most recent
        sorted_items = sorted(cached_articles.items(), key=lambda x: x[1].get("analyzed_at", ""))
        cached_articles = dict(sorted_items[-MAX_CACHE_SIZE:])

    cache["articles"] = cached_articles
    safe_save_state(CACHE_FILE, cache)

    return results


async def _analyze_batch(
    articles: list[dict],
    context: str,
    api_key: str,
) -> list[dict]:
    """Send a batch of articles to Claude Haiku for analysis."""
    # Build prompt
    article_text = ""
    for i, a in enumerate(articles, 1):
        symbols = ", ".join(a.get("symbols", [])[:5])
        article_text += (
            f"\nArticle {i}:\n"
            f"  Headline: {a.get('headline', '')}\n"
            f"  Summary: {a.get('summary', '')[:300]}\n"
            f"  Symbols: {symbols}\n"
            f"  Source: {a.get('source', '')}\n"
            f"  Time: {a.get('created_at', '')}\n"
        )

    prompt = f"""Analyze these financial news articles for trading sentiment.

{f"Market context: {context}" if context else ""}

{article_text}

For EACH article, provide a JSON object with:
- "article_num": the article number (1, 2, etc.)
- "sentiment": float from -1.0 (very bearish) to +1.0 (very bullish). 0 = neutral.
- "event_type": one of ["earnings", "analyst", "regulatory", "insider", "macro", "ma", "product", "legal", "other"]
- "magnitude": float from 0.0 (insignificant) to 1.0 (market-moving). How impactful is this?
- "reasoning": one sentence explaining your score

Important:
- Consider CONTEXT, not just keywords. "Beat estimates but lowered guidance" is NEGATIVE.
- "magnitude" should reflect actual impact: an FDA approval is high magnitude, a routine filing is low.
- For macro articles affecting all stocks, score the broad market impact.

Respond with ONLY a JSON array of objects. No other text."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": HAIKU_MODEL,
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"]

            # Parse JSON response
            # Handle potential markdown code blocks
            if "```" in text:
                text = text.split("```json")[-1].split("```")[0]
            analyses = json.loads(text.strip())

    except Exception as e:
        logger.warning(f"LLM sentiment batch failed: {e}")
        return []

    # Map back to articles
    results = []
    now = datetime.now(timezone.utc).isoformat()

    for analysis in analyses:
        idx = analysis.get("article_num", 0) - 1
        if 0 <= idx < len(articles):
            article = articles[idx]
            h = _article_hash(article.get("headline", ""), article.get("source", ""))

            results.append({
                "article_hash": h,
                "headline": article.get("headline", ""),
                "symbols": article.get("symbols", []),
                "sentiment": float(analysis.get("sentiment", 0)),
                "event_type": analysis.get("event_type", "other"),
                "magnitude": float(analysis.get("magnitude", 0.5)),
                "reasoning": analysis.get("reasoning", ""),
                "decay_weight": _compute_decay(article.get("created_at", "")),
                "analyzed_at": now,
            })

    logger.info(f"LLM analyzed {len(results)} articles (batch of {len(articles)})")
    return results


def _compute_decay(created_at: str) -> float:
    """
    Compute temporal decay weight for an article.

    Articles lose relevance exponentially: half-life of 6 hours.
    A 1-hour-old article has weight ~0.89
    A 6-hour-old article has weight ~0.50
    A 24-hour-old article has weight ~0.06
    """
    if not created_at:
        return 0.5

    try:
        if isinstance(created_at, str):
            article_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            article_time = created_at

        age_hours = (datetime.now(timezone.utc) - article_time).total_seconds() / 3600
        decay = 0.5 ** (age_hours / DECAY_HALF_LIFE_HOURS)
        return max(0.01, min(1.0, decay))
    except Exception:
        return 0.5


def compute_llm_sentiment_features(
    analyses: list[dict],
    symbol: str,
) -> dict[str, float]:
    """
    Compute sentiment features for a single stock from LLM analyses.

    Returns features suitable for the ML model:
    - llm_sentiment_avg: decay-weighted average sentiment
    - llm_sentiment_magnitude: average magnitude of relevant articles
    - llm_sentiment_volume: number of articles (information flow)
    - llm_event_earnings: 1 if recent earnings-related news
    - llm_event_analyst: 1 if recent analyst action
    - llm_event_regulatory: 1 if recent regulatory news
    - llm_event_ma: 1 if recent M&A news
    """
    # Filter to articles mentioning this symbol
    relevant = [a for a in analyses if symbol in a.get("symbols", [])]

    if not relevant:
        return {
            "llm_sentiment_avg": 0.0,
            "llm_sentiment_magnitude": 0.0,
            "llm_sentiment_volume": 0.0,
            "llm_event_earnings": 0.0,
            "llm_event_analyst": 0.0,
            "llm_event_regulatory": 0.0,
            "llm_event_ma": 0.0,
        }

    # Decay-weighted sentiment
    total_weight = sum(a["decay_weight"] * a["magnitude"] for a in relevant)
    if total_weight > 0:
        weighted_sent = sum(
            a["sentiment"] * a["decay_weight"] * a["magnitude"]
            for a in relevant
        ) / total_weight
    else:
        weighted_sent = 0.0

    avg_magnitude = sum(a["magnitude"] for a in relevant) / len(relevant)

    # Event type flags
    event_types = [a["event_type"] for a in relevant]

    return {
        "llm_sentiment_avg": round(weighted_sent, 4),
        "llm_sentiment_magnitude": round(avg_magnitude, 4),
        "llm_sentiment_volume": min(len(relevant) / 10, 1.0),  # normalize to 0-1
        "llm_event_earnings": 1.0 if "earnings" in event_types else 0.0,
        "llm_event_analyst": 1.0 if "analyst" in event_types else 0.0,
        "llm_event_regulatory": 1.0 if "regulatory" in event_types else 0.0,
        "llm_event_ma": 1.0 if "ma" in event_types else 0.0,
    }
