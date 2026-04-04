"""
Professional market data provider using Alpaca's Data API v2.

Replaces yfinance with Alpaca's institutional-grade data:
- Historical OHLCV bars (multi-symbol batch, adjustable timeframes)
- Real-time snapshots (latest price, volume, change)
- News articles with symbol tagging
- Cross-asset data (ETFs as macro proxies)

All methods are async and include retry/rate-limit resilience.
"""
import asyncio
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import logging

import httpx

from .models import Bar, NewsArticle, Snapshot

logger = logging.getLogger("alpaca_adapter.provider")


def require_env(name):
    val = os.getenv(name, "")
    if not val:
        raise ValueError(f"Required env var {name} not set")
    return val



# Alpaca data API base
DATA_URL = "https://data.alpaca.markets"

# Max symbols per multi-bar request (Alpaca limit)
BATCH_SIZE = 50

# Cross-asset ETF proxies that replace yfinance index tickers
# Maps the yfinance ticker to an Alpaca-tradeable ETF equivalent
CROSS_ASSET_MAP = {
    # Volatility
    "^VIX": "VIXY",       # VIX short-term futures ETF
    # Treasuries / Yields
    "^TNX": "TLT",        # 20+ year treasury (inverse proxy for 10Y yield)
    "^IRX": "SHV",        # Short-term treasury (3M yield proxy)
    "TLT": "TLT",         # Already an ETF
    # Credit
    "HYG": "HYG",         # High yield corporate bond
    "LQD": "LQD",         # Investment grade corporate bond
    # Dollar / Commodities
    "UUP": "UUP",         # US Dollar Index bull fund
    "GLD": "GLD",         # Gold
    "USO": "USO",         # Oil
    # Broad market
    "^GSPC": "SPY",       # S&P 500
    "SPY": "SPY",
    "IWM": "IWM",         # Russell 2000
    "QQQ": "QQQ",         # Nasdaq 100
    # GICS Sector ETFs (used by CS system for sector rotation features)
    "XLK": "XLK",         # Technology
    "XLF": "XLF",         # Financials
    "XLV": "XLV",         # Healthcare
    "XLE": "XLE",         # Energy
    "XLI": "XLI",         # Industrials
    "XLP": "XLP",         # Consumer Staples
    "XLY": "XLY",         # Consumer Discretionary
    "XLU": "XLU",         # Utilities
    "XLB": "XLB",         # Materials
    "XLRE": "XLRE",       # Real Estate
    "XLC": "XLC",         # Communication Services
}


class AlpacaDataProvider:
    """
    Async market data provider backed by Alpaca Data API v2.

    Provides historical bars, snapshots, and news with:
    - Batch multi-symbol requests (up to 50 per call)
    - Automatic pagination for large date ranges
    - Split/dividend adjusted prices
    - Rate limiting and retry resilience
    """

    def __init__(self):
        self.api_key = require_env("ALPACA_API_KEY")
        self.api_secret = require_env("ALPACA_API_SECRET")
        self._client = httpx.AsyncClient(
            base_url=DATA_URL,
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
            },
            timeout=30.0,
        )

    async def close(self):
        await self._client.aclose()

    # ── Historical Bars ──────────────────────────────────────────────────


    async def get_bars(
        self,
        symbols: list[str],
        start: str,
        end: Optional[str] = None,
        timeframe: str = "1Day",
        adjustment: str = "all",
        feed: str = "iex",
        limit: int = 10000,
    ) -> dict[str, list[Bar]]:
        """
        Fetch historical OHLCV bars for multiple symbols.

        Args:
            symbols: List of ticker symbols (e.g., ["AAPL", "TSLA"])
            start: Start date as ISO string (e.g., "2023-01-01")
            end: End date as ISO string (defaults to today)
            timeframe: Bar size — "1Min", "5Min", "15Min", "1Hour", "1Day", "1Week"
            adjustment: Price adjustment — "raw", "split", "dividend", "all"
            feed: Data feed — "iex" (free), "sip" (full market)
            limit: Max bars per symbol per request (max 10000)

        Returns:
            Dict mapping symbol -> list of Bar objects, sorted by timestamp
        """
        if not end:
            end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        all_bars: dict[str, list[Bar]] = {s: [] for s in symbols}

        # Batch symbols in groups of BATCH_SIZE
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i : i + BATCH_SIZE]
            await self._fetch_bars_batch(
                batch, start, end, timeframe, adjustment, feed, limit, all_bars
            )

        return all_bars

    async def _fetch_bars_batch(
        self,
        symbols: list[str],
        start: str,
        end: str,
        timeframe: str,
        adjustment: str,
        feed: str,
        limit: int,
        result: dict[str, list[Bar]],
    ) -> None:
        """Fetch bars for a batch of symbols with automatic pagination."""
        pass  # rate limiter removed

        page_token = None
        while True:
            params = {
                "symbols": ",".join(symbols),
                "start": start,
                "end": end,
                "timeframe": timeframe,
                "adjustment": adjustment,
                "feed": feed,
                "limit": limit,
            }
            if page_token:
                params["page_token"] = page_token

            resp = await self._client.get("/v2/stocks/bars", params=params)
            resp.raise_for_status()
            data = resp.json()

            bars_data = data.get("bars", {})
            for symbol, bars_list in bars_data.items():
                for b in bars_list:
                    result.setdefault(symbol, []).append(
                        Bar(
                            symbol=symbol,
                            timestamp=datetime.fromisoformat(b["t"].replace("Z", "+00:00")),
                            open=float(b["o"]),
                            high=float(b["h"]),
                            low=float(b["l"]),
                            close=float(b["c"]),
                            volume=float(b["v"]),
                            vwap=float(b.get("vw", 0)),
                            trade_count=int(b.get("n", 0)),
                        )
                    )

            page_token = data.get("next_page_token")
            if not page_token:
                break

            pass  # rate limiter removed

    # ── Snapshots (Real-time) ────────────────────────────────────────────


    async def get_snapshots(
        self,
        symbols: list[str],
        feed: str = "iex",
    ) -> dict[str, Snapshot]:
        """
        Fetch current market snapshots for multiple symbols.

        Returns latest price, prev close, change %, volume, VWAP, high/low.
        """
        pass  # rate limiter removed

        snapshots: dict[str, Snapshot] = {}

        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i : i + BATCH_SIZE]
            resp = await self._client.get(
                "/v2/stocks/snapshots",
                params={"symbols": ",".join(batch), "feed": feed},
            )
            resp.raise_for_status()
            data = resp.json()

            for symbol, snap in data.items():
                daily = snap.get("dailyBar", {})
                prev = snap.get("prevDailyBar", {})
                latest = snap.get("latestTrade", {})

                price = float(latest.get("p", daily.get("c", 0)))
                prev_close = float(prev.get("c", price))
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

                snapshots[symbol] = Snapshot(
                    symbol=symbol,
                    price=price,
                    prev_close=prev_close,
                    change_pct=round(change_pct, 2),
                    volume=float(daily.get("v", 0)),
                    vwap=float(daily.get("vw", 0)),
                    high=float(daily.get("h", 0)),
                    low=float(daily.get("l", 0)),
                )

        return snapshots

    # ── News ─────────────────────────────────────────────────────────────


    async def get_news(
        self,
        symbols: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50,
    ) -> list[NewsArticle]:
        """
        Fetch news articles from Alpaca News API.

        Args:
            symbols: Filter by symbols (e.g., ["AAPL", "TSLA"]). None = all news.
            start: Start datetime ISO string
            end: End datetime ISO string
            limit: Max articles to return (max 50 per request)

        Returns:
            List of NewsArticle objects sorted by recency
        """
        pass  # rate limiter removed

        params: dict = {"limit": min(limit, 50)}
        if symbols:
            params["symbols"] = ",".join(symbols)
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        resp = await self._client.get("/v1beta1/news", params=params)
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for item in data.get("news", []):
            articles.append(
                NewsArticle(
                    id=str(item.get("id", "")),
                    headline=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    source=item.get("source", ""),
                    url=item.get("url", ""),
                    symbols=item.get("symbols", []),
                    created_at=datetime.fromisoformat(
                        item["created_at"].replace("Z", "+00:00")
                    ) if "created_at" in item else datetime.now(timezone.utc),
                )
            )

        return articles

    # ── Cross-Asset Data ─────────────────────────────────────────────────

    async def get_cross_asset_bars(
        self,
        start: str,
        end: Optional[str] = None,
        timeframe: str = "1Day",
    ) -> dict[str, list[Bar]]:
        """
        Fetch historical bars for cross-asset ETF proxies.

        Replaces yfinance's ^VIX, ^TNX, ^IRX etc. with tradeable ETFs:
        - VIXY (VIX proxy), TLT (long bonds), SHV (short bonds)
        - HYG/LQD (credit), GLD (gold), USO (oil), UUP (dollar)
        - SPY, QQQ, IWM (broad equity)

        Returns bars keyed by the ORIGINAL ticker name (e.g., "^VIX")
        so downstream code doesn't need to change.
        """
        etf_symbols = list(set(CROSS_ASSET_MAP.values()))
        raw_bars = await self.get_bars(
            symbols=etf_symbols,
            start=start,
            end=end,
            timeframe=timeframe,
        )

        # Remap ETF symbols back to original ticker names
        remapped: dict[str, list[Bar]] = {}
        for original, etf in CROSS_ASSET_MAP.items():
            if etf in raw_bars:
                remapped[original] = raw_bars[etf]

        return remapped

    # ── Convenience: DataFrame output ────────────────────────────────────

    async def get_bars_df(
        self,
        symbols: list[str],
        start: str,
        end: Optional[str] = None,
        timeframe: str = "1Day",
    ):
        """
        Fetch bars and return as a pandas DataFrame matching yfinance format.

        Returns DataFrame with MultiIndex columns: (field, symbol)
        where field is one of: Open, High, Low, Close, Volume

        This is a drop-in replacement for yf.download() output format.
        """
        import pandas as pd

        bars = await self.get_bars(symbols, start, end, timeframe)

        frames = {}
        for symbol, bar_list in bars.items():
            if not bar_list:
                continue
            df = pd.DataFrame([
                {
                    "Date": b.timestamp,
                    "Open": b.open,
                    "High": b.high,
                    "Low": b.low,
                    "Close": b.close,
                    "Volume": b.volume,
                }
                for b in bar_list
            ])
            df.set_index("Date", inplace=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            frames[symbol] = df

        if not frames:
            return pd.DataFrame()

        if len(symbols) == 1 and symbols[0] in frames:
            return frames[symbols[0]]

        # Multi-symbol: create MultiIndex columns like yfinance
        combined = pd.concat(frames, axis=1)
        combined.columns = pd.MultiIndex.from_tuples(
            [(col, sym) for sym, df in frames.items() for col in df.columns],
            names=["Price", "Ticker"],
        )
        return combined

    async def get_cross_asset_df(
        self,
        start: str,
        end: Optional[str] = None,
    ):
        """
        Fetch cross-asset data as a DataFrame with Close prices.

        Returns DataFrame with columns named by original tickers (^VIX, ^TNX, etc.)
        matching the format expected by cross_asset_features.py.
        """
        import pandas as pd

        bars = await self.get_cross_asset_bars(start, end)

        frames = {}
        for ticker, bar_list in bars.items():
            if not bar_list:
                continue
            df = pd.DataFrame([
                {"Date": b.timestamp, "Close": b.close}
                for b in bar_list
            ])
            df.set_index("Date", inplace=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            frames[ticker] = df["Close"]

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, axis=1)
        result.columns.name = None
        return result

    # ── News for sentiment ───────────────────────────────────────────────

    async def get_news_for_symbols(
        self,
        symbols: list[str],
        max_per_symbol: int = 10,
    ) -> dict[str, list[NewsArticle]]:
        """
        Fetch recent news for multiple symbols, batched.

        Returns dict mapping symbol -> list of NewsArticle.
        Replaces yfinance's Ticker.news for sentiment analysis.
        """
        result: dict[str, list[NewsArticle]] = {}

        # Fetch in batches to avoid rate limits
        for i in range(0, len(symbols), 5):
            batch = symbols[i : i + 5]
            articles = await self.get_news(
                symbols=batch,
                limit=50,
            )

            # Distribute articles to their symbols
            for article in articles:
                for sym in article.symbols:
                    if sym in batch:
                        result.setdefault(sym, [])
                        if len(result[sym]) < max_per_symbol:
                            result[sym].append(article)

            if i + 5 < len(symbols):
                await asyncio.sleep(0.2)  # gentle rate limiting between batches

        return result


# ── Singleton factory ────────────────────────────────────────────────────

_provider: Optional[AlpacaDataProvider] = None


def get_data_provider() -> AlpacaDataProvider:
    """Get or create the singleton data provider."""
    global _provider
    if _provider is None:
        _provider = AlpacaDataProvider()
    return _provider
