"""
Data models for market data provider.
"""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Bar:
    """Single OHLCV bar."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0
    trade_count: int = 0


@dataclass
class NewsArticle:
    """Single news article from Alpaca news API."""
    id: str
    headline: str
    summary: str
    source: str
    url: str
    symbols: list[str]
    created_at: datetime
    sentiment: str = ""  # populated by downstream processing


@dataclass
class Snapshot:
    """Current market snapshot for a symbol."""
    symbol: str
    price: float
    prev_close: float
    change_pct: float
    volume: float
    vwap: float
    high: float
    low: float
    timestamp: datetime = field(default_factory=datetime.now)
