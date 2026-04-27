"""
Order execution via Alpaca Markets API.
Supports paper trading (default) and live trading.

SETUP:
1. Create free account at https://alpaca.markets
2. Get API keys
3. export ALPACA_API_KEY="..." ALPACA_API_SECRET="..."
"""
import logging
import time
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Transient error substrings that warrant retry
_TRANSIENT_ERRORS = ("timeout", "connection", "503", "502", "429", "rate limit", "reset by peer")
_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_MAX_DELAY = 10.0


def _is_transient(e: Exception) -> bool:
    err = str(e).lower()
    return any(s in err for s in _TRANSIENT_ERRORS)


def _retry_api(func, *args, **kwargs):
    """Call func with exponential backoff retry on transient errors."""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not _is_transient(e) or attempt == _MAX_RETRIES:
                raise
            delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
            logger.warning(f"Transient error (attempt {attempt + 1}/{_MAX_RETRIES}): {e}. "
                           f"Retrying in {delay:.1f}s")
            time.sleep(delay)


class AlpacaExecutor:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        if not ALPACA_AVAILABLE:
            raise ImportError("pip install alpaca-py")
        self.client = TradingClient(api_key, api_secret, paper=paper)
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        logger.info(f"Alpaca: {'paper' if paper else 'LIVE'}")

    def health_check(self) -> bool:
        """Verify Alpaca API is reachable before trading."""
        try:
            self.get_account()
            return True
        except Exception as e:
            logger.error(f"Alpaca health check failed: {e}")
            return False

    def get_account(self) -> dict:
        a = _retry_api(self.client.get_account)
        return {"equity": float(a.equity), "cash": float(a.cash),
                "buying_power": float(a.buying_power)}

    def get_positions(self) -> pd.Series:
        positions = _retry_api(self.client.get_all_positions)
        equity = float(_retry_api(self.client.get_account).equity)
        return pd.Series({p.symbol: float(p.market_value) / equity for p in positions})

    def _wait_for_fill(self, order_id: str, timeout_sec: int = 60) -> dict:
        """Poll order status until terminal state or timeout."""
        start = time.time()
        terminal = {"filled", "canceled", "expired", "rejected"}
        while time.time() - start < timeout_sec:
            try:
                order = _retry_api(self.client.get_order_by_id, order_id)
                status = order.status.value if hasattr(order.status, 'value') else str(order.status)
                if status in terminal:
                    filled_qty = float(order.filled_qty or 0)
                    filled_price = float(order.filled_avg_price or 0)
                    if status == "filled":
                        logger.info(f"  {order.symbol}: filled {filled_qty} @ ${filled_price:.2f}")
                    elif status in ("canceled", "expired", "rejected"):
                        logger.warning(f"  {order.symbol}: {status} "
                                       f"(filled {filled_qty} of requested)")
                    return {
                        "status": status,
                        "filled_qty": filled_qty,
                        "filled_avg_price": filled_price,
                    }
            except Exception as e:
                logger.debug(f"Order status poll error: {e}")
            time.sleep(2)
        logger.warning(f"Order {order_id} timed out after {timeout_sec}s")
        return {"status": "timeout", "order_id": order_id}

    def execute_target_portfolio(self, target_weights: pd.Series) -> List[dict]:
        account = self.get_account()
        equity = account["equity"]
        current = self.get_positions()

        all_tickers = target_weights.index.union(current.index)
        trades = target_weights.reindex(all_tickers, fill_value=0) - current.reindex(all_tickers, fill_value=0)
        trades = trades[trades.abs() > max(500 / equity, 0.005)]

        if trades.empty:
            logger.info("No trades needed")
            return []

        # Fetch latest prices for qty-based orders
        prices = self._get_latest_prices(list(trades.index))

        results = []
        n_errors = 0
        for ticker, wt_chg in trades.items():
            dollar = wt_chg * equity
            try:
                side = OrderSide.BUY if dollar > 0 else OrderSide.SELL

                # Use qty for shorts (notional not supported for short selling)
                if dollar < 0 and ticker in prices and prices[ticker] > 0:
                    qty = int(abs(dollar) / prices[ticker])  # Whole shares only for shorts
                    if qty < 1:
                        logger.warning(f"Skipping {ticker}: short qty < 1 share")
                        continue
                    order = _retry_api(
                        self.client.submit_order,
                        MarketOrderRequest(
                            symbol=ticker, qty=qty,
                            side=side, time_in_force=TimeInForce.DAY,
                        ),
                    )
                else:
                    order = _retry_api(
                        self.client.submit_order,
                        MarketOrderRequest(
                            symbol=ticker, notional=round(abs(dollar), 2),
                            side=side, time_in_force=TimeInForce.DAY,
                        ),
                    )

                # Track fill status
                fill = self._wait_for_fill(order.id, timeout_sec=30)
                results.append({
                    "symbol": ticker, "side": side.value,
                    "notional": abs(dollar),
                    "status": fill.get("status", order.status.value),
                    "filled_qty": fill.get("filled_qty", 0),
                    "filled_avg_price": fill.get("filled_avg_price", 0),
                })
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Order failed for {ticker}: {e}")
                results.append({"symbol": ticker, "status": "error", "error": str(e)})
                n_errors += 1
                # Circuit breaker: if >50% of orders fail, stop
                if n_errors > len(trades) * 0.5:
                    logger.error("Circuit breaker: >50% of orders failed, halting execution")
                    break

        # Summary
        filled = sum(1 for r in results if r.get("status") == "filled")
        errored = sum(1 for r in results if r.get("status") == "error")
        logger.info(f"Execution summary: {filled} filled, {errored} errors, "
                    f"{len(results)} total of {len(trades)} planned")
        return results

    def _get_latest_prices(self, tickers: list) -> dict:
        """Fetch latest prices for qty-based order calculation."""
        prices = {}
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            data_client = StockHistoricalDataClient(
                self.api_key, self.api_secret,
            )
            quotes = _retry_api(
                data_client.get_stock_latest_quote,
                StockLatestQuoteRequest(symbol_or_symbols=tickers),
            )
            for ticker, quote in quotes.items():
                mid = (quote.ask_price + quote.bid_price) / 2
                prices[ticker] = mid if mid > 0 else quote.ask_price
        except Exception as e:
            logger.warning(f"Failed to fetch prices, using notional orders: {e}")
        return prices

    def close_all_positions(self):
        _retry_api(self.client.close_all_positions, cancel_orders=True)
