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
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


class AlpacaExecutor:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        if not ALPACA_AVAILABLE:
            raise ImportError("pip install alpaca-py")
        self.client = TradingClient(api_key, api_secret, paper=paper)
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        logger.info(f"Alpaca: {'paper' if paper else 'LIVE'}")

    def get_account(self) -> dict:
        a = self.client.get_account()
        return {"equity": float(a.equity), "cash": float(a.cash),
                "buying_power": float(a.buying_power)}

    def get_positions(self) -> pd.Series:
        positions = self.client.get_all_positions()
        equity = float(self.client.get_account().equity)
        return pd.Series({p.symbol: float(p.market_value) / equity for p in positions})

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
                    order = self.client.submit_order(MarketOrderRequest(
                        symbol=ticker, qty=qty,
                        side=side, time_in_force=TimeInForce.DAY,
                    ))
                else:
                    order = self.client.submit_order(MarketOrderRequest(
                        symbol=ticker, notional=round(abs(dollar), 2),
                        side=side, time_in_force=TimeInForce.DAY,
                    ))

                results.append({"symbol": ticker, "side": side.value,
                                "notional": abs(dollar), "status": order.status.value})
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Order failed for {ticker}: {e}")
                results.append({"symbol": ticker, "status": "error", "error": str(e)})
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
            quotes = data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=tickers)
            )
            for ticker, quote in quotes.items():
                mid = (quote.ask_price + quote.bid_price) / 2
                prices[ticker] = mid if mid > 0 else quote.ask_price
        except Exception as e:
            logger.warning(f"Failed to fetch prices, using notional orders: {e}")
        return prices

    def close_all_positions(self):
        self.client.close_all_positions(cancel_orders=True)
