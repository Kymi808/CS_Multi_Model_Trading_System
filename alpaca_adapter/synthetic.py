"""
Synthetic data generation for offline/fallback scenarios.
Matches the CS system's synthetic data generators.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def generate_synthetic_prices(
    tickers: List[str], start_date: str, end_date: str, seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic synthetic price and volume data."""
    logger.info(f"Generating synthetic price data for {len(tickers)} tickers")
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_tickers = len(tickers)
    n_days = len(dates)

    market_returns = rng.normal(0.0004, 0.01, n_days)
    all_returns = np.zeros((n_days, n_tickers))
    for i in range(n_tickers):
        beta = 0.5 + rng.random() * 1.0
        idio_vol = 0.005 + rng.random() * 0.015
        drift = rng.normal(0.0002, 0.0003)
        idio = rng.normal(drift, idio_vol, n_days)
        all_returns[:, i] = beta * market_returns + idio

    start_prices = 50 + rng.random(n_tickers) * 400
    prices_arr = np.zeros((n_days, n_tickers))
    prices_arr[0] = start_prices
    for t in range(1, n_days):
        prices_arr[t] = prices_arr[t - 1] * (1 + all_returns[t])

    prices = pd.DataFrame(prices_arr, index=dates, columns=tickers)

    base_volumes = 1e6 + rng.random(n_tickers) * 9e6
    vol_noise = rng.lognormal(0, 0.3, (n_days, n_tickers))
    abs_ret_factor = 1 + 5 * np.abs(all_returns)
    volumes_arr = base_volumes * vol_noise * abs_ret_factor
    volumes = pd.DataFrame(volumes_arr.astype(int), index=dates, columns=tickers)

    return prices, volumes


def generate_synthetic_cross_asset(
    tickers: List[str], start_date: str, end_date: str, seed: int = 123,
) -> pd.DataFrame:
    """Generate synthetic cross-asset data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    data = {}
    for ticker in tickers:
        if "VIX" in ticker:
            vals = np.zeros(n_days)
            vals[0] = 18.0
            for t in range(1, n_days):
                vals[t] = vals[t - 1] + 0.05 * (18 - vals[t - 1]) + rng.normal(0, 1.2)
                vals[t] = max(10, min(45, vals[t]))
        elif "TNX" in ticker or "IRX" in ticker:
            base = 4.0 if "TNX" in ticker else 5.0
            vals = np.cumsum(rng.normal(0, 0.02, n_days)) + base
        else:
            start_price = 50 + rng.random() * 400
            returns = rng.normal(0.0003, 0.01, n_days)
            vals = start_price * np.cumprod(1 + returns)
        data[ticker] = vals

    return pd.DataFrame(data, index=dates)
