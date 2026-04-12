"""
Institutional-grade labeling and sample weighting for daily models.

Based on López de Prado, "Advances in Financial Machine Learning":
1. Triple barrier labeling for daily predictions
2. Sample uniqueness weighting
3. Fractional differentiation for stationary features
4. Meta-labeling for bet sizing

These replace the simple "10-day forward return" target with a more
accurate representation of trade outcomes.
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ── Triple Barrier for Daily Model ───────────────────────────────────────

def daily_triple_barrier(
    prices: pd.DataFrame,
    entry_date: pd.Timestamp,
    ticker: str,
    vol_lookback: int = 63,
    upper_mult: float = 2.0,
    lower_mult: float = 2.0,
    max_holding_days: int = 10,
) -> Tuple[int, float]:
    """
    Triple barrier labeling for a daily-frequency entry.

    Barriers:
    - Upper: entry × (1 + daily_vol × upper_mult × sqrt(holding_period))
    - Lower: entry × (1 - daily_vol × lower_mult × sqrt(holding_period))
    - Time: max_holding_days forward

    Volatility scales with sqrt(time) — wider barriers for longer holding.

    Returns:
        (label, return_at_exit)
        label: +1 hit upper, -1 hit lower, 0 timeout
    """
    if ticker not in prices.columns:
        return 0, 0.0

    px = prices[ticker].dropna()
    if entry_date not in px.index:
        return 0, 0.0

    entry_idx = px.index.get_loc(entry_date)
    entry_price = px.iloc[entry_idx]

    # Compute realized volatility for barrier sizing
    lookback_start = max(0, entry_idx - vol_lookback)
    lookback_prices = px.iloc[lookback_start:entry_idx]
    if len(lookback_prices) < 20:
        daily_vol = 0.02  # default 2% daily vol
    else:
        daily_returns = lookback_prices.pct_change().dropna()
        daily_vol = daily_returns.std()
        daily_vol = max(daily_vol, 0.005)  # floor

    # Scan forward
    end_idx = min(entry_idx + max_holding_days, len(px) - 1)

    for days_forward in range(1, end_idx - entry_idx + 1):
        future_idx = entry_idx + days_forward
        future_price = px.iloc[future_idx]

        # Barriers scale with sqrt(time)
        time_scale = np.sqrt(days_forward)
        upper = entry_price * (1 + daily_vol * upper_mult * time_scale)
        lower = entry_price * (1 - daily_vol * lower_mult * time_scale)

        if future_price >= upper:
            return 1, (future_price - entry_price) / entry_price
        if future_price <= lower:
            return -1, (future_price - entry_price) / entry_price

    # Timeout
    exit_price = px.iloc[end_idx]
    return 0, (exit_price - entry_price) / entry_price


def label_dataset(
    prices: pd.DataFrame,
    entry_dates: list,
    tickers: list[str],
    max_holding_days: int = 10,
    upper_mult: float = 2.0,
    lower_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Label an entire dataset using triple barrier.

    Returns DataFrame with columns: ticker, date, label, return_pct
    """
    results = []
    for date in entry_dates:
        for ticker in tickers:
            label, ret = daily_triple_barrier(
                prices, date, ticker, max_holding_days=max_holding_days,
                upper_mult=upper_mult, lower_mult=lower_mult,
            )
            results.append({
                "date": date,
                "ticker": ticker,
                "label": label,
                "return_pct": ret,
            })

    return pd.DataFrame(results)


# ── Sample Uniqueness Weighting ──────────────────────────────────────────

def compute_sample_uniqueness(
    labels_df: pd.DataFrame,
    max_holding_days: int = 10,
) -> np.ndarray:
    """
    Compute sample uniqueness weights for cross-sectional data.

    Samples that overlap in time carry redundant information.
    We weight each sample by the inverse of its average concurrency.

    For daily cross-sectional models, the main overlap is between
    predictions at date t and t+1 (since both look 10 days forward,
    they share 9 days of the outcome).
    """
    n = len(labels_df)
    if n == 0:
        return np.array([])

    # For cross-sectional models, concurrency is primarily temporal
    # Adjacent dates share (max_holding_days - 1) / max_holding_days of information
    weights = np.ones(n)

    if "date" in labels_df.columns:
        dates = pd.to_datetime(labels_df["date"])
        unique_dates = dates.unique()

        for i in range(n):
            # Count how many other samples share the same outcome window
            date_i = dates.iloc[i]
            # Samples within max_holding_days overlap
            overlapping = np.sum(
                (dates >= date_i - pd.Timedelta(days=max_holding_days)) &
                (dates <= date_i + pd.Timedelta(days=max_holding_days))
            )
            weights[i] = 1.0 / max(overlapping / len(unique_dates), 0.1)

    # Normalize to mean of 1
    weights = weights * n / weights.sum()
    return weights


# ── Fractional Differentiation ───────────────────────────────────────────

def frac_diff_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute weights for fractional differentiation.

    d=0: no differentiation (original series, non-stationary)
    d=1: first difference (stationary but loses memory)
    d=0.3-0.5: sweet spot — stationary while preserving memory

    Reference: López de Prado Ch. 5
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights[::-1])


def frac_diff(series: pd.Series, d: float = 0.4, threshold: float = 1e-5) -> pd.Series:
    """
    Apply fractional differentiation to a price series.

    Args:
        series: price series (e.g., close prices)
        d: differentiation order (0.3-0.5 recommended)
        threshold: weight cutoff

    Returns:
        Fractionally differentiated series (stationary, memory-preserving)
    """
    weights = frac_diff_weights(d, threshold)
    width = len(weights)

    result = pd.Series(index=series.index, dtype=float)
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1:i + 1].values
        result.iloc[i] = np.dot(weights, window)

    return result.dropna()


def add_frac_diff_features(
    prices: pd.DataFrame,
    d: float = 0.4,
    columns: list[str] = None,
) -> pd.DataFrame:
    """
    Add fractionally differentiated price features to the dataset.

    For each stock's close price, adds a frac_diff column that is
    stationary (unlike raw price) but preserves memory (unlike returns).
    """
    if columns is None:
        columns = list(prices.columns)

    frac_features = {}
    for col in columns:
        if col in prices.columns:
            fd = frac_diff(prices[col], d=d)
            frac_features[f"frac_diff_{col}"] = fd

    return pd.DataFrame(frac_features)


# ── Meta-Labeling for Daily Model ────────────────────────────────────────

def create_daily_meta_labels(
    primary_predictions: pd.Series,
    triple_barrier_labels: pd.Series,
) -> pd.Series:
    """
    Create meta-labels for the daily model.

    Meta-label = 1 if primary model's direction was correct
                 0 if incorrect

    The primary model predicts return direction.
    The meta-model predicts whether to trust the primary.
    """
    pred_direction = np.sign(primary_predictions)
    correct = (pred_direction == triple_barrier_labels).astype(float)
    # Timeouts (label=0) → mark as 0.5 (uncertain)
    correct[triple_barrier_labels == 0] = 0.5
    return correct


# build_daily_meta_features removed: used .rank(pct=True) on full series (look-ahead)
# and was not imported anywhere. Reintroduce only with rolling pct rank if needed.
