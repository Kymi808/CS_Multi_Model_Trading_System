"""
Prediction accuracy tracker — compares predicted vs realized returns.

Computes rolling Information Coefficient (IC) to detect model decay.
IC = rank correlation between predicted scores and realized returns.

Usage:
    python prediction_tracker.py              # check all logged predictions
    python prediction_tracker.py --horizon 10 # custom horizon (default: 10 days)

A healthy model maintains IC > 0.02. Sustained IC < 0.02 signals model decay
and warrants investigation (data drift, regime change, feature degradation).
"""
import json
import os
import glob
import logging
import argparse
from datetime import timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
IC_DECAY_THRESHOLD = 0.02
IC_ROLLING_WINDOW = 20  # trading days


def load_prediction_logs() -> list[dict]:
    """Load all prediction log files from results/."""
    pattern = os.path.join(RESULTS_DIR, "predictions_*.json")
    files = sorted(glob.glob(pattern))
    logs = []
    for f in files:
        try:
            with open(f) as fh:
                logs.append(json.load(fh))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Skipping corrupt prediction log {f}: {e}")
    return logs


def fetch_realized_returns(tickers: list, pred_date: str, horizon: int = 10) -> pd.Series:
    """Fetch realized returns for tickers over the prediction horizon."""
    try:
        import yfinance as yf
        start = pd.Timestamp(pred_date)
        end = start + timedelta(days=horizon * 2)  # buffer for weekends
        data = yf.download(
            tickers, start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False,
        )
        if data.empty:
            return pd.Series(dtype=float)
        prices = data["Close"] if len(tickers) > 1 else data["Close"].to_frame(tickers[0])
        # Forward return from pred_date over horizon trading days
        if len(prices) < horizon + 1:
            return pd.Series(dtype=float)
        returns = prices.iloc[horizon] / prices.iloc[0] - 1
        return returns.dropna()
    except Exception as e:
        logger.warning(f"Failed to fetch realized returns for {pred_date}: {e}")
        return pd.Series(dtype=float)


def compute_ic(predictions: pd.Series, realized: pd.Series) -> float:
    """Compute rank IC (Spearman correlation) between predicted and realized."""
    common = predictions.index.intersection(realized.index)
    if len(common) < 10:
        return np.nan
    return predictions.reindex(common).rank().corr(realized.reindex(common).rank())


def run_tracker(horizon: int = 10):
    """Main tracking loop: score all mature predictions."""
    logs = load_prediction_logs()
    if not logs:
        print("No prediction logs found in results/")
        return

    today = pd.Timestamp.now().normalize()
    ic_history = []

    for log in logs:
        pred_date = log["date"]
        pred_ts = pd.Timestamp(pred_date)

        # Only score predictions that are mature (horizon days have passed)
        if today - pred_ts < timedelta(days=horizon + 2):
            continue

        predictions = pd.Series(log["predictions"]).astype(float)
        tickers = list(predictions.index)
        realized = fetch_realized_returns(tickers, pred_date, horizon)

        if realized.empty:
            continue

        ic = compute_ic(predictions, realized)
        if not np.isnan(ic):
            ic_history.append({"date": pred_date, "ic": ic, "n_stocks": len(realized)})

    if not ic_history:
        print("No mature predictions to score yet (need {horizon}+ trading days of history)")
        return

    df = pd.DataFrame(ic_history).set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Rolling IC
    rolling_ic = df["ic"].rolling(IC_ROLLING_WINDOW, min_periods=5).mean()

    print(f"\n{'='*60}")
    print("PREDICTION ACCURACY TRACKER")
    print(f"{'='*60}")
    print(f"  Predictions scored:  {len(df)}")
    print(f"  Horizon:             {horizon} trading days")
    print(f"  Mean IC:             {df['ic'].mean():.4f}")
    print(f"  Median IC:           {df['ic'].median():.4f}")
    print(f"  IC Std:              {df['ic'].std():.4f}")
    print(f"  IC > 0 rate:         {(df['ic'] > 0).mean():.1%}")

    if len(rolling_ic.dropna()) > 0:
        latest_rolling = rolling_ic.dropna().iloc[-1]
        print(f"  Rolling {IC_ROLLING_WINDOW}d IC:    {latest_rolling:.4f}")

        if latest_rolling < IC_DECAY_THRESHOLD:
            print(f"\n  WARNING: Rolling IC ({latest_rolling:.4f}) below decay "
                  f"threshold ({IC_DECAY_THRESHOLD}). Model may need retraining.")
        else:
            print(f"\n  Model signal healthy (IC > {IC_DECAY_THRESHOLD})")

    print("\nRecent ICs:")
    for _, row in df.tail(10).iterrows():
        ic_val = row["ic"]
        bar = "+" * int(max(0, ic_val) * 100) if ic_val > 0 else "-" * int(abs(ic_val) * 100)
        flag = " <-- WEAK" if ic_val < IC_DECAY_THRESHOLD else ""
        print(f"  {row.name.date()}  IC={ic_val:+.4f}  n={int(row['n_stocks'])}  {bar}{flag}")

    # Save IC history
    ic_path = os.path.join(RESULTS_DIR, "ic_history.csv")
    df.to_csv(ic_path)
    print(f"\nIC history saved to {ic_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Track prediction accuracy over time")
    parser.add_argument("--horizon", type=int, default=10, help="Prediction horizon in trading days")
    args = parser.parse_args()
    run_tracker(horizon=args.horizon)
