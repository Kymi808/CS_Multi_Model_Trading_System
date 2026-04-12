"""
Diagnostic logging for backtest runs — captures per-day, per-position, per-window
state so post-run look-back analysis can identify what went wrong on each negative
step and test counterfactual filters/hypotheses.

Design principles:
- Cheap during runs (append to in-memory list, dump to JSON Lines at end)
- Optional (gated by config flag)
- Self-contained file format (JSON Lines = one record per line, easy to load)
- Schema-stable (fields can be added but never renamed; missing fields default
  to None)

Output files (saved to results/{sleeve_dir}/diagnostics/):
- day_diagnostics_{model}_{suffix}.jsonl   — 1 record per trading day
- position_events_{model}_{suffix}.jsonl   — 1 record per position open/close
- window_diagnostics_{model}_{suffix}.jsonl — 1 record per walk-forward window

Usage:
    logger = DiagnosticLogger(enabled=True, output_dir="results/sleeve_10d/diagnostics")
    logger.log_day({...})
    logger.log_position_event({...})
    logger.log_window({...})
    logger.flush(model_type="lightgbm", suffix="100L_30S")
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        # Convert NaN/Inf to None — JSON spec doesn't support them
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_to_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    # Fall back to string
    return str(obj)


class DiagnosticLogger:
    """In-memory diagnostic recorder. Flushed to disk at end of run."""

    def __init__(self, enabled: bool = True, output_dir: Optional[str] = None):
        self.enabled = enabled
        self.output_dir = output_dir
        self._day_records: List[dict] = []
        self._position_events: List[dict] = []
        self._window_records: List[dict] = []

    def log_day(self, record: dict) -> None:
        if not self.enabled:
            return
        self._day_records.append(_to_serializable(record))

    def log_position_event(self, record: dict) -> None:
        if not self.enabled:
            return
        self._position_events.append(_to_serializable(record))

    def log_window(self, record: dict) -> None:
        if not self.enabled:
            return
        self._window_records.append(_to_serializable(record))

    def flush(self, model_type: str = "lightgbm", suffix: str = "") -> Dict[str, str]:
        """
        Write all buffered records to JSON Lines files.
        Returns dict of {kind: filepath}.
        """
        if not self.enabled or not self.output_dir:
            return {}

        os.makedirs(self.output_dir, exist_ok=True)
        files_written = {}

        suffix_part = f"_{suffix}" if suffix else ""

        if self._day_records:
            path = os.path.join(self.output_dir, f"day_diagnostics_{model_type}{suffix_part}.jsonl")
            with open(path, "w") as f:
                for rec in self._day_records:
                    f.write(json.dumps(rec) + "\n")
            files_written["day"] = path
            logger.info(f"Diagnostics: wrote {len(self._day_records)} day records → {path}")

        if self._position_events:
            path = os.path.join(self.output_dir, f"position_events_{model_type}{suffix_part}.jsonl")
            with open(path, "w") as f:
                for rec in self._position_events:
                    f.write(json.dumps(rec) + "\n")
            files_written["position"] = path
            logger.info(f"Diagnostics: wrote {len(self._position_events)} position events → {path}")

        if self._window_records:
            path = os.path.join(self.output_dir, f"window_diagnostics_{model_type}{suffix_part}.jsonl")
            with open(path, "w") as f:
                for rec in self._window_records:
                    f.write(json.dumps(rec) + "\n")
            files_written["window"] = path
            logger.info(f"Diagnostics: wrote {len(self._window_records)} window records → {path}")

        return files_written

    def reset(self):
        """Clear in-memory buffers (e.g. between models in a comparison run)."""
        self._day_records = []
        self._position_events = []
        self._window_records = []


# ============================================================
# LOADER FUNCTIONS — for use in look-back analyzer
# ============================================================

def load_day_diagnostics(path: str) -> pd.DataFrame:
    """Load day diagnostics into a DataFrame indexed by date."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df


def load_position_events(path: str) -> pd.DataFrame:
    """Load position events into a DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    for col in ("date", "entry_date", "exit_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_window_diagnostics(path: str) -> pd.DataFrame:
    """Load window diagnostics into a DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    for col in ("train_start", "train_end", "predict_start", "predict_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def reconstruct_trades(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct closed trades from open/close events.
    Returns one row per completed round-trip.
    """
    if events_df.empty:
        return pd.DataFrame()

    opens = events_df[events_df["event_type"] == "open"].copy()
    closes = events_df[events_df["event_type"] == "close"].copy()

    # Match open events to close events by ticker order
    trades = []
    open_lookup = {}  # ticker -> list of pending open events
    for _, row in opens.iterrows():
        open_lookup.setdefault(row["ticker"], []).append(row.to_dict())

    for _, close in closes.iterrows():
        t = close["ticker"]
        if t in open_lookup and open_lookup[t]:
            o = open_lookup[t].pop(0)  # FIFO
            trade = {
                "ticker": t,
                "side": o.get("side"),
                "entry_date": o.get("date"),
                "exit_date": close.get("date"),
                "hold_days": (close["date"] - o["date"]).days if pd.notna(o["date"]) and pd.notna(close["date"]) else None,
                "entry_price": o.get("entry_price"),
                "exit_price": close.get("exit_price"),
                "pnl_pct": close.get("pnl_pct"),
                "entry_pred": o.get("entry_pred"),
                "entry_rank": o.get("entry_rank"),
                "entry_weight": o.get("entry_weight"),
                "sector": o.get("sector"),
                "exit_reason": close.get("exit_reason"),
                # Entry features
                "entry_vol_63d": o.get("entry_vol_63d"),
                "entry_mom_21d": o.get("entry_mom_21d"),
                "entry_mom_126d": o.get("entry_mom_126d"),
                "entry_vix": o.get("entry_vix"),
            }
            trades.append(trade)

    return pd.DataFrame(trades)
