#!/usr/bin/env python3
"""
Look-back analyzer for backtest runs.

Loads diagnostic logs (from diagnostics.py) and supports:
1. Worst-N day forensics (which days lost the most + commonalities)
2. Worst-N trade forensics (which trades hurt the most)
3. Counterfactual filter testing (what if we'd added filter X?)
4. Hypothesis testing harness (state hypothesis → run test → measure impact)
5. Regime-conditional performance breakdown
6. Window-level diagnostics (which walk-forward windows failed)

Usage:
    # CLI
    python look_back_analyzer.py results/sleeve_10d/diagnostics --top 20

    # Python API
    from look_back_analyzer import LookBackAnalyzer
    a = LookBackAnalyzer("results/sleeve_10d/diagnostics", suffix="100L_30S")
    a.worst_days(20)
    a.test_hypothesis("VIX>25 days lose more", lambda day: day["vix"] > 25)
    a.counterfactual_skip(date_filter=lambda day: day["pred_std"] < 0.005)

The analyzer is intentionally biased toward "logical overfitting" — it lets you
test hypotheses you have a structural reason to believe, then measure their
real impact. The trade-off vs traditional CV is acknowledged: any change found
this way uses the test set as validation. Mitigate by:
- Only testing hypotheses with prior structural reasons
- Limiting the number of hypotheses tested
- Tracking the haircut for multiple testing (Bonferroni or PSR)
"""
import os
import argparse
import logging
from typing import Optional, Callable, List, Dict
import pandas as pd
import numpy as np

from diagnostics import (
    load_day_diagnostics, load_position_events, load_window_diagnostics,
    reconstruct_trades,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("look_back")


class LookBackAnalyzer:
    """Loads diagnostic outputs and runs forensics + counterfactuals."""

    def __init__(self, diag_dir: str, model: str = "lightgbm", suffix: str = ""):
        self.diag_dir = diag_dir
        self.model = model
        self.suffix = suffix

        suffix_part = f"_{suffix}" if suffix else ""
        self.day_path = os.path.join(diag_dir, f"day_diagnostics_{model}{suffix_part}.jsonl")
        self.pos_path = os.path.join(diag_dir, f"position_events_{model}{suffix_part}.jsonl")
        self.win_path = os.path.join(diag_dir, f"window_diagnostics_{model}{suffix_part}.jsonl")

        self.days = load_day_diagnostics(self.day_path) if os.path.exists(self.day_path) else pd.DataFrame()
        self.events = load_position_events(self.pos_path) if os.path.exists(self.pos_path) else pd.DataFrame()
        self.windows = load_window_diagnostics(self.win_path) if os.path.exists(self.win_path) else pd.DataFrame()
        self.trades = reconstruct_trades(self.events) if not self.events.empty else pd.DataFrame()

        logger.info(f"Loaded: {len(self.days)} days, {len(self.events)} position events, "
                    f"{len(self.trades)} reconstructed trades, {len(self.windows)} windows")

    # ============================================================
    # SECTION 1: FORENSICS — what lost the most?
    # ============================================================

    def worst_days(self, n: int = 20) -> pd.DataFrame:
        """Show the N worst days by gross_return + their decision context."""
        if self.days.empty or "gross_return" not in self.days.columns:
            print("No day diagnostics available")
            return pd.DataFrame()

        worst = self.days.nsmallest(n, "gross_return")
        cols = [
            "gross_return", "long_pnl", "short_pnl",
            "vix", "pred_std", "n_long", "n_short",
            "gross_exposure", "current_dd", "dd_circuit_active",
            "vix_short_gate_active", "top_long_sector", "top_short_sector",
        ]
        cols = [c for c in cols if c in worst.columns]
        out = worst[cols].copy()
        # Format
        if "gross_return" in out.columns:
            out["gross_return_bps"] = (out["gross_return"] * 1e4).round(1)
            out = out.drop(columns=["gross_return"])
        if "long_pnl" in out.columns:
            out["long_pnl_bps"] = (out["long_pnl"] * 1e4).round(1)
            out = out.drop(columns=["long_pnl"])
        if "short_pnl" in out.columns:
            out["short_pnl_bps"] = (out["short_pnl"] * 1e4).round(1)
            out = out.drop(columns=["short_pnl"])

        print("\n" + "=" * 100)
        print(f"WORST {n} DAYS BY GROSS RETURN")
        print("=" * 100)
        print(out.to_string())
        return worst

    def best_days(self, n: int = 20) -> pd.DataFrame:
        """Show the N best days for comparison."""
        if self.days.empty or "gross_return" not in self.days.columns:
            return pd.DataFrame()
        return self.days.nlargest(n, "gross_return")

    def worst_trades(self, n: int = 20) -> pd.DataFrame:
        """Show the N most painful trades and their entry features."""
        if self.trades.empty or "pnl_pct" not in self.trades.columns:
            print("No trade data available")
            return pd.DataFrame()

        worst = self.trades.nsmallest(n, "pnl_pct")
        cols = [
            "ticker", "side", "entry_date", "exit_date", "hold_days",
            "pnl_pct", "entry_pred", "entry_rank", "sector",
            "entry_vol_63d", "entry_mom_126d", "entry_vix",
        ]
        cols = [c for c in cols if c in worst.columns]
        out = worst[cols].copy()
        if "pnl_pct" in out.columns:
            out["pnl_bps"] = (out["pnl_pct"] * 1e4).round(0)

        print("\n" + "=" * 100)
        print(f"WORST {n} TRADES BY PNL %")
        print("=" * 100)
        print(out.to_string())
        return worst

    # ============================================================
    # SECTION 2: COMMONALITY ANALYSIS — what's similar about a group?
    # ============================================================

    def commonality_analysis(
        self,
        bad_subset: pd.DataFrame,
        good_subset: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        For each feature, show mean in bad subset vs mean in good subset.
        Highlights features that differ meaningfully (potential filter signals).
        """
        if bad_subset.empty or good_subset.empty:
            return pd.DataFrame()

        if feature_cols is None:
            # Auto-detect numeric columns
            feature_cols = [c for c in bad_subset.columns
                            if pd.api.types.is_numeric_dtype(bad_subset[c])
                            and c in good_subset.columns]

        rows = []
        for c in feature_cols:
            bad_mean = bad_subset[c].mean()
            good_mean = good_subset[c].mean()
            if pd.notna(bad_mean) and pd.notna(good_mean):
                # Z-score of difference (rough)
                std_combined = (bad_subset[c].std() + good_subset[c].std()) / 2
                z = (bad_mean - good_mean) / (std_combined + 1e-12)
                rows.append({
                    "feature": c,
                    "bad_mean": bad_mean,
                    "good_mean": good_mean,
                    "diff": bad_mean - good_mean,
                    "z_diff": z,
                })

        df = pd.DataFrame(rows).sort_values("z_diff", key=lambda s: s.abs(), ascending=False)
        print("\n" + "=" * 100)
        print(f"COMMONALITY ANALYSIS: bad ({len(bad_subset)} rows) vs good ({len(good_subset)} rows)")
        print("=" * 100)
        print("Top 20 most-discriminating features (sorted by |z|):")
        print(df.head(20).to_string(index=False))
        return df

    # ============================================================
    # SECTION 3: COUNTERFACTUAL FILTER TESTING
    # ============================================================

    def counterfactual_skip(
        self,
        date_filter: Callable[[pd.Series], bool],
        label: str = "filter",
    ) -> Dict[str, float]:
        """
        Counterfactual: what if we'd skipped days where date_filter returns True?
        Sets gross_return to 0 on those days, recomputes Sharpe + total return.
        """
        if self.days.empty or "gross_return" not in self.days.columns:
            return {}

        skip_mask = self.days.apply(date_filter, axis=1)
        n_skipped = int(skip_mask.sum())

        baseline = self._sharpe_metrics(self.days["gross_return"])
        modified_returns = self.days["gross_return"].copy()
        modified_returns[skip_mask] = 0
        modified = self._sharpe_metrics(modified_returns)

        result = {
            "label": label,
            "days_skipped": n_skipped,
            "days_total": len(self.days),
            "baseline_total_return": baseline["total"],
            "modified_total_return": modified["total"],
            "delta_total": modified["total"] - baseline["total"],
            "baseline_sharpe": baseline["sharpe"],
            "modified_sharpe": modified["sharpe"],
            "delta_sharpe": modified["sharpe"] - baseline["sharpe"],
            "baseline_mdd": baseline["mdd"],
            "modified_mdd": modified["mdd"],
            "delta_mdd": modified["mdd"] - baseline["mdd"],
        }
        print(f"\n--- COUNTERFACTUAL: {label} ---")
        print(f"  Skipped {n_skipped}/{len(self.days)} days ({n_skipped/len(self.days)*100:.1f}%)")
        print(f"  Total return: {baseline['total']:+.2%} → {modified['total']:+.2%}  Δ={result['delta_total']:+.2%}")
        print(f"  Sharpe:       {baseline['sharpe']:+.3f} → {modified['sharpe']:+.3f}  Δ={result['delta_sharpe']:+.3f}")
        print(f"  Max DD:       {baseline['mdd']:+.2%} → {modified['mdd']:+.2%}  Δ={result['delta_mdd']:+.2%}")
        return result

    def counterfactual_scale(
        self,
        date_filter: Callable[[pd.Series], bool],
        scale: float,
        label: str = "scale",
    ) -> Dict[str, float]:
        """
        Counterfactual: scale gross_return by `scale` on days where filter is True.
        e.g., scale=0.25 simulates a circuit breaker that cuts exposure to 25%.
        """
        if self.days.empty or "gross_return" not in self.days.columns:
            return {}

        scale_mask = self.days.apply(date_filter, axis=1)
        n_scaled = int(scale_mask.sum())

        baseline = self._sharpe_metrics(self.days["gross_return"])
        modified_returns = self.days["gross_return"].copy()
        modified_returns[scale_mask] = modified_returns[scale_mask] * scale
        modified = self._sharpe_metrics(modified_returns)

        result = {
            "label": label,
            "days_scaled": n_scaled,
            "days_total": len(self.days),
            "scale_factor": scale,
            "baseline_total_return": baseline["total"],
            "modified_total_return": modified["total"],
            "delta_total": modified["total"] - baseline["total"],
            "baseline_sharpe": baseline["sharpe"],
            "modified_sharpe": modified["sharpe"],
            "delta_sharpe": modified["sharpe"] - baseline["sharpe"],
            "baseline_mdd": baseline["mdd"],
            "modified_mdd": modified["mdd"],
            "delta_mdd": modified["mdd"] - baseline["mdd"],
        }
        print(f"\n--- COUNTERFACTUAL SCALE: {label} (factor={scale}) ---")
        print(f"  Scaled {n_scaled}/{len(self.days)} days ({n_scaled/len(self.days)*100:.1f}%)")
        print(f"  Sharpe: {baseline['sharpe']:+.3f} → {modified['sharpe']:+.3f}  Δ={result['delta_sharpe']:+.3f}")
        print(f"  MDD:    {baseline['mdd']:+.2%} → {modified['mdd']:+.2%}  Δ={result['delta_mdd']:+.2%}")
        return result

    # ============================================================
    # SECTION 4: HYPOTHESIS TESTING HARNESS
    # ============================================================

    def test_hypothesis(
        self,
        name: str,
        condition: Callable[[pd.Series], bool],
        action: str = "skip",
        scale: float = 0.0,
    ) -> Dict[str, float]:
        """
        Test a hypothesis-driven filter.

        Args:
            name: descriptive label
            condition: function(day_record) → bool. True = matches hypothesis.
            action: "skip" (zero out matched days) or "scale" (multiply by scale)
            scale: multiplier when action="scale" (e.g. 0.25 for circuit breaker)

        Returns counterfactual delta and prints a structured summary.
        """
        print(f"\n{'='*80}")
        print(f"HYPOTHESIS: {name}")
        print(f"{'='*80}")

        if self.days.empty:
            print("No day data")
            return {}

        # Show raw effect first: what's the mean return on matched vs unmatched days?
        matched_mask = self.days.apply(condition, axis=1)
        matched_ret = self.days.loc[matched_mask, "gross_return"]
        unmatched_ret = self.days.loc[~matched_mask, "gross_return"]
        print(f"  Matched days: {matched_mask.sum()}/{len(self.days)} "
              f"({matched_mask.sum()/len(self.days)*100:.1f}%)")
        print(f"  Mean return on matched days:   {matched_ret.mean()*1e4:+7.2f} bps")
        print(f"  Mean return on unmatched days: {unmatched_ret.mean()*1e4:+7.2f} bps")
        print(f"  Difference (matched - unmatched): {(matched_ret.mean() - unmatched_ret.mean())*1e4:+7.2f} bps")

        # Apply counterfactual
        if action == "skip":
            return self.counterfactual_skip(condition, label=name)
        elif action == "scale":
            return self.counterfactual_scale(condition, scale=scale, label=f"{name} (×{scale})")
        else:
            raise ValueError(f"Unknown action {action}")

    # ============================================================
    # SECTION 5: REGIME PERFORMANCE BREAKDOWN
    # ============================================================

    def regime_breakdown(self, regime_col: str = "vix", n_buckets: int = 5):
        """Show return statistics bucketed by a regime column."""
        if self.days.empty or regime_col not in self.days.columns:
            print(f"Column {regime_col} not in day data")
            return None

        df = self.days.dropna(subset=[regime_col, "gross_return"]).copy()
        df["bucket"] = pd.qcut(df[regime_col], n_buckets, duplicates="drop")
        stats = df.groupby("bucket", observed=True)["gross_return"].agg(["count", "mean", "std", "sum"])
        stats["mean_bps"] = stats["mean"] * 1e4
        stats["sum_pct"] = stats["sum"] * 100
        stats["sharpe"] = stats["mean"] / stats["std"] * np.sqrt(252)
        print(f"\n--- Regime breakdown by {regime_col} ({n_buckets} buckets) ---")
        print(stats[["count", "mean_bps", "sum_pct", "sharpe"]].to_string())
        return stats

    # ============================================================
    # SECTION 6: WINDOW-LEVEL DIAGNOSTICS
    # ============================================================

    def window_failure_analysis(self):
        """Identify the worst walk-forward windows by validation IC."""
        if self.windows.empty:
            print("No window data")
            return None

        ic_col = "val_rank_ic" if "val_rank_ic" in self.windows.columns else "val_ic"
        if ic_col not in self.windows.columns:
            print("No IC column found in window data")
            return None

        sorted_w = self.windows.sort_values(ic_col)
        print(f"\n--- WORST 10 WALK-FORWARD WINDOWS by {ic_col} ---")
        cols = ["window", "predict_start", "predict_end", ic_col]
        if "n_train_samples" in sorted_w.columns:
            cols.append("n_train_samples")
        if "mean_sample_weight" in sorted_w.columns:
            cols.append("mean_sample_weight")
        cols = [c for c in cols if c in sorted_w.columns]
        print(sorted_w.head(10)[cols].to_string(index=False))

        # Feature stability: how often does each feature appear across windows?
        if "window_features" in self.windows.columns:
            from collections import Counter
            feat_count = Counter()
            for feats in self.windows["window_features"].dropna():
                if isinstance(feats, list):
                    feat_count.update(feats)
            top = pd.Series(feat_count).sort_values(ascending=False).head(20)
            print(f"\n--- TOP 20 MOST STABLE FEATURES (across {len(self.windows)} windows) ---")
            print(top.to_string())

        return sorted_w

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _sharpe_metrics(returns: pd.Series) -> Dict[str, float]:
        r = returns.dropna()
        if len(r) < 2:
            return {"total": 0, "sharpe": 0, "mdd": 0}
        total = (1 + r).prod() - 1
        sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252)
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        return {"total": float(total), "sharpe": float(sharpe), "mdd": float(mdd)}

    # ============================================================
    # CONVENIENT BATCH RUNS
    # ============================================================

    def run_full_audit(self, top_n: int = 20):
        """Run all default forensics + a battery of standard hypothesis tests."""
        print("\n" + "#" * 100)
        print(f"# LOOK-BACK AUDIT: {self.diag_dir}")
        print("#" * 100)

        # 1. Worst days + trades
        self.worst_days(top_n)
        self.worst_trades(top_n)

        # 2. Commonality between worst quartile and best quartile of days
        if not self.days.empty and "gross_return" in self.days.columns:
            ret = self.days["gross_return"]
            q25 = ret.quantile(0.25)
            q75 = ret.quantile(0.75)
            bad = self.days[ret <= q25]
            good = self.days[ret >= q75]
            self.commonality_analysis(bad, good)

        # 3. Regime breakdowns
        for col in ("vix", "pred_std", "current_dd"):
            if col in self.days.columns:
                self.regime_breakdown(col)

        # 4. Standard hypothesis battery
        print("\n" + "#" * 100)
        print("# STANDARD HYPOTHESIS BATTERY")
        print("#" * 100)
        hypotheses = [
            ("VIX > 30 days lose more", lambda d: d.get("vix", 0) > 30, "skip", None),
            ("VIX > 25 days lose more", lambda d: d.get("vix", 0) > 25, "skip", None),
            ("Low dispersion (<0.005) is bad", lambda d: d.get("pred_std", 1) < 0.005, "skip", None),
            ("Low dispersion (<0.001) is bad", lambda d: d.get("pred_std", 1) < 0.001, "skip", None),
            ("DD > -3% scale to 0.25", lambda d: d.get("current_dd", 0) < -0.03, "scale", 0.25),
            ("DD > -5% scale to 0.5", lambda d: d.get("current_dd", 0) < -0.05, "scale", 0.5),
            ("Net long > 0.6 reduce", lambda d: d.get("net_exposure", 0) > 0.6, "scale", 0.5),
            ("Top long sector >25% concentration cut",
             lambda d: (d.get("top_long_sector_pct", 0) or 0) > 0.25, "scale", 0.5),
        ]
        for name, cond, action, scale in hypotheses:
            try:
                self.test_hypothesis(name, cond, action=action, scale=scale or 0.0)
            except Exception as e:
                print(f"  {name}: error → {e}")

        # 5. Window failures
        self.window_failure_analysis()


def main():
    parser = argparse.ArgumentParser(description="Look-back analyzer for backtest diagnostics")
    parser.add_argument("diag_dir", help="Path to diagnostics directory (e.g. results/sleeve_10d/diagnostics)")
    parser.add_argument("--model", default="lightgbm", help="Model name (default: lightgbm)")
    parser.add_argument("--suffix", default="", help="Run suffix (e.g. 100L_30S)")
    parser.add_argument("--top", type=int, default=20, help="N for worst-N analyses")
    parser.add_argument("--audit", action="store_true", help="Run full audit (default forensics + standard hypotheses)")
    parser.add_argument("--worst-days", action="store_true")
    parser.add_argument("--worst-trades", action="store_true")
    parser.add_argument("--regime", help="Regime column to bucket by (e.g. vix)")
    args = parser.parse_args()

    a = LookBackAnalyzer(args.diag_dir, model=args.model, suffix=args.suffix)

    if args.audit or not any([args.worst_days, args.worst_trades, args.regime]):
        a.run_full_audit(top_n=args.top)
    else:
        if args.worst_days:
            a.worst_days(args.top)
        if args.worst_trades:
            a.worst_trades(args.top)
        if args.regime:
            a.regime_breakdown(args.regime)


if __name__ == "__main__":
    main()
