"""
Factor attribution analysis — decomposes daily P&L into factor contributions.

For each day:
  total_return = market + size + value + momentum + volatility + quality + alpha

Where:
- Factor contributions are computed from the portfolio's factor exposures
  and the cross-sectional factor returns on that day
- Alpha (residual) is the return not explained by any factor — this is
  the actual skill of the model, distinct from factor tilts

This is critical for:
1. Understanding WHERE returns come from (is it alpha or beta?)
2. Debugging underperformance (which factor turned against us?)
3. Risk monitoring (are we unintentionally exposed to a factor?)
4. Reporting to stakeholders

Reference: Grinold & Kahn, "Active Portfolio Management", Chapter 14
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FACTOR_NAMES = ["market", "size", "value", "momentum", "volatility", "quality"]


@dataclass
class DailyAttribution:
    """One day's factor attribution breakdown."""
    date: str
    total_return: float
    factor_contributions: Dict[str, float]
    alpha_return: float
    top_contributors: Dict[str, float]  # top 5 position P&L
    bottom_contributors: Dict[str, float]  # bottom 5 position P&L
    gross_exposure: float
    net_exposure: float


class FactorAttribution:
    """
    Daily P&L decomposition by Barra factor.

    Uses cross-sectional regression to estimate daily factor returns,
    then decomposes portfolio return into factor contributions.

    Portfolio return on day t:
      r_p = sum_i(w_i * r_i)
      r_i = sum_j(B_ij * f_jt) + epsilon_it
      r_p = sum_j(f_jt * sum_i(w_i * B_ij)) + sum_i(w_i * epsilon_it)
          = sum_j(f_jt * portfolio_exposure_j) + alpha
    """

    def __init__(self):
        self.daily_attributions: list[DailyAttribution] = []

    def attribute_day(
        self,
        date,
        weights: pd.Series,
        stock_returns: pd.Series,
        factor_exposures: pd.DataFrame,
    ) -> DailyAttribution:
        """
        Decompose one day's P&L into factor contributions + alpha.

        Args:
            date: the date
            weights: position weights (ticker -> weight)
            stock_returns: per-stock returns on this day (ticker -> return)
            factor_exposures: (tickers x factors) exposure matrix from risk model
        """
        # Align all series to common tickers
        common = weights.index.intersection(stock_returns.index)
        if factor_exposures is not None and not factor_exposures.empty:
            common = common.intersection(factor_exposures.index)

        if len(common) == 0:
            return DailyAttribution(
                date=str(date), total_return=0.0,
                factor_contributions={f: 0.0 for f in FACTOR_NAMES},
                alpha_return=0.0, top_contributors={}, bottom_contributors={},
                gross_exposure=0.0, net_exposure=0.0,
            )

        w = weights.reindex(common, fill_value=0)
        r = stock_returns.reindex(common, fill_value=0)

        # Total portfolio return
        total = float((w * r).sum())

        # Per-position contributions
        position_pnl = (w * r).sort_values()
        top_5 = position_pnl.tail(5).to_dict()
        bottom_5 = position_pnl.head(5).to_dict()

        # Factor decomposition
        factor_contribs = {}
        explained = 0.0

        if factor_exposures is not None and not factor_exposures.empty:
            B = factor_exposures.reindex(common)

            # Estimate factor returns via cross-sectional regression
            # f_t = (B'B)^{-1} B' r_t
            try:
                BtB = B.T @ B
                if np.linalg.det(BtB.values) > 1e-10:
                    factor_returns = pd.Series(
                        np.linalg.solve(BtB.values, (B.T @ r).values),
                        index=B.columns,
                    )
                else:
                    # Singular matrix — use pseudoinverse
                    factor_returns = pd.Series(
                        np.linalg.lstsq(B.values, r.values, rcond=None)[0],
                        index=B.columns,
                    )

                # Portfolio factor exposure = sum(w_i * B_ij) for each factor j
                portfolio_exposures = (w.values[:, None] * B.values).sum(axis=0)

                for j, factor in enumerate(B.columns):
                    contrib = float(factor_returns.iloc[j] * portfolio_exposures[j])
                    factor_contribs[factor] = round(contrib, 8)
                    explained += contrib

            except Exception as e:
                logger.debug(f"Factor regression failed: {e}")
                for f in FACTOR_NAMES:
                    factor_contribs[f] = 0.0

        # Fill missing factors
        for f in FACTOR_NAMES:
            if f not in factor_contribs:
                factor_contribs[f] = 0.0

        # Alpha = total - sum(factor contributions)
        alpha = total - explained

        attr = DailyAttribution(
            date=str(date),
            total_return=round(total, 8),
            factor_contributions=factor_contribs,
            alpha_return=round(alpha, 8),
            top_contributors={str(k): round(v, 6) for k, v in top_5.items()},
            bottom_contributors={str(k): round(v, 6) for k, v in bottom_5.items()},
            gross_exposure=round(float(w.abs().sum()), 4),
            net_exposure=round(float(w.sum()), 4),
        )

        self.daily_attributions.append(attr)
        return attr

    def to_dataframe(self) -> pd.DataFrame:
        """Convert attribution history to DataFrame."""
        if not self.daily_attributions:
            return pd.DataFrame()

        rows = []
        for a in self.daily_attributions:
            row = {"date": a.date, "total": a.total_return, "alpha": a.alpha_return}
            row.update(a.factor_contributions)
            row["gross_exposure"] = a.gross_exposure
            row["net_exposure"] = a.net_exposure
            rows.append(row)

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def summary(self) -> dict:
        """
        Aggregate attribution statistics.

        Key questions answered:
        - What fraction of returns came from alpha vs factors?
        - Which factor contributed most (positive and negative)?
        - Is the alpha consistent or sporadic?
        """
        df = self.to_dataframe()
        if df.empty:
            return {}

        n_days = len(df)
        factor_cols = [c for c in FACTOR_NAMES if c in df.columns]

        # Cumulative contributions
        cumulative = {}
        for col in factor_cols + ["alpha", "total"]:
            if col in df.columns:
                cumulative[col] = round(float(df[col].sum()), 6)

        # Annualized alpha
        alpha_annual = float(df["alpha"].mean() * 252) if "alpha" in df.columns else 0
        alpha_sharpe = (
            float(df["alpha"].mean() / df["alpha"].std() * np.sqrt(252))
            if "alpha" in df.columns and df["alpha"].std() > 0
            else 0
        )

        # Alpha hit rate (% of days with positive alpha)
        alpha_hit_rate = float((df["alpha"] > 0).mean()) if "alpha" in df.columns else 0

        # Largest factor contributor (positive)
        factor_totals = {f: cumulative.get(f, 0) for f in factor_cols}
        largest_positive = max(factor_totals, key=factor_totals.get) if factor_totals else "none"
        largest_negative = min(factor_totals, key=factor_totals.get) if factor_totals else "none"

        return {
            "n_days": n_days,
            "cumulative": cumulative,
            "alpha_annual": round(alpha_annual, 4),
            "alpha_sharpe": round(alpha_sharpe, 2),
            "alpha_hit_rate": round(alpha_hit_rate, 4),
            "largest_positive_factor": largest_positive,
            "largest_negative_factor": largest_negative,
            "factor_totals": factor_totals,
        }

    def format_report(self) -> str:
        """Human-readable attribution report."""
        s = self.summary()
        if not s:
            return "No attribution data available."

        cum = s.get("cumulative", {})
        lines = [
            "Factor Attribution Report",
            f"  Period: {s['n_days']} trading days",
            "",
            "  Cumulative Contributions:",
            f"    Total return:     {cum.get('total', 0):+.4%}",
            f"    Alpha (residual): {cum.get('alpha', 0):+.4%}  "
            f"(annualized: {s['alpha_annual']:+.2%}, Sharpe: {s['alpha_sharpe']:.2f})",
            f"    Alpha hit rate:   {s['alpha_hit_rate']:.0%}",
            "",
            "  Factor Breakdown:",
        ]

        for f in FACTOR_NAMES:
            val = cum.get(f, 0)
            bar = "+" * int(abs(val) * 1000) if val > 0 else "-" * int(abs(val) * 1000)
            lines.append(f"    {f:<12} {val:+.4%}  {bar}")

        lines.extend([
            "",
            f"  Largest positive: {s['largest_positive_factor']}",
            f"  Largest negative: {s['largest_negative_factor']}",
        ])

        return "\n".join(lines)
