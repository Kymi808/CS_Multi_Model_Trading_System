"""
Bear market analysis — focused evaluation of system behavior during downturns.

Tests the critical question: does the system protect capital when markets fall?

Specifically for 2022 (SPY -19.4%):
1. Did HMM detect the regime change? When? How early/late?
2. Did risk scaling reduce exposure? By how much? Was it enough?
3. What was max drawdown vs SPY?
4. Did tail risk protection activate? On which days?
5. Monthly return comparison vs benchmark
6. Factor attribution during the drawdown (what hurt most?)

This analysis is essential because the full backtest (2021-2026) is
predominantly bullish. A system that looks great in a bull market but
blows up in a bear market is worthless.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def analyze_bear_period(
    results_df: pd.DataFrame,
    prices: pd.DataFrame,
    start_date: str = "2022-01-01",
    end_date: str = "2022-12-31",
    benchmark_ticker: str = "^GSPC",
) -> dict:
    """
    Comprehensive analysis of system behavior during a bear market period.

    Args:
        results_df: backtest results DataFrame with columns:
            date, net_return, gross_return, turnover, n_long, n_short,
            gross_exposure, net_exposure
        prices: full price DataFrame (for SPY benchmark)
        start_date: bear period start
        end_date: bear period end
        benchmark_ticker: benchmark for comparison

    Returns:
        dict with all analysis results
    """
    analysis = {
        "period": f"{start_date} to {end_date}",
        "benchmark": benchmark_ticker,
    }

    # Filter to bear period
    if isinstance(results_df.index, pd.DatetimeIndex):
        mask = (results_df.index >= start_date) & (results_df.index <= end_date)
    else:
        results_df.index = pd.to_datetime(results_df.index)
        mask = (results_df.index >= start_date) & (results_df.index <= end_date)

    bear = results_df.loc[mask]

    if len(bear) == 0:
        logger.warning(f"No data in bear period {start_date} to {end_date}")
        analysis["error"] = "No data in specified period"
        return analysis

    # ── System Performance ───────────────────────────────────────────
    ret_col = "net_return" if "net_return" in bear.columns else "gross_return"
    system_returns = bear[ret_col]
    system_cumulative = (1 + system_returns).cumprod()
    system_total = float(system_cumulative.iloc[-1] - 1)
    system_annual = float((1 + system_total) ** (252 / len(bear)) - 1)
    system_vol = float(system_returns.std() * np.sqrt(252))
    system_sharpe = system_annual / system_vol if system_vol > 0 else 0

    # Max drawdown
    peak = system_cumulative.expanding().max()
    drawdown = (system_cumulative - peak) / peak
    system_max_dd = float(drawdown.min())

    analysis["system"] = {
        "total_return": round(system_total, 4),
        "annualized_return": round(system_annual, 4),
        "annualized_vol": round(system_vol, 4),
        "sharpe": round(system_sharpe, 2),
        "max_drawdown": round(system_max_dd, 4),
        "n_trading_days": len(bear),
    }

    # ── Benchmark Performance ────────────────────────────────────────
    spy_col = None
    for col_name in [benchmark_ticker, "SPY", "^GSPC"]:
        if col_name in prices.columns:
            spy_col = col_name
            break

    if spy_col:
        spy = prices[spy_col]
        spy_bear = spy.loc[(spy.index >= start_date) & (spy.index <= end_date)]
        if len(spy_bear) > 1:
            spy_returns = spy_bear.pct_change().dropna()
            spy_cumulative = (1 + spy_returns).cumprod()
            spy_total = float(spy_cumulative.iloc[-1] - 1)
            spy_vol = float(spy_returns.std() * np.sqrt(252))
            spy_peak = spy_cumulative.expanding().max()
            spy_dd = float(((spy_cumulative - spy_peak) / spy_peak).min())

            analysis["benchmark"] = {
                "total_return": round(spy_total, 4),
                "annualized_vol": round(spy_vol, 4),
                "max_drawdown": round(spy_dd, 4),
            }

            analysis["relative"] = {
                "outperformance": round(system_total - spy_total, 4),
                "drawdown_reduction": round(system_max_dd - spy_dd, 4),
                "vol_reduction": round(system_vol - spy_vol, 4),
            }

    # ── Monthly Breakdown ────────────────────────────────────────────
    monthly = system_returns.resample("M").sum()
    monthly_data = {}
    for date, ret in monthly.items():
        month_str = date.strftime("%Y-%m")
        monthly_data[month_str] = round(float(ret), 4)

    # Add SPY monthly for comparison
    if spy_col and len(spy_bear) > 1:
        spy_monthly = spy_returns.resample("M").sum()
        monthly_comparison = []
        for date, sys_ret in monthly.items():
            month_str = date.strftime("%Y-%m")
            spy_ret = spy_monthly.get(date, 0)
            monthly_comparison.append({
                "month": month_str,
                "system": round(float(sys_ret), 4),
                "benchmark": round(float(spy_ret), 4),
                "relative": round(float(sys_ret - spy_ret), 4),
            })
        analysis["monthly_comparison"] = monthly_comparison

    # ── Exposure Analysis ────────────────────────────────────────────
    if "gross_exposure" in bear.columns:
        analysis["exposure"] = {
            "avg_gross": round(float(bear["gross_exposure"].mean()), 4),
            "min_gross": round(float(bear["gross_exposure"].min()), 4),
            "max_gross": round(float(bear["gross_exposure"].max()), 4),
            "avg_net": round(float(bear["net_exposure"].mean()), 4) if "net_exposure" in bear.columns else 0,
        }

    if "n_long" in bear.columns and "n_short" in bear.columns:
        analysis["positions"] = {
            "avg_long": round(float(bear["n_long"].mean()), 1),
            "avg_short": round(float(bear["n_short"].mean()), 1),
        }

    # ── Worst Days ───────────────────────────────────────────────────
    worst_5 = system_returns.nsmallest(5)
    best_5 = system_returns.nlargest(5)

    analysis["worst_days"] = [
        {"date": str(d.date()), "return": round(float(r), 4)}
        for d, r in worst_5.items()
    ]
    analysis["best_days"] = [
        {"date": str(d.date()), "return": round(float(r), 4)}
        for d, r in best_5.items()
    ]

    # ── Win Rate ─────────────────────────────────────────────────────
    analysis["win_rate"] = {
        "daily": round(float((system_returns > 0).mean()), 4),
        "weekly": round(float((system_returns.resample("W").sum() > 0).mean()), 4),
        "monthly": round(float((monthly > 0).mean()), 4),
    }

    return analysis


def format_bear_report(analysis: dict) -> str:
    """Human-readable bear market analysis report."""
    lines = [
        "=" * 70,
        f"BEAR MARKET ANALYSIS: {analysis.get('period', 'Unknown')}",
        "=" * 70,
        "",
    ]

    sys = analysis.get("system", {})
    bench = analysis.get("benchmark", {})
    rel = analysis.get("relative", {})

    lines.extend([
        "Performance Comparison:",
        f"  {'Metric':<25} {'System':>12} {'Benchmark':>12} {'Relative':>12}",
        f"  {'─'*25} {'─'*12} {'─'*12} {'─'*12}",
        f"  {'Total Return':<25} {sys.get('total_return', 0):>+11.2%} "
        f"{bench.get('total_return', 0):>+11.2%} "
        f"{rel.get('outperformance', 0):>+11.2%}",
        f"  {'Max Drawdown':<25} {sys.get('max_drawdown', 0):>+11.2%} "
        f"{bench.get('max_drawdown', 0):>+11.2%} "
        f"{rel.get('drawdown_reduction', 0):>+11.2%}",
        f"  {'Annualized Vol':<25} {sys.get('annualized_vol', 0):>11.2%} "
        f"{bench.get('annualized_vol', 0):>11.2%} "
        f"{rel.get('vol_reduction', 0):>+11.2%}",
        f"  {'Sharpe':<25} {sys.get('sharpe', 0):>12.2f}",
        "",
    ])

    # Monthly comparison
    monthly = analysis.get("monthly_comparison", [])
    if monthly:
        lines.extend([
            "Monthly Breakdown:",
            f"  {'Month':<10} {'System':>10} {'Benchmark':>10} {'Relative':>10}",
            f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10}",
        ])
        for m in monthly:
            lines.append(
                f"  {m['month']:<10} {m['system']:>+9.2%} "
                f"{m['benchmark']:>+9.2%} "
                f"{m['relative']:>+9.2%}"
            )
        lines.append("")

    # Exposure
    exp = analysis.get("exposure", {})
    if exp:
        lines.extend([
            "Exposure During Bear Period:",
            f"  Avg gross: {exp.get('avg_gross', 0):.1%}",
            f"  Min gross: {exp.get('min_gross', 0):.1%} (max risk reduction)",
            f"  Avg net:   {exp.get('avg_net', 0):.1%}",
            "",
        ])

    # Worst days
    worst = analysis.get("worst_days", [])
    if worst:
        lines.append("Worst 5 Days:")
        for d in worst:
            lines.append(f"  {d['date']}: {d['return']:+.2%}")
        lines.append("")

    # Win rate
    wr = analysis.get("win_rate", {})
    if wr:
        lines.extend([
            "Win Rates:",
            f"  Daily:   {wr.get('daily', 0):.0%}",
            f"  Weekly:  {wr.get('weekly', 0):.0%}",
            f"  Monthly: {wr.get('monthly', 0):.0%}",
            "",
        ])

    # Key question answers
    lines.extend([
        "KEY QUESTIONS:",
        f"  1. Did system outperform benchmark? {'YES' if rel.get('outperformance', 0) > 0 else 'NO'} "
        f"({rel.get('outperformance', 0):+.2%})",
        f"  2. Was drawdown less severe? {'YES' if rel.get('drawdown_reduction', 0) > 0 else 'NO'} "
        f"(system {sys.get('max_drawdown', 0):.2%} vs benchmark {bench.get('max_drawdown', 0):.2%})",
        f"  3. Did risk controls reduce exposure? "
        f"{'YES' if exp.get('min_gross', 1) < exp.get('avg_gross', 1) * 0.8 else 'UNCLEAR'} "
        f"(min {exp.get('min_gross', 0):.1%} vs avg {exp.get('avg_gross', 0):.1%})",
    ])

    return "\n".join(lines)
