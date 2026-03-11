"""
Multi-model comparison framework.

Runs LightGBM, TST, and CrossMamba through the same pipeline
(risk model, transaction costs, portfolio construction) and
produces a comprehensive comparison of all metrics.
"""
import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Dict, Tuple, Optional
from config import Config
from model import walk_forward_train
from risk_model import FactorRiskModel
from portfolio import (
    PortfolioConstructor, compute_portfolio_returns, compute_performance_metrics,
)

logger = logging.getLogger(__name__)


def run_single_model_pipeline(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    fundamentals: dict,
    sector_map: dict,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Run a single model through the full pipeline:
    walk-forward train → risk model → portfolio construction → performance.

    Returns:
        (results_df, summary_dict, metrics_df)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING MODEL: {model_type.upper()}")
    logger.info(f"{'='*60}")

    # Select the right config for this model
    model_cfg_map = {
        "lightgbm": cfg.model,
        "tst": cfg.tst,
        "crossmamba": cfg.crossmamba,
    }
    model_cfg = model_cfg_map.get(model_type, cfg.model)

    # Walk-forward training
    models, oos_predictions, metrics_df = walk_forward_train(
        X, y, cfg.model, cfg.features,
        model_type=model_type,
        model_cfg=model_cfg,
    )

    if oos_predictions.empty:
        logger.error(f"[{model_type}] No predictions generated")
        return pd.DataFrame(), {}, pd.DataFrame()

    # Risk model
    risk = FactorRiskModel(cfg.risk)
    if len(prices) > 252:
        risk.estimate(prices, fundamentals, prices.index[-1], lookback=504)
        risk.update_regime(prices)

    # Portfolio construction
    constructor = PortfolioConstructor(cfg.portfolio)
    weights_history: Dict[pd.Timestamp, pd.Series] = {}
    portfolio_returns = pd.Series(dtype=float)
    stock_vol = np.log(prices / prices.shift(1)).rolling(63).std() * np.sqrt(252)
    prev_weights = pd.Series(dtype=float)

    n_long = cfg.portfolio.max_positions_long
    n_short = cfg.portfolio.max_positions_short

    if isinstance(oos_predictions.index, pd.MultiIndex):
        dates = sorted(oos_predictions.index.get_level_values(0).unique())
    else:
        dates = []

    risk_reestimate_every = 63
    last_risk_estimate = 0

    for di, date in enumerate(dates):
        if isinstance(oos_predictions.index, pd.MultiIndex):
            day_preds = oos_predictions.loc[date]
        else:
            continue
        if not isinstance(day_preds, pd.Series) or len(day_preds) == 0:
            continue

        if di - last_risk_estimate >= risk_reestimate_every:
            if date in prices.index:
                risk.estimate(prices, fundamentals, date, lookback=504)
                risk.update_regime(prices.loc[:date])
                last_risk_estimate = di

        vol_est = stock_vol.loc[date] if date in stock_vol.index else None

        target_weights = constructor.construct_portfolio(
            predictions=day_preds, date=date,
            prev_weights=prev_weights, vol_estimates=vol_est,
        )

        target_weights = risk.apply_risk_scaling(
            target_weights, portfolio_returns, sector_map,
            n_long=n_long, n_short=n_short,
        )

        target_weights = target_weights.clip(
            -cfg.portfolio.max_position_pct,
            cfg.portfolio.max_position_pct,
        )

        if not portfolio_returns.empty:
            risk.update(portfolio_returns.iloc[-1])

        weights_history[date] = target_weights
        prev_weights = target_weights

        if date in prices.index:
            prev_idx = prices.index.get_loc(date) - 1
            if prev_idx >= 0:
                prev_date = prices.index[prev_idx]
                common = target_weights.index.intersection(prices.columns)
                if len(common) > 0:
                    day_ret = (
                        (prices.loc[date, common] / prices.loc[prev_date, common] - 1)
                        * target_weights.reindex(common, fill_value=0)
                    ).sum()
                    portfolio_returns = pd.concat([
                        portfolio_returns, pd.Series([day_ret], index=[date])
                    ])

    # Compute full results
    results_df = compute_portfolio_returns(weights_history, prices, cfg.portfolio)

    if results_df.empty:
        logger.error(f"[{model_type}] No returns computed")
        return pd.DataFrame(), {}, metrics_df

    summary = compute_performance_metrics(results_df["net_return"])
    summary["model_type"] = model_type

    # Add IC metrics from walk-forward
    if "val_rank_ic" in metrics_df.columns:
        summary["avg_rank_ic"] = float(metrics_df["val_rank_ic"].mean())
        summary["ic_std"] = float(metrics_df["val_rank_ic"].std())
        ic_std = metrics_df["val_rank_ic"].std()
        summary["ic_ir"] = float(
            metrics_df["val_rank_ic"].mean() / (ic_std + 1e-8)
        )

    # Save model
    if models:
        model_path = os.path.join(cfg.model_dir, f"latest_{model_type}_model.pkl")
        models[-1].save(model_path)

    return results_df, summary, metrics_df


def build_ensemble_predictions(
    model_predictions: Dict[str, pd.Series],
    weights: Dict[str, float],
) -> pd.Series:
    """
    Combine predictions from multiple models into an ensemble.

    Normalizes each model's predictions to ranks first (so they're
    on the same scale), then weighted-averages.
    """
    ranked_preds = {}
    for name, preds in model_predictions.items():
        if preds.empty:
            continue
        # Rank within each date
        if isinstance(preds.index, pd.MultiIndex):
            ranked = preds.groupby(level=0).rank(pct=True)
        else:
            ranked = preds.rank(pct=True)
        ranked_preds[name] = ranked

    if not ranked_preds:
        return pd.Series(dtype=float)

    # Find common index
    common_idx = None
    for preds in ranked_preds.values():
        if common_idx is None:
            common_idx = preds.index
        else:
            common_idx = common_idx.intersection(preds.index)

    if common_idx is None or len(common_idx) == 0:
        return pd.Series(dtype=float)

    # Weighted average
    total_weight = sum(weights.get(name, 1.0) for name in ranked_preds)
    ensemble = pd.Series(0.0, index=common_idx)
    for name, preds in ranked_preds.items():
        w = weights.get(name, 1.0) / total_weight
        ensemble += w * preds.reindex(common_idx, fill_value=0.5)

    return ensemble


def compute_spy_benchmark(start_date: str, end_date: str) -> Optional[pd.Series]:
    """Fetch SPY returns for benchmark comparison."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
        if spy.empty:
            return None
        if isinstance(spy.columns, pd.MultiIndex):
            spy_close = spy[("Close", "SPY")]
        else:
            spy_close = spy["Close"]
        return spy_close.pct_change().dropna()
    except Exception:
        return None


def run_comparison(
    cfg: Config,
    X: pd.DataFrame,
    y: pd.Series,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    fundamentals: dict,
    sector_map: dict,
    selected_features: list,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict], dict]:
    """
    Run all models through the same pipeline and compare.

    Returns:
        (all_results, all_summaries, comparison_table)
    """
    models_to_run = cfg.comparison.models_to_run
    all_results = {}
    all_summaries = {}
    all_predictions = {}
    all_metrics = {}

    for model_type in models_to_run:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# MODEL: {model_type.upper()}")
        logger.info(f"{'#'*60}")

        results_df, summary, metrics_df = run_single_model_pipeline(
            model_type=model_type,
            X=X, y=y, cfg=cfg,
            prices=prices, volumes=volumes,
            fundamentals=fundamentals,
            sector_map=sector_map,
        )

        all_results[model_type] = results_df
        all_summaries[model_type] = summary
        all_metrics[model_type] = metrics_df

        # Store predictions for ensemble
        model_cfg_map = {
            "lightgbm": cfg.model,
            "tst": cfg.tst,
            "crossmamba": cfg.crossmamba,
        }
        model_cfg = model_cfg_map.get(model_type, cfg.model)
        _, preds, _ = walk_forward_train(
            X, y, cfg.model, cfg.features,
            model_type=model_type,
            model_cfg=model_cfg,
        )
        all_predictions[model_type] = preds

    # Ensemble
    if cfg.comparison.run_ensemble and len(all_predictions) > 1:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# ENSEMBLE (weighted average)")
        logger.info(f"{'#'*60}")

        ensemble_preds = build_ensemble_predictions(
            all_predictions, cfg.comparison.ensemble_weights,
        )

        if not ensemble_preds.empty:
            # Run ensemble through portfolio pipeline
            risk = FactorRiskModel(cfg.risk)
            if len(prices) > 252:
                risk.estimate(prices, fundamentals, prices.index[-1], lookback=504)
                risk.update_regime(prices)

            constructor = PortfolioConstructor(cfg.portfolio)
            weights_history = {}
            portfolio_returns = pd.Series(dtype=float)
            stock_vol = np.log(prices / prices.shift(1)).rolling(63).std() * np.sqrt(252)
            prev_weights = pd.Series(dtype=float)
            n_long = cfg.portfolio.max_positions_long
            n_short = cfg.portfolio.max_positions_short

            if isinstance(ensemble_preds.index, pd.MultiIndex):
                dates = sorted(ensemble_preds.index.get_level_values(0).unique())
            else:
                dates = []

            risk_reestimate_every = 63
            last_risk_estimate = 0

            for di, date in enumerate(dates):
                if isinstance(ensemble_preds.index, pd.MultiIndex):
                    day_preds = ensemble_preds.loc[date]
                else:
                    continue
                if not isinstance(day_preds, pd.Series) or len(day_preds) == 0:
                    continue

                if di - last_risk_estimate >= risk_reestimate_every:
                    if date in prices.index:
                        risk.estimate(prices, fundamentals, date, lookback=504)
                        risk.update_regime(prices.loc[:date])
                        last_risk_estimate = di

                vol_est = stock_vol.loc[date] if date in stock_vol.index else None

                target_weights = constructor.construct_portfolio(
                    predictions=day_preds, date=date,
                    prev_weights=prev_weights, vol_estimates=vol_est,
                )
                target_weights = risk.apply_risk_scaling(
                    target_weights, portfolio_returns, sector_map,
                    n_long=n_long, n_short=n_short,
                )
                target_weights = target_weights.clip(
                    -cfg.portfolio.max_position_pct,
                    cfg.portfolio.max_position_pct,
                )

                if not portfolio_returns.empty:
                    risk.update(portfolio_returns.iloc[-1])

                weights_history[date] = target_weights
                prev_weights = target_weights

                if date in prices.index:
                    prev_idx = prices.index.get_loc(date) - 1
                    if prev_idx >= 0:
                        prev_date = prices.index[prev_idx]
                        common = target_weights.index.intersection(prices.columns)
                        if len(common) > 0:
                            day_ret = (
                                (prices.loc[date, common] / prices.loc[prev_date, common] - 1)
                                * target_weights.reindex(common, fill_value=0)
                            ).sum()
                            portfolio_returns = pd.concat([
                                portfolio_returns, pd.Series([day_ret], index=[date])
                            ])

            ens_results = compute_portfolio_returns(weights_history, prices, cfg.portfolio)
            if not ens_results.empty:
                ens_summary = compute_performance_metrics(ens_results["net_return"])
                ens_summary["model_type"] = "ensemble"
                all_results["ensemble"] = ens_results
                all_summaries["ensemble"] = ens_summary

    # SPY benchmark
    spy_returns = None
    for name, res in all_results.items():
        if not res.empty:
            spy_returns = compute_spy_benchmark(
                str(res.index[0].date()), str(res.index[-1].date()),
            )
            break

    if spy_returns is not None:
        spy_metrics = compute_performance_metrics(spy_returns)
        all_summaries["spy_benchmark"] = spy_metrics

    # Build comparison table
    comparison = build_comparison_table(all_summaries)

    return all_results, all_summaries, comparison


def build_comparison_table(summaries: Dict[str, dict]) -> dict:
    """Build a structured comparison of all models."""
    metrics_of_interest = [
        ("annual_return", "Annual Return", ".2%"),
        ("annual_volatility", "Volatility", ".2%"),
        ("sharpe_ratio", "Sharpe Ratio", ".3f"),
        ("sortino_ratio", "Sortino Ratio", ".3f"),
        ("max_drawdown", "Max Drawdown", ".2%"),
        ("calmar_ratio", "Calmar Ratio", ".3f"),
        ("win_rate", "Win Rate", ".2%"),
        ("profit_factor", "Profit Factor", ".3f"),
        ("avg_rank_ic", "Avg Rank IC", ".4f"),
        ("ic_ir", "IC Info Ratio", ".3f"),
        ("total_return", "Total Return", ".2%"),
    ]

    comparison = {}
    for key, label, fmt in metrics_of_interest:
        row = {}
        for model_name, summary in summaries.items():
            val = summary.get(key)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                row[model_name] = val
        if row:
            comparison[label] = row

    return comparison


def print_comparison(comparison: dict, summaries: Dict[str, dict]):
    """Pretty-print the comparison table to stdout."""
    model_names = []
    for row in comparison.values():
        for name in row:
            if name not in model_names:
                model_names.append(name)

    # Header
    col_width = 14
    header = f"{'Metric':>20s}"
    for name in model_names:
        header += f"  {name:>{col_width}s}"
    sep = "─" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"{'MULTI-MODEL COMPARISON':^{len(header)}s}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    format_map = {
        "Annual Return": ".2%", "Volatility": ".2%", "Max Drawdown": ".2%",
        "Win Rate": ".2%", "Total Return": ".2%",
        "Sharpe Ratio": ".3f", "Sortino Ratio": ".3f", "Calmar Ratio": ".3f",
        "Profit Factor": ".3f", "IC Info Ratio": ".3f",
        "Avg Rank IC": ".4f",
    }

    for metric, row in comparison.items():
        fmt = format_map.get(metric, ".4f")
        line = f"{metric:>20s}"
        # Find best value for highlighting
        vals = {k: v for k, v in row.items() if isinstance(v, (int, float))}
        best_model = None
        if vals:
            # Higher is better for most metrics, except vol, drawdown
            if metric in ("Volatility", "Max Drawdown"):
                # For drawdown (negative), closest to 0 is best
                if metric == "Max Drawdown":
                    best_model = max(vals, key=lambda k: vals[k])
                else:
                    best_model = min(vals, key=lambda k: vals[k])
            else:
                best_model = max(vals, key=lambda k: vals[k])

        for name in model_names:
            val = row.get(name)
            if val is not None:
                formatted = f"{val:{fmt}}"
                if name == best_model:
                    formatted = f"*{formatted}"
                line += f"  {formatted:>{col_width}s}"
            else:
                line += f"  {'N/A':>{col_width}s}"
        print(line)

    print(sep)
    print("* = best in category")
    print(f"{'='*len(header)}")

    # Alpha comparison vs SPY
    spy_ret = summaries.get("spy_benchmark", {}).get("annual_return")
    if spy_ret is not None:
        print(f"\nAlpha vs S&P 500 (SPY annual return: {spy_ret:.2%}):")
        for name in model_names:
            if name == "spy_benchmark":
                continue
            model_ret = summaries.get(name, {}).get("annual_return")
            if model_ret is not None:
                alpha = model_ret - spy_ret
                print(f"  {name:>15s}: {alpha:>+.2%}")


def save_comparison(
    comparison: dict,
    summaries: Dict[str, dict],
    all_results: Dict[str, pd.DataFrame],
    results_dir: str,
):
    """Save all comparison artifacts."""
    os.makedirs(results_dir, exist_ok=True)

    # Save comparison table as JSON
    comparison_json = {}
    for metric, row in comparison.items():
        comparison_json[metric] = {
            k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
            for k, v in row.items()
        }
    with open(os.path.join(results_dir, "model_comparison.json"), "w") as f:
        json.dump(comparison_json, f, indent=2)

    # Save per-model summaries
    for name, summary in summaries.items():
        clean = {}
        for k, v in summary.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                clean[k] = float(v)
            elif isinstance(v, dict):
                clean[k] = str(v)
            else:
                clean[k] = str(v)
        with open(os.path.join(results_dir, f"summary_{name}.json"), "w") as f:
            json.dump(clean, f, indent=2)

    # Save per-model returns
    for name, results_df in all_results.items():
        if not results_df.empty:
            results_df.to_csv(os.path.join(results_dir, f"returns_{name}.csv"))

    # Save comparison as CSV table
    df_rows = []
    for metric, row in comparison.items():
        entry = {"metric": metric}
        entry.update(row)
        df_rows.append(entry)
    if df_rows:
        pd.DataFrame(df_rows).to_csv(
            os.path.join(results_dir, "comparison_table.csv"), index=False,
        )

    logger.info(f"Comparison results saved to {results_dir}/")


def generate_comparison_plots(
    all_results: Dict[str, pd.DataFrame],
    comparison: dict,
    results_dir: str,
):
    """Generate comparative visualization across all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = [k for k in all_results if not all_results[k].empty]
    if not model_names:
        return

    colors = {
        "lightgbm": "#2196F3",
        "tst": "#FF5722",
        "crossmamba": "#4CAF50",
        "ensemble": "#9C27B0",
    }

    fig, axes = plt.subplots(4, 1, figsize=(14, 20), dpi=100)

    # 1. Cumulative returns comparison
    for name in model_names:
        res = all_results[name]
        cum = (1 + res["net_return"]).cumprod()
        c = colors.get(name, "gray")
        axes[0].plot(cum.index, cum, color=c, lw=1.5, label=name.upper())
    axes[0].axhline(1, color="gray", ls="--", alpha=0.5)
    axes[0].set_ylabel("Cumulative Return (Net)")
    axes[0].set_title("Cumulative Returns — All Models")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Drawdown comparison
    for name in model_names:
        res = all_results[name]
        cum = (1 + res["net_return"]).cumprod()
        dd = cum / cum.cummax() - 1
        c = colors.get(name, "gray")
        axes[1].plot(dd.index, dd, color=c, lw=1, alpha=0.7, label=name.upper())
    axes[1].set_ylabel("Drawdown")
    axes[1].set_title("Drawdown — All Models")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Rolling Sharpe comparison
    for name in model_names:
        res = all_results[name]
        rs = (
            res["net_return"].rolling(63).mean() * 252
            / (res["net_return"].rolling(63).std() * np.sqrt(252) + 1e-8)
        )
        c = colors.get(name, "gray")
        axes[2].plot(rs.index, rs, color=c, lw=1, alpha=0.7, label=name.upper())
    axes[2].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[2].set_ylabel("Rolling Sharpe (63d)")
    axes[2].set_title("Rolling Sharpe Ratio — All Models")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 4. Bar chart of key metrics
    metrics_to_bar = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Avg Rank IC"]
    bar_data = {}
    for metric in metrics_to_bar:
        if metric in comparison:
            bar_data[metric] = comparison[metric]

    if bar_data:
        bar_models = [m for m in model_names if m != "spy_benchmark"]
        n_metrics = len(bar_data)
        x = np.arange(n_metrics)
        bar_width = 0.8 / max(len(bar_models), 1)

        for i, model in enumerate(bar_models):
            vals = []
            for metric in bar_data:
                vals.append(bar_data[metric].get(model, 0))
            c = colors.get(model, "gray")
            axes[3].bar(x + i * bar_width, vals, bar_width, label=model.upper(), color=c, alpha=0.8)

        axes[3].set_xticks(x + bar_width * (len(bar_models) - 1) / 2)
        axes[3].set_xticklabels(list(bar_data.keys()), rotation=15)
        axes[3].set_title("Key Metrics Comparison")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison plot saved: {path}")
