#!/usr/bin/env python3
"""
Quant-Grade Cross-Sectional Equity Ranking System
===================================================
Multi-Model Alpha Comparison: LightGBM vs TST vs CrossMamba

Usage:
    python main.py backtest                # LightGBM backtest (original)
    python main.py backtest --long-biased  # Bull-market mode 15L/5S
    python main.py backtest --optimize     # With Optuna HP tuning
    python main.py compare                 # Compare ALL 3 models + ensemble
    python main.py compare --models lightgbm,tst  # Compare specific models
    python main.py compare --no-ensemble   # Skip ensemble
    python main.py signal                  # Today's signal
    python main.py trade                   # Paper trade via Alpaca
    python main.py trade --live            # LIVE (careful!)
"""
import sys
import os
import logging
import argparse
from datetime import datetime

# Create directories BEFORE logging setup
for _d in ["data", "models", "logs", "results"]:
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/trading_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
logger = logging.getLogger(__name__)


def _apply_portfolio_args(cfg, args):
    """Apply common portfolio CLI args to config."""
    if hasattr(args, 'universe') and args.universe:
        cfg.data.universe_source = args.universe
    if hasattr(args, 'horizon') and args.horizon:
        cfg.features.primary_target_horizon = args.horizon
    if hasattr(args, 'long_only') and args.long_only:
        cfg.portfolio.long_short = False
        cfg.portfolio.max_positions_long = 20
        cfg.portfolio.max_positions_short = 0
    if hasattr(args, 'long_biased') and args.long_biased:
        cfg.portfolio.max_positions_long = 15
        cfg.portfolio.max_positions_short = 5
        cfg.portfolio.max_gross_leverage = 1.6
        cfg.portfolio.max_net_leverage = 0.60
        cfg.risk.sector_neutral = False
        cfg.risk.max_sector_net_pct = 0.10
        cfg.risk.target_annual_vol = 0.15
        cfg.risk.max_drawdown_threshold = -0.12
    if hasattr(args, 'n_long') and args.n_long is not None:
        cfg.portfolio.max_positions_long = args.n_long
    if hasattr(args, 'n_short') and args.n_short is not None:
        cfg.portfolio.max_positions_short = args.n_short
    return cfg


def cmd_backtest(args):
    from config import Config
    from backtest import run_backtest

    cfg = Config()
    cfg = _apply_portfolio_args(cfg, args)
    if args.no_fundamentals:
        pass  # Handled in backtest.py

    logger.info("Starting quant-grade backtest...")
    results, summary = run_backtest(cfg, optimize=args.optimize)

    if not results.empty:
        try:
            generate_plots(results, summary)
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")

    return results, summary


def cmd_compare(args):
    """Run all three alpha models + ensemble through the same pipeline and compare."""
    from config import Config
    from data_loader import (
        fetch_price_data, fetch_cross_asset_data,
        fetch_fundamental_data, fetch_earnings_dates,
    )
    from universe import get_universe, filter_universe_by_liquidity, load_sector_map
    from fundamental_features import build_fundamental_features
    from cross_asset_features import build_cross_asset_features
    from sentiment_features import fetch_news_sentiment, build_sentiment_features
    from features import build_all_features, panel_to_ml_format
    from backtest import select_features_by_ic
    from model_comparison import (
        run_comparison, print_comparison, save_comparison,
        generate_comparison_plots,
    )

    cfg = Config()
    cfg = _apply_portfolio_args(cfg, args)

    # Select which models to run
    if args.models:
        cfg.comparison.models_to_run = [m.strip() for m in args.models.split(",")]
    if args.no_ensemble:
        cfg.comparison.run_ensemble = False

    logger.info("Starting multi-model comparison...")
    logger.info(f"Models: {cfg.comparison.models_to_run}")

    for d in [cfg.data_dir, cfg.model_dir, cfg.results_dir]:
        os.makedirs(d, exist_ok=True)

    # ============================================================
    # DATA PIPELINE (shared across all models — identical features)
    # ============================================================
    logger.info("=" * 60)
    logger.info("STEP 1-7: Shared data pipeline (identical for all models)")
    logger.info("=" * 60)

    # 1. Universe & prices
    tickers = get_universe(cfg.data)
    prices, volumes = fetch_price_data(tickers, cfg.data, cache_dir=cfg.data_dir)
    tickers = filter_universe_by_liquidity(tickers, cfg.data, prices, volumes)
    prices = prices[[t for t in tickers if t in prices.columns]]
    volumes = volumes[[t for t in tickers if t in volumes.columns]]
    tickers = list(prices.columns)
    logger.info(f"Universe: {len(tickers)} tickers, {len(prices)} days")

    # 2. Sectors
    sector_map = load_sector_map(tickers, cache_dir=cfg.data_dir)

    # 3. Fundamentals
    fundamentals = fetch_fundamental_data(tickers, cache_dir=cfg.data_dir)
    earnings_dates = fetch_earnings_dates(tickers, cache_dir=cfg.data_dir)
    fund_feats = build_fundamental_features(fundamentals, prices, earnings_dates, sector_map)

    # 4. Sentiment
    sentiment_data = fetch_news_sentiment(tickers, cache_dir=cfg.data_dir)
    sent_feats = build_sentiment_features(sentiment_data, prices)

    # 5. Cross-asset
    all_ca = cfg.data.cross_asset_tickers + cfg.data.sector_etfs
    ca_prices = fetch_cross_asset_data(
        all_ca, prices.index[0].strftime("%Y-%m-%d"),
        prices.index[-1].strftime("%Y-%m-%d"), cache_dir=cfg.data_dir,
    )
    ca_only = ca_prices[[c for c in cfg.data.cross_asset_tickers if c in ca_prices.columns]] if not ca_prices.empty else pd.DataFrame()
    sect_etf = ca_prices[[c for c in cfg.data.sector_etfs if c in ca_prices.columns]] if not ca_prices.empty else pd.DataFrame()
    ca_feats = build_cross_asset_features(ca_only, prices, sect_etf, sector_map, cfg.features.cross_asset_windows)

    # 6. Feature engineering
    features, targets = build_all_features(
        prices, volumes, cfg.features,
        fundamental_feats=fund_feats,
        cross_asset_feats={**sent_feats, **ca_feats},
    )
    h = cfg.features.primary_target_horizon
    target_key = f"fwd_rank_{h}d"
    target = targets.get(target_key, targets.get(f"fwd_ret_{h}d"))
    X, y = panel_to_ml_format(features, target)

    # 7. Feature selection
    selected_features = select_features_by_ic(X, y, max_features=50)
    X = X[selected_features]

    # ============================================================
    # RUN ALL MODELS
    # ============================================================
    all_results, all_summaries, comparison = run_comparison(
        cfg=cfg, X=X, y=y,
        prices=prices, volumes=volumes,
        fundamentals=fundamentals,
        sector_map=sector_map,
        selected_features=selected_features,
    )

    # ============================================================
    # OUTPUT RESULTS
    # ============================================================
    print_comparison(comparison, all_summaries)

    # Build suffix for result filenames (e.g. "14L_7S" for long-biased)
    n_l = cfg.portfolio.max_positions_long
    n_s = cfg.portfolio.max_positions_short
    suffix = f"{n_l}L_{n_s}S" if (n_l != 10 or n_s != 10) else ""

    # Save everything
    save_comparison(comparison, all_summaries, all_results, cfg.results_dir, suffix=suffix)

    # Generate comparison plots
    try:
        generate_comparison_plots(all_results, comparison, cfg.results_dir, suffix=suffix)
    except Exception as e:
        logger.warning(f"Comparison plotting failed: {e}")

    # Also generate individual plots for each model
    sfx = f"_{suffix}" if suffix else ""
    for name, results in all_results.items():
        if not results.empty and name in all_summaries:
            try:
                generate_plots_named(results, all_summaries[name], f"{name}{sfx}", cfg.results_dir)
            except Exception as e:
                logger.warning(f"Plot for {name} failed: {e}")

    return all_results, all_summaries, comparison


def cmd_signal(args):
    from config import Config
    from signal_generator import SignalGenerator

    cfg = Config()
    if args.long_biased:
        cfg.portfolio.max_positions_long = 15
        cfg.portfolio.max_positions_short = 5
        cfg.portfolio.max_gross_leverage = 1.6
        cfg.portfolio.max_net_leverage = 0.60
        cfg.risk.sector_neutral = False
        cfg.risk.max_sector_net_pct = 0.10
        cfg.risk.target_annual_vol = 0.15
        cfg.risk.max_drawdown_threshold = -0.12
    if args.n_long is not None:
        cfg.portfolio.max_positions_long = args.n_long
    if args.n_short is not None:
        cfg.portfolio.max_positions_short = args.n_short

    sig = SignalGenerator(cfg)
    sig.load_model("models/latest_model.pkl")
    sig.initialize_risk()

    weights, info = sig.generate_signals()

    print(f"\n{'='*50}")
    print(f"TRADING SIGNALS — {info.get('date', 'N/A')}")
    print(f"{'='*50}")
    print(f"Regime score: {info.get('regime_score', 0):.4f}")
    print(f"Positions: {info.get('n_long', 0)}L / {info.get('n_short', 0)}S")
    print(f"Net exposure: {info.get('net_exposure', 0):.1%}")
    print(f"Gross exposure: {info.get('gross_exposure', 0):.1%}")
    print(f"\n{'LONGS':>10s}  {'Weight':>8s}")
    print(f"{'─'*10}  {'─'*8}")
    for t, w in weights[weights > 0].sort_values(ascending=False).items():
        print(f"{t:>10s}  {w:>8.2%}")
    print(f"\n{'SHORTS':>10s}  {'Weight':>8s}")
    print(f"{'─'*10}  {'─'*8}")
    for t, w in weights[weights < 0].sort_values().items():
        print(f"{t:>10s}  {w:>8.2%}")
    print(f"{'='*50}")

    # Save signals
    weights.to_csv("results/latest_signals.csv")
    logger.info("Signals saved to results/latest_signals.csv")


def cmd_trade(args):
    from config import Config
    from signal_generator import SignalGenerator
    from execution import AlpacaExecutor

    cfg = Config()
    if not cfg.execution.api_key or not cfg.execution.api_secret:
        print("\n⚠️  Set ALPACA_API_KEY and ALPACA_API_SECRET")
        print("Get free keys at: https://alpaca.markets")
        return

    if args.long_biased:
        cfg.portfolio.max_positions_long = 15
        cfg.portfolio.max_positions_short = 5
        cfg.portfolio.max_gross_leverage = 1.6
        cfg.portfolio.max_net_leverage = 0.60
        cfg.risk.sector_neutral = False
        cfg.risk.max_sector_net_pct = 0.10
        cfg.risk.target_annual_vol = 0.15
        cfg.risk.max_drawdown_threshold = -0.12
    if args.n_long is not None:
        cfg.portfolio.max_positions_long = args.n_long
    if args.n_short is not None:
        cfg.portfolio.max_positions_short = args.n_short

    paper = not args.live
    print(f"\nMode: {'PAPER' if paper else '🔴 LIVE'}")
    if not paper:
        if input("Type 'CONFIRM' for live trading: ") != "CONFIRM":
            print("Aborted.")
            return

    # 1. Generate signals (identical to backtest pipeline)
    sig = SignalGenerator(cfg)
    sig.load_model("models/latest_model.pkl")
    sig.initialize_risk()
    target_weights, info = sig.generate_signals()

    print(f"\n{'='*50}")
    print(f"TRADE PLAN — {info.get('date', 'N/A')}")
    print(f"{'='*50}")
    print(f"Regime: {info.get('regime_score', 0):.4f}")
    print(f"Positions: {info.get('n_long', 0)}L / {info.get('n_short', 0)}S")
    print(f"Net exposure: {info.get('net_exposure', 0):.1%}")
    print(f"\n{'TICKER':>10s}  {'Weight':>8s}  {'Side':>6s}")
    print(f"{'─'*10}  {'─'*8}  {'─'*6}")
    for t, w in target_weights.sort_values(ascending=False).items():
        side = "LONG" if w > 0 else "SHORT"
        print(f"{t:>10s}  {w:>8.2%}  {side:>6s}")

    # 2. Connect to Alpaca
    executor = AlpacaExecutor(
        cfg.execution.api_key, cfg.execution.api_secret, paper=paper,
    )
    account = executor.get_account()
    print(f"\nAccount equity: ${account['equity']:,.2f}")

    # 3. Show current vs target
    current = executor.get_positions()
    if not current.empty:
        print(f"\nCurrent positions: {len(current)}")
        for t, w in current.sort_values(ascending=False).items():
            print(f"  {t:>10s}: {w:>8.2%}")

    # 4. Confirm and execute
    all_tickers = target_weights.index.union(current.index)
    trades = target_weights.reindex(all_tickers, fill_value=0) - \
             current.reindex(all_tickers, fill_value=0)
    trades = trades[trades.abs() > 0.005]
    turnover = trades.abs().sum()

    print(f"\nTrades: {len(trades)}, turnover: {turnover:.1%}")
    if trades.empty:
        print("No trades needed.")
        return

    confirm = input(f"\nExecute {len(trades)} trades? (y/n): ")
    if confirm.lower() != "y":
        print("Aborted.")
        return

    results = executor.execute_target_portfolio(target_weights)
    print(f"\nExecuted {len(results)} orders:")
    for r in results:
        status = r.get("status", "?")
        symbol = r.get("symbol", "?")
        notional = r.get("notional", 0)
        if status == "error":
            error = r.get("error", "unknown")
            print(f"  {symbol:>10s}: FAILED — {error}")
        else:
            print(f"  {symbol:>10s}: ${notional:>10,.2f} ({status})")

    # Save trade log
    import json
    log_path = f"results/trades_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Trade log: {log_path}")


def generate_plots(results: "pd.DataFrame", summary: dict):
    generate_plots_named(results, summary, "strategy", "results")


def generate_plots_named(results: "pd.DataFrame", summary: dict, name: str, results_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    import pandas as pd

    fig, axes = plt.subplots(5, 1, figsize=(14, 20), dpi=100)

    title_prefix = name.upper()

    # 1. Cumulative returns
    cum = (1 + results["net_return"]).cumprod()
    cum_gross = (1 + results["gross_return"]).cumprod()
    axes[0].plot(cum.index, cum, "b-", lw=1.5, label="Net of costs")
    axes[0].plot(cum_gross.index, cum_gross, "b--", alpha=0.4, label="Gross")
    axes[0].axhline(1, color="gray", ls="--", alpha=0.5)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].set_title(
        f"{title_prefix} | Sharpe: {summary.get('sharpe_ratio',0):.2f} | "
        f"Ann: {summary.get('annual_return',0):.1%} | MaxDD: {summary.get('max_drawdown',0):.1%}"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Drawdown
    dd = cum / cum.cummax() - 1
    axes[1].fill_between(dd.index, dd, 0, color="red", alpha=0.3)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_title(f"{title_prefix} — Drawdown")
    axes[1].grid(True, alpha=0.3)

    # 3. Rolling Sharpe
    rs = results["net_return"].rolling(63).mean() * 252 / (results["net_return"].rolling(63).std() * np.sqrt(252) + 1e-8)
    axes[2].plot(rs.index, rs, "g-", lw=1)
    axes[2].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[2].axhline(1, color="green", ls="--", alpha=0.3)
    axes[2].axhline(-1, color="red", ls="--", alpha=0.3)
    axes[2].set_ylabel("Rolling Sharpe (63d)")
    axes[2].set_title(f"{title_prefix} — Rolling Sharpe Ratio")
    axes[2].grid(True, alpha=0.3)

    # 4. Net exposure
    axes[3].plot(results.index, results["net_exposure"], "purple", lw=1)
    axes[3].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[3].set_ylabel("Net Exposure")
    axes[3].set_title(f"{title_prefix} — Net Exposure (avg: {results['net_exposure'].mean():.2%})")
    axes[3].grid(True, alpha=0.3)

    # 5. Turnover
    axes[4].bar(results.index, results["turnover"], color="orange", alpha=0.5, width=1)
    axes[4].axhline(results["turnover"].mean(), color="orange", ls="--", alpha=0.7)
    axes[4].set_ylabel("Daily Turnover")
    axes[4].set_title(f"{title_prefix} — Turnover (avg: {results['turnover'].mean():.1%})")
    axes[4].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    path = os.path.join(results_dir, f"backtest_performance_{name}.png")
    plt.savefig(path, bbox_inches="tight")
    logger.info(f"Plot saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Quant-Grade Cross-Sectional Equity Ranking System\n"
                    "with Multi-Model Comparison (LightGBM / TST / CrossMamba)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # --- backtest (original LightGBM-only) ---
    bt = sub.add_parser("backtest", help="Run backtest with LightGBM (original)")
    bt.add_argument("--universe", choices=["sp500", "nasdaq100", "custom"])
    bt.add_argument("--horizon", type=int)
    bt.add_argument("--long-only", action="store_true")
    bt.add_argument("--long-biased", action="store_true", help="15L/5S for bull markets (captures beta + alpha)")
    bt.add_argument("--n-long", type=int, help="Number of long positions (e.g. 14)")
    bt.add_argument("--n-short", type=int, help="Number of short positions (e.g. 6)")
    bt.add_argument("--no-fundamentals", action="store_true")
    bt.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter optimization")

    # --- compare (multi-model) ---
    cmp = sub.add_parser("compare", help="Compare all alpha models (LightGBM, TST, CrossMamba, Ensemble)")
    cmp.add_argument("--models", type=str, default=None,
                     help="Comma-separated model list (default: lightgbm,tst,crossmamba)")
    cmp.add_argument("--no-ensemble", action="store_true", help="Skip ensemble combination")
    cmp.add_argument("--universe", choices=["sp500", "nasdaq100", "custom"])
    cmp.add_argument("--horizon", type=int)
    cmp.add_argument("--long-only", action="store_true")
    cmp.add_argument("--long-biased", action="store_true")
    cmp.add_argument("--n-long", type=int)
    cmp.add_argument("--n-short", type=int)

    # --- signal ---
    sig = sub.add_parser("signal", help="Generate today's trading signals")
    sig.add_argument("--long-biased", action="store_true")
    sig.add_argument("--n-long", type=int)
    sig.add_argument("--n-short", type=int)

    # --- trade ---
    trade = sub.add_parser("trade", help="Execute trades via Alpaca")
    trade.add_argument("--live", action="store_true")
    trade.add_argument("--long-biased", action="store_true")
    trade.add_argument("--n-long", type=int)
    trade.add_argument("--n-short", type=int)

    args = parser.parse_args()

    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "signal":
        cmd_signal(args)
    elif args.command == "trade":
        cmd_trade(args)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python main.py backtest              # LightGBM backtest")
        print("  python main.py compare               # Compare all 3 models + ensemble")
        print("  python main.py compare --models lightgbm,tst  # Compare specific models")


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    main()
