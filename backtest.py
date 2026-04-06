"""
Quant-grade backtesting engine v4.

Key improvements over v3:
1. Stability-based feature selection (IC must be consistent across periods)
2. SPY benchmark comparison in output
3. Long-biased mode support (15L/5S to capture market returns + alpha)
4. Tighter feature cap (50 max — less noise)
5. Regime overlay scales with portfolio mode
"""
import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Dict, Tuple, Optional
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
from model import EnsembleRanker, walk_forward_train, create_model
from risk_model import FactorRiskModel
from portfolio import (
    PortfolioConstructor, compute_portfolio_returns, compute_performance_metrics,
)

logger = logging.getLogger(__name__)


# =====================================================================
# FEATURE SELECTION — stability-based
# =====================================================================
def select_features_by_ic(
    X: pd.DataFrame, y: pd.Series,
    max_features: int = 50,
    min_abs_ic: float = 0.005,
    n_splits: int = 3,
) -> list:
    """
    Stability-based feature selection.

    Instead of computing IC on the full sample (which can overfit to
    one regime), we split the data into n_splits time periods and
    require features to have consistent IC sign across periods.

    Score = mean(abs(IC)) * sign_consistency
    where sign_consistency = fraction of periods where IC has same sign

    This selects features with STABLE predictive power, not just strong
    but potentially spurious IC.
    """
    logger.info(f"Feature selection: screening {X.shape[1]} features across {n_splits} periods...")

    if isinstance(X.index, pd.MultiIndex):
        dates = sorted(X.index.get_level_values(0).unique())
    else:
        dates = sorted(X.index.unique())

    # Split dates into periods
    period_size = len(dates) // n_splits
    if period_size < 30:
        n_splits = max(2, len(dates) // 30)
        period_size = len(dates) // n_splits

    period_ics = {col: [] for col in X.columns}

    for p in range(n_splits):
        start = p * period_size
        end = (p + 1) * period_size if p < n_splits - 1 else len(dates)
        period_dates = dates[start:end]

        if isinstance(X.index, pd.MultiIndex):
            mask = X.index.get_level_values(0).isin(period_dates)
        else:
            mask = X.index.isin(period_dates)

        X_p = X.loc[mask]
        y_p = y.loc[y.index.isin(X_p.index)]

        if len(X_p) < 100:
            continue

        for col in X.columns:
            try:
                valid = X_p[col].notna() & y_p.notna()
                if valid.sum() < 50:
                    continue
                ic = X_p.loc[valid, col].corr(y_p.loc[valid], method="spearman")
                if not np.isnan(ic):
                    period_ics[col].append(ic)
            except Exception:
                continue

    # Score features
    scores = {}
    for col, ics in period_ics.items():
        if len(ics) < 2:
            continue
        mean_abs_ic = np.mean([abs(x) for x in ics])
        # Sign consistency: do all periods agree on direction?
        signs = [np.sign(x) for x in ics if x != 0]
        if not signs:
            continue
        dominant_sign = np.sign(np.sum(signs))
        consistency = sum(1 for s in signs if s == dominant_sign) / len(signs)
        # Stability score: strong IC × consistent direction
        scores[col] = mean_abs_ic * consistency

    if not scores:
        logger.warning("Feature selection failed, using all features")
        return list(X.columns)

    score_series = pd.Series(scores).sort_values(ascending=False)
    selected = score_series[score_series >= min_abs_ic].head(max_features).index.tolist()

    if len(selected) < 15:
        selected = score_series.head(min(30, len(score_series))).index.tolist()

    logger.info(f"  Selected {len(selected)}/{X.shape[1]} features")
    for feat, sc in score_series.head(10).items():
        n_periods = len(period_ics.get(feat, []))
        logger.info(f"    {feat:>50s}: score={sc:.4f} ({n_periods} periods)")

    return selected



# =====================================================================
# SPY BENCHMARK
# =====================================================================
def compute_spy_benchmark(
    start_date: str, end_date: str, cache_dir: str = "data",
) -> Optional[pd.Series]:
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


# =====================================================================
# MAIN BACKTEST
# =====================================================================
def run_backtest(cfg: Config, optimize: bool = False) -> Tuple[pd.DataFrame, dict]:
    for d in [cfg.data_dir, cfg.model_dir, cfg.results_dir]:
        os.makedirs(d, exist_ok=True)

    # ==================================================================
    # 1. UNIVERSE & PRICE DATA
    # ==================================================================
    _log_step(1, "Universe & price data")
    tickers = get_universe(cfg.data)
    prices, volumes = fetch_price_data(tickers, cfg.data, cache_dir=cfg.data_dir)
    tickers = filter_universe_by_liquidity(tickers, cfg.data, prices, volumes)
    prices = prices[[t for t in tickers if t in prices.columns]]
    volumes = volumes[[t for t in tickers if t in volumes.columns]]
    tickers = list(prices.columns)
    logger.info(f"Universe: {len(tickers)} tickers, {len(prices)} days")

    # ==================================================================
    # 2. SECTOR DATA
    # ==================================================================
    _log_step(2, "Sector classification")
    sector_map = load_sector_map(tickers, cache_dir=cfg.data_dir)

    # ==================================================================
    # 3. FUNDAMENTAL DATA
    # ==================================================================
    _log_step(3, "Fundamental data")
    fundamentals = fetch_fundamental_data(tickers, cache_dir=cfg.data_dir)
    earnings_dates = fetch_earnings_dates(tickers, cache_dir=cfg.data_dir)
    fund_feats = build_fundamental_features(fundamentals, prices, earnings_dates, sector_map)
    logger.info(f"Fundamental signals: {len(fund_feats)}")

    # ==================================================================
    # 4. NEWS SENTIMENT
    # ==================================================================
    _log_step(4, "News sentiment")
    sentiment_data = fetch_news_sentiment(tickers, cache_dir=cfg.data_dir)
    sent_feats = build_sentiment_features(sentiment_data, prices)
    logger.info(f"Sentiment signals: {len(sent_feats)}")

    # ==================================================================
    # 5. CROSS-ASSET DATA
    # ==================================================================
    _log_step(5, "Cross-asset signals")
    all_ca = cfg.data.cross_asset_tickers + cfg.data.sector_etfs
    ca_prices = fetch_cross_asset_data(
        all_ca, prices.index[0].strftime("%Y-%m-%d"),
        prices.index[-1].strftime("%Y-%m-%d"), cache_dir=cfg.data_dir,
    )
    ca_only = ca_prices[[c for c in cfg.data.cross_asset_tickers if c in ca_prices.columns]] if not ca_prices.empty else pd.DataFrame()
    sect_etf = ca_prices[[c for c in cfg.data.sector_etfs if c in ca_prices.columns]] if not ca_prices.empty else pd.DataFrame()
    ca_feats = build_cross_asset_features(ca_only, prices, sect_etf, sector_map, cfg.features.cross_asset_windows)
    logger.info(f"Cross-asset signals: {len(ca_feats)}")

    # ==================================================================
    # 5.5 INSIDER TRADING DATA (SEC Form 4)
    # ==================================================================
    _log_step(5.5, "Insider trading data")
    try:
        from insider_features import fetch_insider_data, build_insider_features
        insider_data = fetch_insider_data(tickers, cache_dir=cfg.data_dir)
        insider_feats = build_insider_features(insider_data, prices, fundamentals)
        logger.info(f"Insider signals: {len(insider_feats)}")
    except Exception as e:
        logger.info(f"Insider features skipped: {e}")
        insider_feats = {}

    # ==================================================================
    # 5.6 FMP POINT-IN-TIME FUNDAMENTALS
    # ==================================================================
    _log_step(5.6, "FMP fundamentals (point-in-time)")
    try:
        from fmp_features import fetch_fmp_fundamentals, build_fmp_features
        fmp_data = fetch_fmp_fundamentals(tickers, cfg.data.fmp_api_key, cfg.data_dir)
        fmp_feats = build_fmp_features(fmp_data, prices)
        logger.info(f"FMP signals: {len(fmp_feats)}")
    except Exception as e:
        logger.info(f"FMP features skipped: {e}")
        fmp_feats = {}

    # ==================================================================
    # 6. BUILD ALL FEATURES (with interactions + new data sources)
    # ==================================================================
    _log_step(6, "Feature engineering (with institutional interactions)")
    features, targets = build_all_features(
        prices, volumes, cfg.features,
        fundamental_feats=fund_feats,
        cross_asset_feats={**sent_feats, **ca_feats},
        insider_feats=insider_feats,
        fmp_feats=fmp_feats,
        sector_map=sector_map,
    )
    h = cfg.features.primary_target_horizon

    # Select target based on config (risk-adjusted is default)
    target_type = getattr(cfg.features, "target_type", "raw_rank")
    if target_type == "risk_adjusted":
        target_key = f"fwd_risk_adj_{h}d"
    elif target_type == "industry_relative" and sector_map:
        target_key = f"fwd_ind_rel_{h}d"
    else:
        target_key = f"fwd_rank_{h}d"

    target = targets.get(target_key, targets.get(f"fwd_ret_{h}d"))
    logger.info(f"Target: {target_key}")

    X, y = panel_to_ml_format(features, target)

    # ==================================================================
    # 7. FEATURE SELECTION — stability-based (expanded cap for new features)
    # ==================================================================
    _log_step(7, "Feature selection (stability-based)")
    max_feats = getattr(cfg.features, "max_features", 65)
    selected_features = select_features_by_ic(
        X, y, max_features=max_feats, min_abs_ic=0.005, n_splits=3,
    )
    X = X[selected_features]

    # ==================================================================
    # 8. OPTUNA OPTIMIZATION (optional)
    # ==================================================================
    if optimize:
        _log_step(8, "Optuna hyperparameter optimization")
        from optuna_tuner import optimize_hyperparameters, apply_optuna_params
        best_params = optimize_hyperparameters(
            X, y, n_trials=50, n_cv_windows=5,
            train_window=cfg.model.train_window_days,
        )
        if best_params:
            cfg.model = apply_optuna_params(cfg.model, best_params)
            with open(os.path.join(cfg.results_dir, "optuna_best_params.json"), "w") as f:
                json.dump(best_params, f, indent=2)
            logger.info("Optuna params applied and saved")
        else:
            logger.info("Optuna returned no params, using defaults")
    else:
        opt_path = os.path.join(cfg.results_dir, "optuna_best_params.json")
        if os.path.exists(opt_path):
            from optuna_tuner import apply_optuna_params
            with open(opt_path) as f:
                best_params = json.load(f)
            cfg.model = apply_optuna_params(cfg.model, best_params)
            logger.info("Loaded previous Optuna params")

    # ==================================================================
    # 9. TRAIN MODELS
    # ==================================================================
    _log_step(9, "Walk-forward training (ensemble)")
    models, oos_predictions, _metrics_df = walk_forward_train(X, y, cfg.model, cfg.features)

    if oos_predictions.empty:
        logger.error("No predictions generated")
        return pd.DataFrame(), {}

    # ==================================================================
    # 10. FACTOR RISK MODEL
    # ==================================================================
    _log_step(10, "Factor risk model")
    risk = FactorRiskModel(cfg.risk)
    if len(prices) > 252:
        risk.estimate(prices, fundamentals, prices.index[-1], lookback=504)
        risk.update_regime(prices)

    # ==================================================================
    # 12. PORTFOLIO CONSTRUCTION
    # ==================================================================
    _log_step(12, "Portfolio construction + risk management")
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

        # Re-estimate risk model periodically
        if di - last_risk_estimate >= risk_reestimate_every:
            if date in prices.index:
                risk.estimate(prices, fundamentals, date, lookback=504)
                risk.update_regime(prices.loc[:date])
                last_risk_estimate = di

        vol_est = stock_vol.loc[date] if date in stock_vol.index else None

        # Build portfolio
        target_weights = constructor.construct_portfolio(
            predictions=day_preds, date=date,
            prev_weights=prev_weights, vol_estimates=vol_est,
        )

        # Apply risk management (includes clip + regime)
        target_weights = risk.apply_risk_scaling(
            target_weights, portfolio_returns, sector_map,
            n_long=n_long, n_short=n_short,
        )

        # Position limits
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

    logger.info(f"Portfolios: {len(weights_history)} dates")

    # ==================================================================
    # 13. PERFORMANCE ANALYSIS
    # ==================================================================
    _log_step(13, "Performance analysis")
    results_df = compute_portfolio_returns(weights_history, prices, cfg.portfolio)

    if results_df.empty:
        logger.error("No returns computed")
        return pd.DataFrame(), {}

    summary = compute_performance_metrics(results_df["net_return"])

    # SPY Benchmark
    spy_returns = compute_spy_benchmark(
        str(results_df.index[0].date()),
        str(results_df.index[-1].date()),
        cfg.data_dir,
    )
    if spy_returns is not None:
        spy_metrics = compute_performance_metrics(spy_returns)
        summary["spy_annual_return"] = spy_metrics.get("annual_return", np.nan)
        summary["spy_sharpe"] = spy_metrics.get("sharpe_ratio", np.nan)
        summary["spy_max_dd"] = spy_metrics.get("max_drawdown", np.nan)
        # Alpha vs SPY
        summary["alpha_vs_spy"] = summary.get("annual_return", 0) - spy_metrics.get("annual_return", 0)

    # Risk decomposition
    if risk.factor_exposures is not None and len(weights_history) > 0:
        last_weights = list(weights_history.values())[-1]
        risk_info = risk.get_portfolio_risk(last_weights)
        summary["factor_risk"] = risk_info

    summary.update({
        "strategy": "quant_grade_v4",
        "mode": "long_biased" if n_long > n_short else "market_neutral",
        "n_tickers": len(tickers),
        "n_features_total": features.shape[1] // max(len(tickers), 1),
        "n_features_selected": len(selected_features),
        "target_horizon": h,
        "n_long": n_long,
        "n_short": n_short,
        "ensemble_size": cfg.model.n_ensemble,
        "sector_neutral": cfg.risk.sector_neutral,
        "vol_target": cfg.risk.target_annual_vol,
        "optimized": optimize,
    })

    _print_summary(summary, results_df)

    # Save
    results_df.to_csv(os.path.join(cfg.results_dir, "backtest_returns.csv"))
    with open(os.path.join(cfg.results_dir, "backtest_summary.json"), "w") as f:
        json.dump({k: str(v) for k, v in summary.items()}, f, indent=2)

    if models:
        fi = models[-1].feature_importance
        if fi is not None:
            fi.to_csv(os.path.join(cfg.results_dir, "feature_importance.csv"))
            print("\nTop 25 Features:")
            for feat, imp in fi.head(25).items():
                print(f"  {feat:>55s}: {imp:>8.1f}")
        models[-1].save(os.path.join(cfg.model_dir, "latest_model.pkl"))

    # Factor risk
    if "factor_risk" in summary:
        fr = summary["factor_risk"]
        print(f"\nFactor Risk Decomposition:")
        print(f"  Total Vol:    {fr.get('total_vol', 0):.1%}")
        print(f"  Factor Vol:   {fr.get('factor_vol', 0):.1%}")
        print(f"  Specific Vol: {fr.get('specific_vol', 0):.1%}")
        if "factor_exposures" in fr:
            print(f"  Factor Exposures:")
            for f, e in fr["factor_exposures"].items():
                print(f"    {f:>12s}: {e:>+.3f}")

    print(f"\nFeature Selection:")
    print(f"  Total features: {summary.get('n_features_total', '?')}")
    print(f"  Selected:       {summary.get('n_features_selected', '?')}")

    return results_df, summary


def _log_step(n, name):
    logger.info("=" * 60)
    logger.info(f"STEP {n}: {name}")
    logger.info("=" * 60)


def _print_summary(summary: dict, results: pd.DataFrame):
    mode = summary.get("mode", "market_neutral")
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS (v4 — {mode})")
    print(f"{'='*60}")
    print(f"Period: {results.index[0].date()} → {results.index[-1].date()}")
    print(f"Trading days: {summary.get('n_days', 'N/A')}")
    print(f"Features: {summary.get('n_features_selected', 'N/A')} (from {summary.get('n_features_total', '?')})")
    print(f"Ensemble: {summary.get('ensemble_size', 'N/A')} models")
    print(f"Target: {summary.get('target_horizon', '?')}d forward rank")
    print(f"Optimized: {summary.get('optimized', False)}")
    print("-" * 40)
    print(f"{'STRATEGY':>20s} {'SPY':>10s}")
    print(f"{'─'*20} {'─'*10}")
    ann_ret = summary.get('annual_return', 0)
    spy_ret = summary.get('spy_annual_return')
    sharpe = summary.get('sharpe_ratio', 0)
    spy_sharpe = summary.get('spy_sharpe')
    max_dd = summary.get('max_drawdown', 0)
    spy_dd = summary.get('spy_max_dd')

    spy_ret_s = f"{spy_ret:>9.2%}" if spy_ret is not None else "     N/A"
    spy_sh_s = f"{spy_sharpe:>9.2f}" if spy_sharpe is not None else "     N/A"
    spy_dd_s = f"{spy_dd:>9.2%}" if spy_dd is not None else "     N/A"

    print(f"Annual Return:   {ann_ret:>9.2%}  {spy_ret_s}")
    print(f"Volatility:      {summary.get('annual_volatility', 0):>9.2%}")
    print(f"Sharpe:          {sharpe:>9.2f}  {spy_sh_s}")
    print(f"Sortino:         {summary.get('sortino_ratio', 0):>9.2f}")
    print(f"Max Drawdown:    {max_dd:>9.2%}  {spy_dd_s}")
    print(f"Calmar:          {summary.get('calmar_ratio', 0):>9.2f}")
    print(f"Win Rate:        {summary.get('win_rate', 0):>9.2%}")
    print(f"Profit Factor:   {summary.get('profit_factor', 0):>9.2f}")

    alpha = summary.get('alpha_vs_spy')
    if alpha is not None:
        print(f"Alpha vs SPY:    {alpha:>9.2%}")

    print("-" * 40)
    print(f"Avg Turnover:    {results['turnover'].mean():>9.2%}")
    print(f"Avg TC (bps):    {results['tc_cost'].mean() * 10000:>9.1f}")
    print(f"Avg Positions:   {results['n_long'].mean():>5.1f}L / {results['n_short'].mean():.1f}S")
    print(f"Net Exposure:    {results['net_exposure'].mean():>9.2%}")
    print(f"Gross Exposure:  {results['gross_exposure'].mean():>9.2%}")
    print(f"{'='*60}")
