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


def _get_tradeable_at_date(
    date, pit_snapshots: Optional[dict], prices: pd.DataFrame, min_price: float = 5.0,
) -> Optional[set]:
    """
    Return the set of tradeable tickers at a given date.

    Applies two filters:
    1. PIT constituent membership (only stocks in S&P 500 at that date)
    2. Minimum price filter (excludes penny stocks, delisted stubs)

    Returns None if no filtering is needed (both filters disabled).
    """
    valid = None

    # PIT constituent gate
    if pit_snapshots:
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)[:10]
        # Find most recent snapshot <= date (snapshots are monthly)
        snapshot_dates = sorted(pit_snapshots.keys(), reverse=True)
        for snap_date in snapshot_dates:
            if snap_date <= date_str:
                valid = set(pit_snapshots[snap_date])
                break

    # Price filter (drop penny stocks / delisted stubs)
    if min_price > 0 and date in prices.index:
        day_prices = prices.loc[date].dropna()
        tradeable = set(day_prices[day_prices >= min_price].index)
        if valid is not None:
            valid = valid & tradeable
        else:
            valid = tradeable

    return valid


def run_single_model_pipeline(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    fundamentals: dict,
    sector_map: dict,
    fmp_historical: dict = None,
    pit_snapshots: Optional[dict] = None,
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
    max_feats = getattr(cfg.features, "max_features", 0)
    models, oos_predictions, metrics_df = walk_forward_train(
        X, y, cfg.model, cfg.features,
        model_type=model_type,
        model_cfg=model_cfg,
        max_features=max_feats,
    )

    if oos_predictions.empty:
        logger.error(f"[{model_type}] No predictions generated")
        return pd.DataFrame(), {}, pd.DataFrame(), pd.Series(dtype=float)

    # Risk model
    risk = FactorRiskModel(cfg.risk)
    if len(prices) > 252:
        _fund = fundamentals
        if fmp_historical:
            from fmp_features import get_pit_fundamentals
            _fund = get_pit_fundamentals(fmp_historical, str(prices.index[-1].date()))
        risk.estimate(prices, _fund or fundamentals, prices.index[-1], lookback=504)
        risk.update_regime(prices)

    # Portfolio construction
    constructor = PortfolioConstructor(cfg.portfolio)
    weights_history: Dict[pd.Timestamp, pd.Series] = {}
    portfolio_returns = pd.Series(dtype=float)
    stock_vol = np.log(prices / prices.shift(1)).rolling(63).std() * np.sqrt(252)
    # Short-side filter inputs (prevent short squeeze)
    stock_mom_6m = prices.pct_change(126).shift(1)  # 6-month trailing return, lagged 1 day
    rolling_52w_high = prices.rolling(252, min_periods=60).max().shift(1)
    dist_from_52w_high = (prices.shift(1) / rolling_52w_high) - 1  # negative = below high
    prev_weights = pd.Series(dtype=float)

    # Diagnostic logger — captures per-day, per-position state for post-run analysis
    from diagnostics import DiagnosticLogger
    diag_enabled = getattr(cfg, "diagnostics_enabled", True)
    diag_dir = os.path.join(cfg.results_dir, "diagnostics")
    diag = DiagnosticLogger(enabled=diag_enabled, output_dir=diag_dir)
    if diag_enabled:
        logger.info(f"Diagnostics enabled → {diag_dir}")
    # Position open tracker — keyed by ticker, holds entry context for matching close events
    position_state: Dict[str, dict] = {}

    # VIX series for portfolio-level short gate (Citadel regime proxy)
    vix_series = None
    try:
        ca_path = os.path.join(cfg.data_dir, "cross_asset.csv")
        if os.path.exists(ca_path):
            _ca = pd.read_csv(ca_path, index_col=0, parse_dates=True)
            if "^VIX" in _ca.columns:
                vix_series = _ca["^VIX"]
                logger.info(f"Loaded VIX for short regime gate ({len(vix_series)} dates)")
    except Exception as e:
        logger.debug(f"VIX load failed: {e}")

    # PIT market cap and earnings yield matrices for universal short quality filters
    # (blocks the "IonQ pattern": small-cap unprofitable speculative shorts)
    mcap_matrix = None  # date × ticker
    ey_matrix = None    # date × ticker
    if fmp_historical:
        try:
            from fundamental_features import _broadcast_pit
            mcap_matrix = _broadcast_pit(fmp_historical, "marketCap", list(prices.columns), prices.index)
            # Earnings yield = 1/PE (with PE > 0)
            pe_matrix = _broadcast_pit(fmp_historical, "trailingPE", list(prices.columns), prices.index)
            ey_matrix = 1.0 / pe_matrix.where(pe_matrix > 0)
            ey_matrix = ey_matrix.clip(-10, 10)  # bound extremes
            logger.info(f"Loaded mcap + EY matrices for short quality filters "
                        f"(mcap coverage: {mcap_matrix.notna().sum().sum()/mcap_matrix.size*100:.0f}%)")
        except Exception as e:
            logger.warning(f"Short quality filter data failed: {e}")
            mcap_matrix = None
            ey_matrix = None

    n_long = cfg.portfolio.max_positions_long
    n_short = cfg.portfolio.max_positions_short

    if isinstance(oos_predictions.index, pd.MultiIndex):
        dates = sorted(oos_predictions.index.get_level_values(0).unique())
    else:
        dates = []

    risk_reestimate_every = 63
    last_risk_estimate = 0

    # Log filter config
    use_pit = getattr(cfg.portfolio, "use_pit_constituent_gate", False)
    effective_pit = pit_snapshots if use_pit else None
    min_price = getattr(cfg.portfolio, "min_stock_price", 5.0)
    if use_pit and pit_snapshots:
        logger.info(f"Portfolio filtering: PIT constituents + price >= ${min_price:.2f}")
    else:
        logger.info(f"Portfolio filtering: price >= ${min_price:.2f} (PIT gate disabled)")

    for di, date in enumerate(dates):
        if isinstance(oos_predictions.index, pd.MultiIndex):
            day_preds = oos_predictions.loc[date]
        else:
            continue
        if not isinstance(day_preds, pd.Series) or len(day_preds) == 0:
            continue

        # Apply tradability filters BEFORE portfolio construction:
        # - PIT constituent gate (optional, off by default — too restrictive)
        # - Price filter (always on — excludes penny stocks / delisted stubs)
        tradeable = _get_tradeable_at_date(date, effective_pit, prices, min_price=min_price)
        if tradeable is not None:
            day_preds = day_preds[day_preds.index.isin(tradeable)]
            if len(day_preds) < n_long + n_short:
                continue  # not enough tradeable stocks

        if di - last_risk_estimate >= risk_reestimate_every:
            if date in prices.index:
                _fund_at_date = fundamentals
                if fmp_historical:
                    from fmp_features import get_pit_fundamentals
                    _fund_at_date = get_pit_fundamentals(fmp_historical, str(date.date()))
                risk.estimate(prices, _fund_at_date or fundamentals, date, lookback=504)
                risk.update_regime(prices.loc[:date])
                last_risk_estimate = di

        vol_est = stock_vol.loc[date] if date in stock_vol.index else None
        mom_6m = stock_mom_6m.loc[date] if date in stock_mom_6m.index else None
        dist_high = dist_from_52w_high.loc[date] if date in dist_from_52w_high.index else None
        curr_prices = prices.loc[date] if date in prices.index else None
        vix_today = None
        if vix_series is not None and date in vix_series.index:
            vix_today = float(vix_series.loc[date])
        mcap_today = mcap_matrix.loc[date] if mcap_matrix is not None and date in mcap_matrix.index else None
        ey_today = ey_matrix.loc[date] if ey_matrix is not None and date in ey_matrix.index else None

        target_weights = constructor.construct_portfolio(
            predictions=day_preds, date=date,
            prev_weights=prev_weights, vol_estimates=vol_est,
            sector_map=sector_map,
            momentum_6m=mom_6m,
            dist_from_52w_high=dist_high,
            current_prices=curr_prices,
            vix_current=vix_today,
            mcap_current=mcap_today,
            earnings_yield_current=ey_today,
        )

        # Risk scaling removed — portfolio constructor now handles sector caps,
        # position limits, and turnover internally. Factor neutralization and
        # vol/drawdown/tail risk scaling were destroying alpha signal.
        # See portfolio.py docstring for design rationale.

        if not portfolio_returns.empty:
            risk.update(portfolio_returns.iloc[-1])

        # Compute today's realized return using PRE-UPDATE prev_weights
        # (what was actually held overnight from yesterday to today).
        # Previously used target_weights — that's same-day execution look-ahead.
        day_ret_for_dd = 0.0  # default: no position = no P&L change
        long_pnl = 0.0
        short_pnl = 0.0
        if date in prices.index and not prev_weights.empty:
            prev_idx = prices.index.get_loc(date) - 1
            if prev_idx >= 0:
                prev_date = prices.index[prev_idx]
                common = prev_weights.index.intersection(prices.columns)
                if len(common) > 0:
                    stock_rets = prices.loc[date, common] / prices.loc[prev_date, common] - 1
                    weighted_rets = prev_weights.reindex(common, fill_value=0) * stock_rets
                    day_ret = weighted_rets.sum()
                    # Decompose by leg for diagnostics
                    long_mask = prev_weights.reindex(common, fill_value=0) > 0
                    short_mask = prev_weights.reindex(common, fill_value=0) < 0
                    long_pnl = float(weighted_rets[long_mask].sum())
                    short_pnl = float(weighted_rets[short_mask].sum())
                    portfolio_returns = pd.concat([
                        portfolio_returns, pd.Series([day_ret], index=[date])
                    ])
                    day_ret_for_dd = day_ret
        # ALWAYS advance the DD circuit breaker state. If prev_weights is empty
        # (e.g., after a dust-filter collapse), realized return is 0 and cum/peak
        # stay constant — but we avoid freezing the update loop, so if the next
        # construct_portfolio rebuilds a position, we start recording P&L again.
        constructor.update_cum_return(day_ret_for_dd)

        # ============================================================
        # DIAGNOSTIC LOGGING
        # ============================================================
        if diag_enabled:
            # Per-day record (decision context + realized P&L)
            target_long_mask = target_weights > 0
            target_short_mask = target_weights < 0
            target_long_sum = float(target_weights[target_long_mask].sum())
            target_short_sum = float(target_weights[target_short_mask].abs().sum())

            # Sector concentration of target portfolio
            top_long_sec = None
            top_long_sec_pct = None
            top_short_sec = None
            top_short_sec_pct = None
            if sector_map and not target_weights.empty:
                long_w = target_weights[target_long_mask]
                short_w = target_weights[target_short_mask].abs()
                if not long_w.empty:
                    long_secs = pd.Series({t: sector_map.get(t, "Unknown") for t in long_w.index})
                    sec_sum_long = long_w.groupby(long_secs).sum()
                    if not sec_sum_long.empty:
                        top_long_sec = str(sec_sum_long.idxmax())
                        top_long_sec_pct = float(sec_sum_long.max() / target_long_sum) if target_long_sum > 0 else None
                if not short_w.empty:
                    short_secs = pd.Series({t: sector_map.get(t, "Unknown") for t in short_w.index})
                    sec_sum_short = short_w.groupby(short_secs).sum()
                    if not sec_sum_short.empty:
                        top_short_sec = str(sec_sum_short.idxmax())
                        top_short_sec_pct = float(sec_sum_short.max() / target_short_sum) if target_short_sum > 0 else None

            day_record = {
                "date": date,
                "n_universe": int(len(day_preds)),
                # Prediction signal
                "pred_mean": float(day_preds.mean()),
                "pred_std": float(day_preds.std()),
                "pred_p10": float(day_preds.quantile(0.10)),
                "pred_p90": float(day_preds.quantile(0.90)),
                # Regime context
                "vix": vix_today,
                # Portfolio state (target)
                "n_long": int(target_long_mask.sum()),
                "n_short": int(target_short_mask.sum()),
                "long_gross": target_long_sum,
                "short_gross": target_short_sum,
                "gross_exposure": float(target_weights.abs().sum()),
                "net_exposure": float(target_weights.sum()),
                # Risk control state
                "current_dd": float(constructor.get_current_dd()),
                "cum_return": float(constructor._cum_return),
                "peak_return": float(constructor._peak_return),
                "dd_circuit_active": bool(
                    constructor.get_current_dd() < getattr(cfg.portfolio, "dd_circuit_breaker_threshold", -1.0)
                ),
                "vix_short_gate_active": bool(
                    vix_today is not None
                    and vix_today > getattr(cfg.portfolio, "short_max_vix", 999.0)
                ),
                # Realized P&L (from prev_weights)
                "gross_return": day_ret_for_dd,
                "long_pnl": long_pnl,
                "short_pnl": short_pnl,
                # Sector concentration
                "top_long_sector": top_long_sec,
                "top_long_sector_pct": top_long_sec_pct,
                "top_short_sector": top_short_sec,
                "top_short_sector_pct": top_short_sec_pct,
            }
            diag.log_day(day_record)

            # Position event tracking — open/close detection by comparing prev_weights and target_weights
            if not target_weights.empty or position_state:
                tickers_now = set(target_weights.index) if not target_weights.empty else set()
                tickers_prev = set(position_state.keys())

                new_opens = tickers_now - tickers_prev
                new_closes = tickers_prev - tickers_now

                # Log opens
                for t in new_opens:
                    w = float(target_weights.get(t, 0))
                    side = "L" if w > 0 else "S"
                    entry_price = float(curr_prices.get(t, np.nan)) if curr_prices is not None else None
                    if entry_price is None or not np.isfinite(entry_price):
                        entry_price = None
                    ticker_pred = float(day_preds.get(t, np.nan))
                    ticker_pred = ticker_pred if np.isfinite(ticker_pred) else None
                    # Compute rank within day
                    rank_pct = float(day_preds.rank(pct=True).get(t, np.nan))
                    rank_pct = rank_pct if np.isfinite(rank_pct) else None
                    # Entry features
                    e_vol = float(vol_est.get(t, np.nan)) if vol_est is not None else None
                    e_vol = e_vol if e_vol is not None and np.isfinite(e_vol) else None
                    e_mom_6m = float(mom_6m.get(t, np.nan)) if mom_6m is not None else None
                    e_mom_6m = e_mom_6m if e_mom_6m is not None and np.isfinite(e_mom_6m) else None

                    open_rec = {
                        "event_type": "open",
                        "date": date,
                        "ticker": str(t),
                        "side": side,
                        "entry_price": entry_price,
                        "entry_pred": ticker_pred,
                        "entry_rank": rank_pct,
                        "entry_weight": w,
                        "sector": sector_map.get(t, "Unknown") if sector_map else None,
                        "entry_vol_63d": e_vol,
                        "entry_mom_126d": e_mom_6m,
                        "entry_vix": vix_today,
                    }
                    diag.log_position_event(open_rec)
                    position_state[t] = {
                        "entry_date": date,
                        "entry_price": entry_price,
                        "side": side,
                    }

                # Log closes
                for t in new_closes:
                    pos = position_state.pop(t)
                    exit_price = float(curr_prices.get(t, np.nan)) if curr_prices is not None else None
                    exit_price = exit_price if exit_price is not None and np.isfinite(exit_price) else None
                    pnl_pct = None
                    if exit_price is not None and pos.get("entry_price") and pos["entry_price"] > 0:
                        sign = 1 if pos["side"] == "L" else -1
                        pnl_pct = sign * (exit_price / pos["entry_price"] - 1)

                    close_rec = {
                        "event_type": "close",
                        "date": date,
                        "ticker": str(t),
                        "side": pos["side"],
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "rank_drop_or_filter",
                    }
                    diag.log_position_event(close_rec)

        weights_history[date] = target_weights
        prev_weights = target_weights

    # Compute full results
    results_df = compute_portfolio_returns(weights_history, prices, cfg.portfolio)

    if results_df.empty:
        logger.error(f"[{model_type}] No returns computed")
        return pd.DataFrame(), {}, metrics_df, oos_predictions

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

    # Add portfolio exposure, turnover, and position metrics
    if "gross_exposure" in results_df.columns:
        summary["avg_gross_exposure"] = float(results_df["gross_exposure"].mean())
        summary["max_gross_exposure"] = float(results_df["gross_exposure"].max())
        summary["avg_net_exposure"] = float(results_df["net_exposure"].mean())
        summary["min_net_exposure"] = float(results_df["net_exposure"].min())
        summary["max_net_exposure"] = float(results_df["net_exposure"].max())
    if "turnover" in results_df.columns:
        summary["avg_turnover"] = float(results_df["turnover"].mean())
    if "n_long" in results_df.columns:
        summary["avg_n_long"] = float(results_df["n_long"].mean())
        summary["avg_n_short"] = float(results_df["n_short"].mean())
    if "tc_cost" in results_df.columns:
        summary["total_tc_cost"] = float(results_df["tc_cost"].sum())
        summary["avg_daily_tc_bps"] = float(results_df["tc_cost"].mean() * 10000)

    # Long vs short attribution
    if "gross_return" in results_df.columns and "net_exposure" in results_df.columns:
        gross_exp = results_df["gross_exposure"]
        net_exp = results_df["net_exposure"]
        # Approximate: long_weight ≈ (gross + net) / 2, short_weight ≈ (gross - net) / 2
        avg_long_wt = float(((gross_exp + net_exp) / 2).mean())
        avg_short_wt = float(((gross_exp - net_exp) / 2).mean())
        summary["avg_long_weight"] = avg_long_wt
        summary["avg_short_weight"] = avg_short_wt

    # Drawdown duration
    cum = (1 + results_df["net_return"]).cumprod()
    dd = cum / cum.cummax() - 1
    in_dd = dd < 0
    if in_dd.any():
        dd_groups = (~in_dd).cumsum()
        dd_durations = in_dd.groupby(dd_groups).sum()
        dd_durations = dd_durations[dd_durations > 0]
        if len(dd_durations) > 0:
            summary["max_dd_duration_days"] = int(dd_durations.max())
            summary["avg_dd_duration_days"] = float(dd_durations.mean())

    # Monthly returns breakdown
    monthly_rets = results_df["net_return"].resample("ME").apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
    )
    if len(monthly_rets) > 0:
        summary["best_month"] = float(monthly_rets.max())
        summary["worst_month"] = float(monthly_rets.min())
        summary["pct_positive_months"] = float((monthly_rets > 0).mean())
        summary["monthly_returns"] = {
            str(d.date()): round(float(v), 6) for d, v in monthly_rets.items()
        }

    # Save model
    if models:
        model_path = os.path.join(cfg.model_dir, f"latest_{model_type}_model.pkl")
        models[-1].save(model_path)

    # Flush diagnostic logger to disk
    if diag_enabled:
        n_l = cfg.portfolio.max_positions_long
        n_s = cfg.portfolio.max_positions_short
        diag_suffix = f"{n_l}L_{n_s}S"

        # Capture per-window metrics from the walk-forward run
        if not metrics_df.empty:
            for _, row in metrics_df.iterrows():
                diag.log_window(row.to_dict())

        diag.flush(model_type=model_type, suffix=diag_suffix)

    return results_df, summary, metrics_df, oos_predictions


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
    fmp_historical: dict = None,
    pit_snapshots: Optional[dict] = None,
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

        results_df, summary, metrics_df, oos_preds = run_single_model_pipeline(
            model_type=model_type,
            X=X, y=y, cfg=cfg,
            prices=prices, volumes=volumes,
            fundamentals=fundamentals,
            sector_map=sector_map,
            fmp_historical=fmp_historical,
            pit_snapshots=pit_snapshots,
        )

        # Save OOS predictions to CSV for offline ensemble experiments
        if not oos_preds.empty:
            oos_path = os.path.join(cfg.results_dir, f"oos_predictions_{model_type}.csv")
            oos_preds.to_csv(oos_path)
            logger.info(f"Saved OOS predictions: {oos_path}")

        all_results[model_type] = results_df
        all_summaries[model_type] = summary
        all_metrics[model_type] = metrics_df
        all_predictions[model_type] = oos_preds

    # Ensemble
    if cfg.comparison.run_ensemble and len(all_predictions) > 1:
        logger.info(f"\n{'#'*60}")
        logger.info("# ENSEMBLE (weighted average)")
        logger.info(f"{'#'*60}")

        ensemble_preds = build_ensemble_predictions(
            all_predictions, cfg.comparison.ensemble_weights,
        )

        if not ensemble_preds.empty:
            # Run ensemble through portfolio pipeline
            risk = FactorRiskModel(cfg.risk)
            if len(prices) > 252:
                _fund = fundamentals
                if fmp_historical:
                    from fmp_features import get_pit_fundamentals
                    _fund = get_pit_fundamentals(fmp_historical, str(prices.index[-1].date()))
                risk.estimate(prices, _fund or fundamentals, prices.index[-1], lookback=504)
                risk.update_regime(prices)

            constructor = PortfolioConstructor(cfg.portfolio)
            weights_history = {}
            portfolio_returns = pd.Series(dtype=float)
            stock_vol = np.log(prices / prices.shift(1)).rolling(63).std() * np.sqrt(252)
            # Pass filter inputs so ensemble pipeline matches single-model pipeline
            stock_mom_6m_ens = prices.pct_change(126).shift(1)
            rolling_52w_high_ens = prices.rolling(252, min_periods=60).max().shift(1)
            dist_from_52w_high_ens = (prices.shift(1) / rolling_52w_high_ens) - 1
            prev_weights = pd.Series(dtype=float)
            n_long = cfg.portfolio.max_positions_long
            n_short = cfg.portfolio.max_positions_short

            # VIX for short regime gate (matches single-model pipeline)
            _vix_series_ens = None
            try:
                _ca_path = os.path.join(cfg.data_dir, "cross_asset.csv")
                if os.path.exists(_ca_path):
                    _ca = pd.read_csv(_ca_path, index_col=0, parse_dates=True)
                    if "^VIX" in _ca.columns:
                        _vix_series_ens = _ca["^VIX"]
            except Exception:
                pass

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
                        _fund_at_date = fundamentals
                        if fmp_historical:
                            from fmp_features import get_pit_fundamentals
                            _fund_at_date = get_pit_fundamentals(fmp_historical, str(date.date()))
                        risk.estimate(prices, _fund_at_date or fundamentals, date, lookback=504)
                        risk.update_regime(prices.loc[:date])
                        last_risk_estimate = di

                vol_est = stock_vol.loc[date] if date in stock_vol.index else None
                mom_6m_ens = stock_mom_6m_ens.loc[date] if date in stock_mom_6m_ens.index else None
                dist_high_ens = dist_from_52w_high_ens.loc[date] if date in dist_from_52w_high_ens.index else None
                curr_prices_ens = prices.loc[date] if date in prices.index else None
                vix_today_ens = None
                if _vix_series_ens is not None and date in _vix_series_ens.index:
                    vix_today_ens = float(_vix_series_ens.loc[date])

                target_weights = constructor.construct_portfolio(
                    predictions=day_preds, date=date,
                    prev_weights=prev_weights, vol_estimates=vol_est,
                    sector_map=sector_map,
                    momentum_6m=mom_6m_ens,
                    dist_from_52w_high=dist_high_ens,
                    current_prices=curr_prices_ens,
                    vix_current=vix_today_ens,
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

                # Compute today's return using PRE-UPDATE prev_weights (held overnight),
                # not target_weights (would be same-day execution look-ahead)
                day_ret_for_dd = 0.0
                if date in prices.index and not prev_weights.empty:
                    prev_idx = prices.index.get_loc(date) - 1
                    if prev_idx >= 0:
                        prev_date = prices.index[prev_idx]
                        common = prev_weights.index.intersection(prices.columns)
                        if len(common) > 0:
                            day_ret = (
                                (prices.loc[date, common] / prices.loc[prev_date, common] - 1)
                                * prev_weights.reindex(common, fill_value=0)
                            ).sum()
                            portfolio_returns = pd.concat([
                                portfolio_returns, pd.Series([day_ret], index=[date])
                            ])
                            day_ret_for_dd = day_ret
                # Always advance DD state (uses 0 when prev_weights is empty,
                # preventing dust-collapse death spiral)
                constructor.update_cum_return(day_ret_for_dd)

                weights_history[date] = target_weights
                prev_weights = target_weights

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
    suffix: str = "",
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
    sfx = f"_{suffix}" if suffix else ""
    with open(os.path.join(results_dir, f"model_comparison{sfx}.json"), "w") as f:
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
        with open(os.path.join(results_dir, f"summary_{name}{sfx}.json"), "w") as f:
            json.dump(clean, f, indent=2)

    # Save per-model returns
    for name, results_df in all_results.items():
        if not results_df.empty:
            results_df.to_csv(os.path.join(results_dir, f"returns_{name}{sfx}.csv"))

    # Save comparison as CSV table
    df_rows = []
    for metric, row in comparison.items():
        entry = {"metric": metric}
        entry.update(row)
        df_rows.append(entry)
    if df_rows:
        pd.DataFrame(df_rows).to_csv(
            os.path.join(results_dir, f"comparison_table{sfx}.csv"), index=False,
        )

    logger.info(f"Comparison results saved to {results_dir}/")


def generate_comparison_plots(
    all_results: Dict[str, pd.DataFrame],
    comparison: dict,
    results_dir: str,
    suffix: str = "",
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
    sfx = f"_{suffix}" if suffix else ""
    path = os.path.join(results_dir, f"model_comparison{sfx}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison plot saved: {path}")
