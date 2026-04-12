#!/usr/bin/env python3
"""
Fast production retraining — trains on ALL available data in a single pass.

This is NOT a backtest. It trains one model on the full dataset and saves it.
Use this for the 14-day scheduled retraining, not `main.py compare`.

Speed comparison:
  main.py compare  → 54 walk-forward windows → 1-2 hours on CPU
  retrain.py       → 1 training pass          → 5-10 minutes on CPU

Usage:
  python retrain.py                      # retrain all models
  python retrain.py --models crossmamba  # retrain just CrossMamba
  python retrain.py --models crossmamba,lightgbm
"""
import argparse
import logging
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("retrain")


def _fix_for_pickle(data: dict) -> dict:
    """
    Fix StringDtype indexes before pickling.
    GitHub Actions uses newer pandas that creates StringDtype indexes by default.
    Older pandas on Mac can't unpickle these. Force object dtype explicitly.
    """
    import pandas as pd
    import numpy as np

    if "feature_importance" in data and isinstance(data["feature_importance"], pd.Series):
        fi = data["feature_importance"]
        # Force object dtype — pd.Index(strings) creates StringDtype on newer pandas
        # but np.array(strings, dtype=object) guarantees plain object dtype
        idx = np.array([str(x) for x in fi.index], dtype=object)
        data["feature_importance"] = pd.Series(
            fi.values.copy(),
            index=pd.Index(idx, dtype=object),
            name=fi.name,
        )
    if "feature_names" in data:
        data["feature_names"] = [str(f) for f in data["feature_names"]]
    return data



def _patch_for_alpaca():
    """
    Replace yfinance data fetchers with local Alpaca adapter.
    No OpenClaw dependency — uses alpaca_adapter/ in this repo.
    """
    alpaca_key = os.environ.get("ALPACA_API_KEY", "")
    if not alpaca_key or alpaca_key in ("", "xxxxx"):
        logger.info("Alpaca not configured — using yfinance/cache for data")
        return

    try:
        from alpaca_adapter import (
            fetch_price_data as alpaca_prices,
            fetch_cross_asset_data as alpaca_cross_asset,
        )

        import data_loader

        data_loader.fetch_price_data = alpaca_prices
        data_loader.fetch_cross_asset_data = alpaca_cross_asset

        logger.info("Data fetchers patched to use Alpaca")
    except Exception as e:
        logger.warning(f"Could not patch for Alpaca: {e} — falling back to yfinance/cache")


def retrain(models_to_train: list[str] = None):
    from config import Config
    from data_loader import (
        fetch_price_data, fetch_cross_asset_data,
        fetch_fundamental_data, fetch_earnings_dates,
    )
    from universe import get_universe, filter_universe_by_liquidity, load_sector_map
    from fundamental_features import build_fundamental_features, build_pit_fundamental_features
    from cross_asset_features import build_cross_asset_features
    # Sentiment removed from ML model — used only in agent layer (OpenClaw)
    from features import build_all_features, panel_to_ml_format
    from model import EnsembleRanker, create_model

    if models_to_train is None:
        models_to_train = ["crossmamba", "lightgbm"]

    cfg = Config()
    start = time.time()

    # ── Patch data fetchers to use Alpaca if available ────────────────
    _patch_for_alpaca()

    # ── Step 1: Fetch data (uses Alpaca if keys available, else yfinance/cache)
    logger.info("Step 1: Fetching data...")
    tickers = get_universe(cfg.data)
    prices, volumes = fetch_price_data(tickers, cfg.data, cache_dir=cfg.data_dir)
    tickers = filter_universe_by_liquidity(tickers, cfg.data, prices, volumes)
    prices = prices[[t for t in tickers if t in prices.columns]]
    volumes = volumes[[t for t in tickers if t in volumes.columns]]
    tickers = list(prices.columns)
    logger.info(f"  Universe: {len(tickers)} tickers, {len(prices)} days")

    # Sectors
    sector_map = load_sector_map(tickers, cache_dir=cfg.data_dir)

    # Fundamentals — POINT-IN-TIME (no look-ahead bias)
    # Mirrors main.py: prefer FMP historical quarterly with filingDate, fall back
    # to yfinance snapshot only if FMP is unavailable.
    fmp_historical = None
    fundamentals = None
    if cfg.data.fmp_api_key:
        try:
            from fmp_data_provider import fetch_fmp_historical_fundamentals, get_pit_fundamentals
            fmp_historical = fetch_fmp_historical_fundamentals(
                tickers, cfg.data.fmp_api_key, cfg.data_dir,
            )
            if fmp_historical and len(fmp_historical) > len(tickers) * 0.3:
                logger.info(f"  FMP historical PIT: {len(fmp_historical)} tickers")
                # Latest snapshot for risk model (current state, not historical)
                fundamentals = get_pit_fundamentals(
                    fmp_historical, datetime.now().strftime("%Y-%m-%d"),
                )
            else:
                fmp_historical = None
        except Exception as e:
            logger.warning(f"  FMP historical failed: {e}")
            fmp_historical = None
    if fundamentals is None:
        fundamentals = fetch_fundamental_data(tickers, cache_dir=cfg.data_dir)
    earnings_dates = fetch_earnings_dates(tickers, cache_dir=cfg.data_dir)

    if fmp_historical:
        # PIT fundamentals — uses filingDate, no look-ahead
        fund_feats = build_pit_fundamental_features(
            fmp_historical, prices, earnings_dates, sector_map,
        )
    else:
        logger.warning("  Using yfinance fundamentals (KNOWN LOOK-AHEAD BIAS)")
        fund_feats = build_fundamental_features(fundamentals, prices, earnings_dates, sector_map)

    # Cross-asset
    all_ca = cfg.data.cross_asset_tickers + cfg.data.sector_etfs
    ca_prices = fetch_cross_asset_data(
        all_ca, prices.index[0].strftime("%Y-%m-%d"),
        prices.index[-1].strftime("%Y-%m-%d"),
        cache_dir=cfg.data_dir,
    )
    ca_only = ca_prices[[c for c in cfg.data.cross_asset_tickers if c in ca_prices.columns]] if not ca_prices.empty else pd.DataFrame()
    sect_etf = ca_prices[[c for c in cfg.data.sector_etfs if c in ca_prices.columns]] if not ca_prices.empty else pd.DataFrame()
    ca_feats = build_cross_asset_features(ca_only, prices, sect_etf, sector_map, cfg.features.cross_asset_windows)

    # ── Step 2a: Additional data sources ────────────────────────────
    insider_feats = {}
    try:
        from insider_features import fetch_insider_data, build_insider_features
        insider_data = fetch_insider_data(tickers, cache_dir=cfg.data_dir)
        insider_feats = build_insider_features(insider_data, prices, fundamentals)
        logger.info(f"  Insider signals: {len(insider_feats)}")
    except Exception as e:
        logger.debug(f"Insider features skipped: {e}")

    fmp_feats = {}
    try:
        from fmp_features import fetch_fmp_fundamentals, build_fmp_features
        fmp_data = fetch_fmp_fundamentals(tickers, cfg.data.fmp_api_key, cfg.data_dir)
        fmp_feats = build_fmp_features(fmp_data, prices)
        logger.info(f"  FMP signals: {len(fmp_feats)}")
    except Exception as e:
        logger.debug(f"FMP features skipped: {e}")

    openbb_feats = {}
    try:
        from openbb_features import fetch_options_data, fetch_short_interest, build_openbb_features
        # live_mode=False: train on the same feature set validated in backtest.
        # OpenBB features (options IV, short interest) lack reliable historical data,
        # so they are excluded from training and only used as supplemental signal in
        # live inference (signal_generator.py uses live_mode=True).
        options_data = fetch_options_data(tickers, cache_dir=cfg.data_dir, live_mode=False)
        short_data = fetch_short_interest(tickers, cache_dir=cfg.data_dir, live_mode=False)
        openbb_feats = build_openbb_features(options_data, short_data, prices)
        logger.info(f"  OpenBB signals: {len(openbb_feats)}")
    except Exception as e:
        logger.debug(f"OpenBB features skipped: {e}")

    # ── Step 2b: Premium features ──────────────────────────────────────
    premium_feats = {}
    try:
        from fmp_data_provider import build_premium_features
        premium_feats = build_premium_features(tickers, prices, cfg.data.fmp_api_key, cfg.data_dir)
        if premium_feats:
            logger.info(f"  Premium features: {len(premium_feats)} signals")
    except Exception as e:
        logger.debug(f"Premium features skipped: {e}")

    # ── Step 2c: Build features ──────────────────────────────────────
    logger.info("Step 2: Building features...")
    features, targets = build_all_features(
        prices, volumes, cfg.features,
        fundamental_feats=fund_feats,
        cross_asset_feats=ca_feats,
        insider_feats=insider_feats,
        fmp_feats=fmp_feats,
        openbb_feats=openbb_feats,
        premium_feats=premium_feats,
        sector_map=sector_map,
    )

    # Use risk-adjusted target (matches backtest)
    h = cfg.features.primary_target_horizon
    target_type = getattr(cfg.features, "target_type", "risk_adjusted")
    if target_type == "risk_adjusted":
        target_key = f"fwd_risk_adj_{h}d"
    elif target_type == "industry_relative" and sector_map:
        target_key = f"fwd_ind_rel_{h}d"
    else:
        target_key = f"fwd_rank_{h}d"
    target = targets.get(target_key, targets.get(f"fwd_ret_{h}d"))
    logger.info(f"  Target: {target_key}")

    X, y = panel_to_ml_format(features, target)
    logger.info(f"  ML dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # ── Step 3: Train/val split FIRST (before feature selection to avoid leakage) ──
    # Single pass: train on all but last val_size + embargo days, validate on last val_size
    # Embargo prevents label overlap between train and val (horizon=10, embargo=10).
    embargo = cfg.model.embargo_days
    val_size = 21
    dates = sorted(X.index.get_level_values(0).unique()) if isinstance(X.index, pd.MultiIndex) else sorted(X.index.unique())

    if len(dates) <= val_size + embargo + 100:
        logger.warning(f"Too few dates ({len(dates)}) for embargoed split, using simple split")
        train_dates = dates[:-val_size]
        val_dates = dates[-val_size:]
    else:
        train_dates = dates[:-(val_size + embargo)]
        val_dates = dates[-val_size:]

    if isinstance(X.index, pd.MultiIndex):
        X_train_full = X.loc[X.index.get_level_values(0).isin(train_dates)]
        y_train_full = y.loc[y.index.isin(X_train_full.index)]
        X_val_full = X.loc[X.index.get_level_values(0).isin(val_dates)]
        y_val_full = y.loc[y.index.isin(X_val_full.index)]
    else:
        X_train_full = X.loc[train_dates]
        y_train_full = y.loc[train_dates]
        X_val_full = X.loc[val_dates]
        y_val_full = y.loc[val_dates]

    # ── Step 4: Feature selection on TRAINING DATA ONLY (no val leakage) ──
    logger.info("Step 4: Feature selection (training data only)...")
    from backtest import select_features_by_ic
    max_feats = getattr(cfg.features, "max_features", 50)
    selected = select_features_by_ic(
        X_train_full, y_train_full, max_features=max_feats, min_abs_ic=0.005, n_splits=3,
    )
    X_train = X_train_full[selected]
    X_val = X_val_full[selected]
    y_train = y_train_full
    y_val = y_val_full
    logger.info(f"  Selected {len(selected)} features from training data")

    # ── Step 5: Train each model ─────────────────────────────────────
    train_window = cfg.model.train_window_days
    logger.info(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    # Sample weights: uniqueness (Lopez de Prado) × contemporaneous VIX regime weight.
    # Matches main walk-forward loop in model.py — see _compute_regime_sample_weights.
    # Down-weights training samples from high-VIX regimes (COVID, 2022, 2023 stress)
    # using ONLY data available at each sample's date. Causal, no hardcoded dates.
    sample_weight = None
    try:
        from advanced_labeling import compute_sample_uniqueness
        labels_df = pd.DataFrame({
            "date": X_train.index.get_level_values(0)
                    if isinstance(X_train.index, pd.MultiIndex)
                    else X_train.index
        })
        sample_weight = compute_sample_uniqueness(labels_df, max_holding_days=10)
    except Exception:
        pass

    # VIX regime weighting (import from model.py to keep the logic single-sourced)
    try:
        from model import _load_vix_regime_series, _compute_regime_sample_weights
        vix_series = _load_vix_regime_series(cfg.data_dir)
        if vix_series is not None:
            train_dates_idx = (
                X_train.index.get_level_values(0)
                if isinstance(X_train.index, pd.MultiIndex)
                else X_train.index
            )
            regime_w = _compute_regime_sample_weights(
                train_dates_idx, vix_series,
                vix_ref=20.0, vix_sat=35.0, floor_weight=0.30,
            )
            if regime_w is not None:
                if sample_weight is None:
                    sample_weight = regime_w
                else:
                    sample_weight = np.asarray(sample_weight) * regime_w
                # Renormalize to mean 1 so LightGBM gradient scale stays comparable
                sample_weight = (
                    sample_weight * len(sample_weight) / sample_weight.sum()
                )
                n_down = int((regime_w < 0.5).sum())
                logger.info(
                    f"  VIX regime weighting: {n_down}/{len(regime_w)} samples at <0.5× "
                    f"(avg regime_w={regime_w.mean():.3f})"
                )
    except Exception as e:
        logger.debug(f"  VIX regime weighting skipped: {e}")

    for model_name in models_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name.upper()}...")
        logger.info(f"{'='*60}")

        # Force garbage collection between models to prevent memory issues
        import gc
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        model_start = time.time()

        if model_name == "lightgbm":
            model = EnsembleRanker(cfg.model)
            metrics = model.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight)

            # Save (fix StringDtype for cross-pandas compatibility)
            save_path = f"models/latest_lightgbm_model.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(_fix_for_pickle({
                    "models": model.models,
                    "feature_names": model.feature_names,
                    "feature_importance": model.feature_importance,
                    "config": cfg.model,
                }), f)

        elif model_name in ("crossmamba", "tst"):
            model_cfg = cfg.crossmamba if model_name == "crossmamba" else cfg.tst
            # Reduce batch size for CPU training to prevent segfaults
            model_cfg.batch_size = 64
            model_cfg.epochs = 2  # fewer epochs for faster retraining
            model = create_model(model_name, model_cfg)
            logger.info(f"  Config: batch_size={model_cfg.batch_size}, epochs={model_cfg.epochs}, seq_len={model_cfg.sequence_length}")

            # For sequence models, need context
            seq_len = getattr(model_cfg, "sequence_length", 21)
            logger.info(f"  Starting {model_name} training (this may take several minutes)...")
            try:
                metrics = model.train(X_train, y_train, X_val, y_val)
            except Exception as e:
                logger.error(f"  {model_name} training failed: {e}")
                logger.info(f"  Skipping {model_name}, LightGBM will be used as fallback")
                continue

            # Predict on validation
            if isinstance(X.index, pd.MultiIndex):
                context_start = max(0, len(train_dates) - seq_len)
                context_dates = dates[context_start:]
                X_ctx = X.loc[X.index.get_level_values(0).isin(context_dates)]
            else:
                X_ctx = X

            val_pred = model.predict(X_ctx)
            if isinstance(val_pred.index, pd.MultiIndex):
                val_pred = val_pred[val_pred.index.get_level_values(0).isin(val_dates)]

            if len(val_pred) > 0 and len(y_val) > 0:
                common = val_pred.index.intersection(y_val.index)
                if len(common) > 2:
                    ic = np.corrcoef(val_pred.loc[common], y_val.loc[common])[0, 1]
                    logger.info(f"  Validation IC: {ic:.4f}")

            # Save (fix StringDtype for cross-pandas compatibility)
            save_path = f"models/latest_{model_name}_model.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(_fix_for_pickle({
                    "models": model.models if hasattr(model, "models") else [],
                    "feature_names": model.feature_names,
                    "feature_importance": model.feature_importance if hasattr(model, "feature_importance") else pd.Series(dtype=float),
                    "config": model_cfg,
                    "model_states": [m.state_dict() for m in model.models] if hasattr(model, "models") and model.models and hasattr(model.models[0], "state_dict") else [],
                    "n_features": len(model.feature_names),
                }), f)

        model_elapsed = time.time() - model_start
        logger.info(f"  {model_name} trained in {model_elapsed:.1f}s")
        logger.info(f"  Saved to models/latest_{model_name}_model.pkl")

    # Mark retrain complete (for feedback loop)
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "VSNX", "openclaw-fintech"))
        from skills.feedback.loop import mark_retrain_complete
        mark_retrain_complete()
        logger.info("Marked retrain complete in feedback system")
    except Exception:
        pass

    total = time.time() - start
    logger.info(f"\nTotal retrain time: {total:.1f}s")
    logger.info(f"Models trained: {', '.join(models_to_train)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast production model retraining")
    parser.add_argument("--models", default="crossmamba,lightgbm",
                        help="Comma-separated models to train (crossmamba,lightgbm,tst)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    retrain(models)
