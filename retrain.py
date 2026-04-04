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
    GitHub Actions uses newer pandas that creates StringDtype indexes.
    Older pandas on Mac can't unpickle these. Convert to plain object dtype.
    """
    import pandas as pd
    if "feature_importance" in data and isinstance(data["feature_importance"], pd.Series):
        fi = data["feature_importance"]
        data["feature_importance"] = pd.Series(
            fi.values,
            index=pd.Index([str(x) for x in fi.index]),
            name=fi.name,
        )
    if "feature_names" in data:
        data["feature_names"] = [str(f) for f in data["feature_names"]]
    return data

# Path to OpenClaw (for Alpaca data adapter)
OPENCLAW_PATH = os.environ.get(
    "OPENCLAW_PATH",
    os.path.join(os.path.dirname(__file__), "..", "VSNX", "openclaw-fintech"),
)


def _patch_for_alpaca():
    """
    Replace yfinance data fetchers with Alpaca adapter if available.
    This ensures retraining uses professional data, not yfinance scraping.
    """
    alpaca_key = os.environ.get("ALPACA_API_KEY", "")
    if not alpaca_key or alpaca_key in ("", "xxxxx"):
        logger.info("Alpaca not configured — using yfinance/cache for data")
        return

    # Add OpenClaw to path for the adapter
    if os.path.exists(OPENCLAW_PATH):
        import sys
        if OPENCLAW_PATH not in sys.path:
            sys.path.insert(0, OPENCLAW_PATH)
    else:
        logger.info(f"OpenClaw not found at {OPENCLAW_PATH} — using yfinance/cache")
        return

    try:
        from skills.market_data.adapter import (
            fetch_price_data as alpaca_prices,
            fetch_cross_asset_data as alpaca_cross_asset,
            fetch_news_sentiment as alpaca_sentiment,
        )

        import data_loader
        import sentiment_features

        data_loader.fetch_price_data = alpaca_prices
        data_loader.fetch_cross_asset_data = alpaca_cross_asset
        sentiment_features.fetch_news_sentiment = alpaca_sentiment

        # Also enhance sentiment with LLM analysis when Anthropic key available
        try:
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if anthropic_key and not anthropic_key.startswith("sk-ant-xxx"):
                from skills.news.llm_sentiment import (
                    analyze_articles_batch, compute_llm_sentiment_features,
                )
                _base_sentiment = sentiment_features.fetch_news_sentiment

                def _enhanced_sentiment(tickers, max_per_ticker=10, cache_dir="data"):
                    base = _base_sentiment(tickers, max_per_ticker, cache_dir)
                    try:
                        import asyncio
                        from skills.market_data import get_data_provider

                        async def _llm():
                            provider = get_data_provider()
                            articles = await provider.get_news(symbols=tickers[:30], limit=30)
                            dicts = [{"headline": a.headline, "summary": a.summary,
                                      "symbols": a.symbols, "source": a.source,
                                      "created_at": a.created_at.isoformat()} for a in articles]
                            return await analyze_articles_batch(dicts)

                        analyses = asyncio.run(_llm())
                        if analyses:
                            for t in tickers:
                                feats = compute_llm_sentiment_features(analyses, t)
                                if t in base:
                                    base[t].update(feats)
                                else:
                                    base[t] = feats
                            logger.info(f"LLM sentiment added for {len(tickers)} tickers")
                    except Exception as e:
                        logger.debug(f"LLM sentiment skipped: {e}")
                    return base

                sentiment_features.fetch_news_sentiment = _enhanced_sentiment
                logger.info("Sentiment enhanced with LLM (Claude Haiku)")
        except Exception:
            pass

        logger.info("Data fetchers patched to use Alpaca + LLM sentiment")
    except Exception as e:
        logger.warning(f"Could not patch for Alpaca: {e} — falling back to yfinance/cache")


def retrain(models_to_train: list[str] = None):
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

    # Fundamentals
    fundamentals = fetch_fundamental_data(tickers, cache_dir=cfg.data_dir)
    earnings_dates = fetch_earnings_dates(tickers, cache_dir=cfg.data_dir)
    fund_feats = build_fundamental_features(fundamentals, prices, earnings_dates, sector_map)

    # Sentiment
    sentiment_data = fetch_news_sentiment(tickers, cache_dir=cfg.data_dir)
    sent_feats = build_sentiment_features(sentiment_data, prices)

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

    # ── Step 2: Build features ───────────────────────────────────────
    logger.info("Step 2: Building features...")
    features, targets = build_all_features(prices, volumes, cfg.features, fund_feats, ca_feats)
    target = targets[f"fwd_ret_{cfg.features.primary_target_horizon}d"]
    X, y = panel_to_ml_format(features, target)
    logger.info(f"  ML dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # ── Step 3: Feature selection ────────────────────────────────────
    logger.info("Step 3: Feature selection...")
    from backtest import select_features_by_ic
    selected = select_features_by_ic(X, y, max_features=50, min_abs_ic=0.005, n_splits=3)
    X = X[selected]
    logger.info(f"  Selected {len(selected)} features")

    # ── Step 4: Train each model ─────────────────────────────────────
    # Single pass: train on last train_window_days, validate on most recent data
    train_window = cfg.model.train_window_days
    dates = sorted(X.index.get_level_values(0).unique()) if isinstance(X.index, pd.MultiIndex) else sorted(X.index.unique())

    # Split: train on all but last 21 days, validate on last 21 days
    val_size = 21
    train_dates = dates[:-val_size]
    val_dates = dates[-val_size:]

    if isinstance(X.index, pd.MultiIndex):
        X_train = X.loc[X.index.get_level_values(0).isin(train_dates)]
        y_train = y.loc[y.index.isin(X_train.index)]
        X_val = X.loc[X.index.get_level_values(0).isin(val_dates)]
        y_val = y.loc[y.index.isin(X_val.index)]
    else:
        X_train = X.loc[train_dates]
        y_train = y.loc[train_dates]
        X_val = X.loc[val_dates]
        y_val = y.loc[val_dates]

    logger.info(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    # Sample weights
    sample_weight = None
    try:
        from advanced_labeling import compute_sample_uniqueness
        labels_df = pd.DataFrame({"date": X_train.index.get_level_values(0) if isinstance(X_train.index, pd.MultiIndex) else X_train.index})
        sample_weight = compute_sample_uniqueness(labels_df, max_holding_days=10)
    except Exception:
        pass

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
