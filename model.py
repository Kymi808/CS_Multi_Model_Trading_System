"""
Alpha model ensemble rankers with walk-forward validation.

Supports three architectures:
- LightGBM (gradient boosting) — default, fast, interpretable
- TST (Time Series Transformer) — attention-based sequence model
- CrossMamba (selective state-space) — linear-time sequence model

All share the same interface: train(), predict(), save(), load()
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union
from config import ModelConfig, FeatureConfig, TSTConfig, CrossMambaConfig

logger = logging.getLogger(__name__)


def create_model(model_type: str, cfg) -> "Union[EnsembleRanker, object]":
    """Factory function to create a model by name."""
    if model_type == "lightgbm":
        return EnsembleRanker(cfg)
    elif model_type == "tst":
        from models.tst_model import TSTRanker
        return TSTRanker(cfg)
    elif model_type == "crossmamba":
        from models.crossmamba_model import CrossMambaRanker
        return CrossMambaRanker(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class EnsembleRanker:
    """Ensemble of LightGBM models for cross-sectional ranking."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.models: List[lgb.LGBMRegressor] = []
        self.feature_names: List[str] = []
        self.feature_importance: Optional[pd.Series] = None

    def _get_lgb_params(self, seed: int) -> dict:
        # n_jobs is configurable so callers can reduce OMP thread count when
        # running multiple LightGBM fits concurrently (avoids thread contention).
        n_jobs = getattr(self.cfg, "lightgbm_n_jobs", -1)
        params = {
            "n_estimators": self.cfg.n_estimators,
            "max_depth": self.cfg.max_depth,
            "learning_rate": self.cfg.learning_rate,
            "num_leaves": self.cfg.num_leaves,
            "min_child_samples": self.cfg.min_child_samples,
            "subsample": self.cfg.subsample,
            "colsample_bytree": self.cfg.colsample_bytree,
            "reg_alpha": self.cfg.reg_alpha,
            "reg_lambda": self.cfg.reg_lambda,
            "min_split_gain": self.cfg.min_split_gain,
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        # Use GPU if available (CUDA-compatible LightGBM)
        try:
            import torch
            if torch.cuda.is_available():
                params["device"] = "gpu"
                params["gpu_use_dp"] = False  # single precision is fine for ranking
        except ImportError:
            pass
        return params

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> dict:
        self.feature_names = list(X_train.columns)
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        if X_val is not None:
            X_val = X_val.replace([np.inf, -np.inf], np.nan)

        self.models = []
        all_importances = []

        for i, seed in enumerate(self.cfg.ensemble_seeds[:self.cfg.n_ensemble]):
            model = lgb.LGBMRegressor(**self._get_lgb_params(seed))
            callbacks = [lgb.log_evaluation(period=0)]

            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight

            if X_val is not None and y_val is not None:
                callbacks.append(lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False))
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                          eval_metric="l2", callbacks=callbacks, **fit_kwargs)
            else:
                model.fit(X_train, y_train, callbacks=callbacks, **fit_kwargs)

            self.models.append(model)
            all_importances.append(
                pd.Series(model.feature_importances_, index=self.feature_names)
            )

        # Average feature importance across ensemble
        self.feature_importance = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)

        metrics = {
            "n_train": len(X_train),
            "n_features": len(self.feature_names),
            "n_ensemble": len(self.models),
        }

        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics["val_ic"] = np.corrcoef(val_pred, y_val)[0, 1]
            metrics["val_rank_ic"] = pd.Series(val_pred.values).corr(
                pd.Series(y_val.values), method="spearman"
            )
            logger.info(
                f"  IC: {metrics['val_ic']:.4f}, "
                f"Rank IC: {metrics['val_rank_ic']:.4f} "
                f"(ensemble of {len(self.models)})"
            )

        return metrics

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.models:
            raise ValueError("Model not trained")
        X = X.replace([np.inf, -np.inf], np.nan)
        missing = set(self.feature_names) - set(X.columns)
        for f in missing:
            X[f] = np.nan

        # Average predictions from ensemble
        preds = np.zeros(len(X))
        for model in self.models:
            preds += model.predict(X[self.feature_names])
        preds /= len(self.models)

        return pd.Series(preds, index=X.index)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "feature_names": self.feature_names,
                "feature_importance": self.feature_importance,
                "config": self.cfg,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.models = data["models"]
        self.feature_names = data["feature_names"]
        self.feature_importance = data["feature_importance"]


def _load_vix_regime_series(data_dir: str = "data") -> Optional[pd.Series]:
    """
    Load VIX time series for regime-aware sample weighting.

    Returns None if unavailable — the training loop falls back to uniform weights.
    Used to down-weight training samples from high-VIX regimes (COVID, 2022 rate
    shock, 2023 SVB stress) without hardcoding dates. Citadel-standard approach:
    use a contemporaneous, causal regime indicator rather than hindsight dates.
    """
    try:
        ca_path = os.path.join(data_dir, "cross_asset.csv")
        if os.path.exists(ca_path):
            _ca = pd.read_csv(ca_path, index_col=0, parse_dates=True)
            if "^VIX" in _ca.columns:
                return _ca["^VIX"]
    except Exception:
        pass
    return None


def _compute_regime_sample_weights(
    train_dates_idx: pd.Index,
    vix_series: Optional[pd.Series],
    vix_ref: float = 20.0,
    vix_sat: float = 35.0,
    floor_weight: float = 0.30,
) -> Optional[np.ndarray]:
    """
    Compute per-sample regime weights from contemporaneous VIX level.

    Mapping (linear, clipped):
      VIX <= vix_ref (20)     → weight 1.0
      VIX == vix_sat (35)     → weight floor_weight (0.30)
      VIX >= vix_sat          → weight floor_weight (0.30)
      VIX between             → linear interpolation

    This auto-catches:
      - COVID March 2020 (VIX 80+)  → 0.30
      - 2022 rate shock (VIX 30+)    → ~0.40
      - 2023 SVB stress (VIX 26+)    → ~0.65
      - Normal conditions (VIX 15)  → 1.00

    Returns None if vix_series is unavailable — caller falls back to uniform.
    """
    if vix_series is None:
        return None
    vix_at_dates = vix_series.reindex(train_dates_idx)
    # Linear decay from vix_ref to vix_sat
    span = max(vix_sat - vix_ref, 1e-6)
    decay = (vix_at_dates - vix_ref) / span
    regime_weight = 1.0 - decay * (1.0 - floor_weight)
    regime_weight = regime_weight.clip(floor_weight, 1.0)
    # Missing VIX dates → full weight (neutral assumption)
    regime_weight = regime_weight.fillna(1.0)
    return regime_weight.values


def walk_forward_train(
    X: pd.DataFrame, y: pd.Series, cfg: ModelConfig, feature_cfg: FeatureConfig,
    model_type: str = "lightgbm", model_cfg=None, max_features: int = 0,
) -> Tuple[list, pd.Series, pd.DataFrame]:
    """
    Walk-forward training with purge/embargo gap.

    Args:
        model_type: "lightgbm", "tst", or "crossmamba"
        model_cfg: Config object for the chosen model type (uses cfg for lightgbm)
        max_features: if > 0, run per-window feature selection on training data only
                      (eliminates look-ahead bias from global feature selection)
    """
    if isinstance(X.index, pd.MultiIndex):
        dates = sorted(X.index.get_level_values(0).unique())
    else:
        dates = sorted(X.index.unique())

    train_window = cfg.train_window_days
    retrain_every = cfg.retrain_every_days
    purge = cfg.purge_gap_days
    embargo = cfg.embargo_days

    # Use the appropriate config for non-LightGBM models
    effective_cfg = model_cfg if model_cfg is not None else cfg

    # VIX regime series for sample weighting (causal, contemporaneous)
    vix_regime_series = _load_vix_regime_series()
    if vix_regime_series is not None:
        logger.info(f"[{model_type.upper()}] VIX regime weighting active "
                    f"({len(vix_regime_series)} dates loaded)")

    models = []
    all_preds = []
    metrics_history = []

    # Pre-compute date-to-index mapping for O(1) lookups instead of O(n) isin()
    if isinstance(X.index, pd.MultiIndex):
        date_level = X.index.get_level_values(0)
        date_to_mask = {}
        for d in dates:
            date_to_mask[d] = date_level == d
    else:
        date_to_mask = None

    def _select_by_dates(df, date_list):
        """Fast date-based selection using pre-computed masks."""
        if date_to_mask is not None:
            mask = np.zeros(len(df), dtype=bool)
            for d in date_list:
                if d in date_to_mask:
                    mask |= date_to_mask[d]
            return df.iloc[mask]
        return df.loc[date_list]

    # Walk-forward window layout (Lopez de Prado):
    #   train:      [i, train_end)                    — training data, no overlap with future
    #   purge gap:  [train_end, val_start)            — empty, prevents label leakage (size: purge)
    #   val:        [val_start, val_end)              — for early stopping (size: val_size)
    #   embargo:    [val_end, pred_start)             — prevents autocorrelation leakage
    #   pred:       [pred_start, pred_end)            — out-of-sample predictions
    val_size = max(embargo, 5)  # at least 5 days of val for early stopping

    # Phase 1: build window plans (cheap, sequential)
    window_plans = []
    for i in range(0, len(dates) - train_window, retrain_every):
        train_end = i + train_window - purge - val_size - embargo
        val_start = train_end + purge
        val_end = val_start + val_size
        pred_start = val_end + embargo
        pred_end = min(pred_start + retrain_every, len(dates))

        if train_end < 0 or val_start >= len(dates) or pred_start >= len(dates):
            continue

        train_dates = dates[i:train_end]
        val_dates = dates[val_start:val_end]
        pred_dates = dates[pred_start:pred_end]

        if len(train_dates) < 100 or not pred_dates:
            continue

        window_plans.append({
            "window_num": len(window_plans) + 1,
            "train_dates": train_dates,
            "val_dates": val_dates,
            "pred_dates": pred_dates,
            "pred_start": pred_start,
            "pred_end": pred_end,
        })

    # Phase 2: per-window worker function
    def _train_one_window(plan):
        X_tr = _select_by_dates(X, plan["train_dates"])
        y_tr = y.loc[X_tr.index]
        X_v = _select_by_dates(X, plan["val_dates"])
        y_v = y.loc[X_v.index] if len(X_v) > 0 else pd.Series(dtype=float)
        X_p = _select_by_dates(X, plan["pred_dates"])

        if len(X_tr) < 100 or len(X_p) == 0:
            return None

        # Per-window feature selection (training data only — no look-ahead)
        # n_splits controls regime robustness: more splits = features must be
        # IC-consistent across more sub-periods → naturally filters value traps
        # that work in one regime but fail in another.
        window_features = None
        n_splits = getattr(feature_cfg, 'feature_selection_n_splits', 2)
        if max_features > 0:
            from backtest import select_features_by_ic
            window_features = select_features_by_ic(
                X_tr, y_tr, max_features=max_features, n_splits=n_splits,
            )
            X_tr = X_tr[window_features]
            if len(X_v) > 0:
                X_v = X_v[window_features]
            X_p = X_p[window_features]

        logger.info(
            f"[{model_type.upper()}] Window {plan['window_num']}: "
            f"Train {plan['train_dates'][0].date()}→{plan['train_dates'][-1].date()} ({len(X_tr)}), "
            f"Predict {plan['pred_dates'][0].date()}→{plan['pred_dates'][-1].date()}"
            f"{f' ({len(window_features)} feats)' if window_features else ''}"
        )

        model = create_model(model_type, effective_cfg)

        # Sample weights: temporal decay × contemporaneous VIX regime weight.
        # The VIX component is Citadel-standard regime weighting — it down-weights
        # training samples from high-VIX periods (COVID, 2022 rate shock, 2023 SVB)
        # using ONLY data available at each training sample's own date. No hindsight,
        # no hardcoded "COVID was weird" flags. Linear mapping:
        #   VIX≤20 → 1.0,  VIX=35 → 0.3,  VIX>35 → 0.3 floor
        # COVID March 2020 (VIX 80) → 0.3, while VIX=15 days → 1.0.
        sample_weight = None
        if len(X_tr) > 100:
            n = len(X_tr)
            sample_weight = np.exp(np.linspace(-1, 0, n))  # oldest=0.37, newest=1.0
            sample_weight = sample_weight * n / sample_weight.sum()  # normalize to mean=1

            if isinstance(X_tr.index, pd.MultiIndex):
                train_dates_idx = X_tr.index.get_level_values(0)
            else:
                train_dates_idx = X_tr.index

            regime_w = _compute_regime_sample_weights(
                train_dates_idx, vix_regime_series,
                vix_ref=20.0, vix_sat=35.0, floor_weight=0.30,
            )
            if regime_w is not None:
                sample_weight = sample_weight * regime_w
                sample_weight = sample_weight * n / sample_weight.sum()  # renormalize
                # Diagnostic (debug-level): how many samples are heavily down-weighted
                n_downweighted = int((regime_w < 0.5).sum())
                if n_downweighted > 0:
                    logger.debug(
                        f"  VIX regime weighting: {n_downweighted}/{n} samples at "
                        f"<0.5× (avg regime_w={regime_w.mean():.3f})"
                    )

        # Only LightGBM supports sample_weight — TST/CrossMamba don't
        train_kwargs = {}
        if model_type == "lightgbm" and sample_weight is not None:
            train_kwargs["sample_weight"] = sample_weight

        metrics = model.train(
            X_tr, y_tr,
            X_v if len(X_v) > 0 else None,
            y_v if len(y_v) > 0 else None,
            **train_kwargs,
        )
        metrics["window"] = plan["window_num"]
        # Capture diagnostic context (used by post-run analyzer)
        metrics["train_start"] = str(plan["train_dates"][0].date()) if len(plan["train_dates"]) > 0 else None
        metrics["train_end"] = str(plan["train_dates"][-1].date()) if len(plan["train_dates"]) > 0 else None
        metrics["predict_start"] = str(plan["pred_dates"][0].date()) if len(plan["pred_dates"]) > 0 else None
        metrics["predict_end"] = str(plan["pred_dates"][-1].date()) if len(plan["pred_dates"]) > 0 else None
        metrics["n_train_samples"] = int(len(X_tr))
        metrics["n_features_post_select"] = int(len(window_features)) if window_features else int(X_tr.shape[1])
        # Sample weight stats (if applied)
        if sample_weight is not None:
            metrics["mean_sample_weight"] = float(np.mean(sample_weight))
            metrics["pct_downweighted_below_05"] = float((np.asarray(sample_weight) < 0.5).mean())
        # Top selected features (for stability analysis)
        if window_features:
            metrics["window_features"] = list(window_features[:20])  # top 20 names

        # Predictions
        preds = None
        if len(X_p) > 0:
            if model_type in ("tst", "crossmamba"):
                # Sequence models need lookback context to build input sequences
                seq_len = getattr(effective_cfg, "sequence_length", 21)
                context_start = max(0, plan["pred_start"] - seq_len)
                context_dates = dates[context_start:plan["pred_end"]]
                X_p_ctx = _select_by_dates(X, context_dates)
                if window_features is not None:
                    X_p_ctx = X_p_ctx[window_features]
                preds = model.predict(X_p_ctx)
                if isinstance(preds.index, pd.MultiIndex):
                    mask = preds.index.get_level_values(0).isin(plan["pred_dates"])
                    preds = preds[mask]
                else:
                    preds = preds[preds.index.isin(plan["pred_dates"])]
            else:
                preds = model.predict(X_p)

            # Compute OOS IC for non-LightGBM (LightGBM does this in .train via val set)
            if model_type != "lightgbm" and preds is not None and len(preds) > 10:
                try:
                    y_p = y.loc[preds.index]
                    common = preds.index.intersection(y_p.index)
                    if len(common) > 10:
                        ic = float(np.corrcoef(preds.loc[common].values, y_p.loc[common].values)[0, 1])
                        rank_ic = float(pd.Series(preds.loc[common].values).corr(
                            pd.Series(y_p.loc[common].values), method="spearman"
                        ))
                        metrics["val_ic"] = ic
                        metrics["val_rank_ic"] = rank_ic
                        logger.info(f"  IC: {ic:.4f}, Rank IC: {rank_ic:.4f}")
                except Exception:
                    pass

        return {
            "window_num": plan["window_num"],
            "model": model,
            "preds": preds,
            "metrics": metrics,
        }

    # Phase 3: run windows in parallel (LightGBM only — neural nets have their own threading)
    parallel_windows = getattr(cfg, "parallel_windows", 1)
    if parallel_windows > 1 and model_type == "lightgbm" and len(window_plans) > 1:
        from joblib import Parallel, delayed
        logger.info(
            f"[{model_type.upper()}] Walk-forward parallelism: {parallel_windows} windows concurrently "
            f"(lightgbm n_jobs={getattr(cfg, 'lightgbm_n_jobs', 2)}), {len(window_plans)} total"
        )
        # Threading backend: shares memory (no X pickling), LightGBM releases GIL during fit.
        # Prefer='threads' forces true threading regardless of joblib defaults.
        results = Parallel(n_jobs=parallel_windows, backend="threading", prefer="threads")(
            delayed(_train_one_window)(plan) for plan in window_plans
        )
    else:
        results = [_train_one_window(plan) for plan in window_plans]

    # Phase 4: collect results in window order
    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r["window_num"])

    for r in results:
        models.append(r["model"])
        metrics_history.append(r["metrics"])
        if r["preds"] is not None and len(r["preds"]) > 0:
            all_preds.append(r["preds"])

    predictions = pd.concat(all_preds) if all_preds else pd.Series(dtype=float)
    metrics_df = pd.DataFrame(metrics_history)

    logger.info(f"[{model_type.upper()}] Walk-forward: {len(models)} windows, {len(predictions)} predictions")
    if "val_rank_ic" in metrics_df.columns:
        avg_ic = metrics_df["val_rank_ic"].mean()
        std_ic = metrics_df["val_rank_ic"].std()
        logger.info(f"[{model_type.upper()}] Avg Rank IC: {avg_ic:.4f} ± {std_ic:.4f}")
        ir = avg_ic / (std_ic + 1e-8)
        logger.info(f"[{model_type.upper()}] IC Information Ratio: {ir:.2f}")

    return models, predictions, metrics_df
