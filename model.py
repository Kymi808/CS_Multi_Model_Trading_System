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
            "n_jobs": -1,
            "verbose": -1,
        }
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

        # Compute group sizes for LambdaRank (samples per date for cross-sectional ranking)
        if isinstance(X_train.index, pd.MultiIndex):
            date_counts = X_train.index.get_level_values(0).value_counts().sort_index()
            train_groups = date_counts.values.tolist()
        else:
            train_groups = [len(X_train)]  # single group fallback

        if X_val is not None and isinstance(X_val.index, pd.MultiIndex):
            val_date_counts = X_val.index.get_level_values(0).value_counts().sort_index()
            val_groups = val_date_counts.values.tolist()
        else:
            val_groups = [len(X_val)] if X_val is not None else None

        for i, seed in enumerate(self.cfg.ensemble_seeds[:self.cfg.n_ensemble]):
            params = self._get_lgb_params(seed)
            params["objective"] = "lambdarank"
            params["metric"] = "ndcg"
            params["ndcg_eval_at"] = [10, 20]  # optimize ranking quality at top/bottom 10-20

            model = lgb.LGBMRanker(**params)
            callbacks = [lgb.log_evaluation(period=0)]

            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight

            if X_val is not None and y_val is not None:
                callbacks.append(lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False))
                model.fit(X_train, y_train, group=train_groups,
                          eval_set=[(X_val, y_val)], eval_group=[val_groups],
                          eval_metric="ndcg", eval_at=[10, 20],
                          callbacks=callbacks, **fit_kwargs)
            else:
                model.fit(X_train, y_train, group=train_groups,
                          callbacks=callbacks, **fit_kwargs)

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


def walk_forward_train(
    X: pd.DataFrame, y: pd.Series, cfg: ModelConfig, feature_cfg: FeatureConfig,
    model_type: str = "lightgbm", model_cfg=None,
) -> Tuple[list, pd.Series, pd.DataFrame]:
    """
    Walk-forward training with purge/embargo gap.

    Args:
        model_type: "lightgbm", "tst", or "crossmamba"
        model_cfg: Config object for the chosen model type (uses cfg for lightgbm)
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

    for i in range(0, len(dates) - train_window, retrain_every):
        train_end = i + train_window - purge
        val_start = i + train_window - purge + embargo
        val_end = i + train_window
        pred_start = i + train_window
        pred_end = min(i + train_window + retrain_every, len(dates))

        if train_end < 0 or val_start >= len(dates) or pred_start >= len(dates):
            continue

        train_dates = dates[i:train_end]
        val_dates = dates[val_start:val_end]
        pred_dates = dates[pred_start:pred_end]

        if len(train_dates) < 100 or not pred_dates:
            continue

        X_tr = _select_by_dates(X, train_dates)
        y_tr = y.loc[X_tr.index]
        X_v = _select_by_dates(X, val_dates)
        y_v = y.loc[X_v.index] if len(X_v) > 0 else pd.Series(dtype=float)
        X_p = _select_by_dates(X, pred_dates)

        if len(X_tr) < 100 or len(X_p) == 0:
            continue

        window_num = len(models) + 1
        logger.info(
            f"[{model_type.upper()}] Window {window_num}: "
            f"Train {train_dates[0].date()}→{train_dates[-1].date()} ({len(X_tr)}), "
            f"Predict {pred_dates[0].date()}→{pred_dates[-1].date()}"
        )

        model = create_model(model_type, effective_cfg)

        # Sample weights: simple temporal decay (fast, no expensive computation)
        sample_weight = None
        if len(X_tr) > 100:
            # More recent samples get higher weight (exponential decay)
            n = len(X_tr)
            sample_weight = np.exp(np.linspace(-1, 0, n))  # oldest=0.37, newest=1.0
            sample_weight = sample_weight * n / sample_weight.sum()  # normalize to mean=1

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
        metrics["window"] = window_num
        metrics_history.append(metrics)

        if len(X_p) > 0:
            # For sequence models (TST, CrossMamba), include lookback context
            # so they can build sequences for the first prediction dates
            if model_type in ("tst", "crossmamba"):
                seq_len = getattr(effective_cfg, "sequence_length", 21)
                context_start = max(0, pred_start - seq_len)
                context_dates = dates[context_start:pred_end]
                X_p_ctx = _select_by_dates(X, context_dates)
                preds = model.predict(X_p_ctx)
                # Only keep predictions for actual pred_dates
                if isinstance(preds.index, pd.MultiIndex):
                    mask = preds.index.get_level_values(0).isin(pred_dates)
                    preds = preds[mask]
                else:
                    preds = preds[preds.index.isin(pred_dates)]
                if len(preds) > 0:
                    all_preds.append(preds)
            else:
                preds = model.predict(X_p)
                all_preds.append(preds)

            # Compute IC for ALL model types (not just LightGBM)
            # Uses out-of-sample predictions vs actual target
            if model_type != "lightgbm" and len(preds) > 10:
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

        models.append(model)

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
