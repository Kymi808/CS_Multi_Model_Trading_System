"""
Optuna hyperparameter optimization for LightGBM.

Objective: maximize out-of-sample Rank IC across walk-forward windows.
"""
import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    n_cv_windows: int = 5,
    train_window: int = 504,
    seed: int = 42,
) -> Dict:
    if isinstance(X.index, pd.MultiIndex):
        dates = sorted(X.index.get_level_values(0).unique())
    else:
        dates = sorted(X.index.unique())

    n_dates = len(dates)

    # Auto-size: use at most 2/3 of dates for training
    effective_window = min(train_window, n_dates * 2 // 3)
    if effective_window < 50:
        logger.warning(f"Too few dates ({n_dates}) for optimization. Skipping.")
        return {}

    cv_splits = _make_cv_splits(dates, n_cv_windows, effective_window)
    if not cv_splits:
        effective_window = max(50, n_dates // 3)
        cv_splits = _make_cv_splits(dates, n_cv_windows, effective_window)
    if not cv_splits:
        logger.warning("No valid CV splits. Skipping optimization.")
        return {}

    logger.info(
        f"Optuna: {n_trials} trials, {len(cv_splits)} CV windows, "
        f"{n_dates} dates, train_window={effective_window}"
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96, step=4),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, step=10),
            "subsample": trial.suggest_float("subsample", 0.5, 0.95, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9, step=0.05),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.05),
        }

        rank_ics = []
        for train_dates, val_dates in cv_splits:
            try:
                if isinstance(X.index, pd.MultiIndex):
                    X_tr = X.loc[X.index.get_level_values(0).isin(train_dates)]
                    y_tr = y.loc[y.index.isin(X_tr.index)]
                    X_va = X.loc[X.index.get_level_values(0).isin(val_dates)]
                    y_va = y.loc[y.index.isin(X_va.index)]
                else:
                    X_tr = X.loc[train_dates]; y_tr = y.loc[train_dates]
                    X_va = X.loc[val_dates]; y_va = y.loc[val_dates]

                if len(X_tr) < 100 or len(X_va) < 50:
                    continue

                X_tr_c = X_tr.replace([np.inf, -np.inf], np.nan)
                X_va_c = X_va.replace([np.inf, -np.inf], np.nan)

                model = lgb.LGBMRegressor(
                    **params, random_state=seed, n_jobs=-1, verbose=-1,
                )
                model.fit(
                    X_tr_c, y_tr,
                    eval_set=[(X_va_c, y_va)],
                    eval_metric="l2",
                    callbacks=[
                        lgb.early_stopping(30, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
                preds = model.predict(X_va_c)
                rank_ic = pd.Series(preds).corr(pd.Series(y_va.values), method="spearman")
                if not np.isnan(rank_ic):
                    rank_ics.append(rank_ic)
            except Exception:
                continue

        if not rank_ics:
            return -1.0

        mean_ic = np.mean(rank_ics)
        std_ic = np.std(rank_ics) if len(rank_ics) > 1 else 0
        score = mean_ic - 0.5 * std_ic
        trial.set_user_attr("mean_ic", float(mean_ic))
        trial.set_user_attr("std_ic", float(std_ic))
        trial.set_user_attr("ir", float(mean_ic / (std_ic + 1e-8)))
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    logger.info(f"Optuna complete: {n_trials} trials, best score: {best.value:.4f}")
    mean_ic = best.user_attrs.get("mean_ic")
    ir = best.user_attrs.get("ir")
    if isinstance(mean_ic, (int, float)):
        logger.info(f"  Best IC: {mean_ic:.4f}")
    if isinstance(ir, (int, float)):
        logger.info(f"  Best IC IR: {ir:.2f}")
    logger.info(f"  Best params: {best.params}")

    _print_summary(study)

    # Save best params to disk for production retraining
    import json, os
    params_path = os.path.join("results", "optuna_best_params.json")
    os.makedirs("results", exist_ok=True)
    with open(params_path, "w") as f:
        json.dump(best.params, f, indent=2)
    logger.info(f"Saved best params to {params_path}")

    return best.params


def _make_cv_splits(dates, n_windows, train_window, purge=10, embargo=5):
    total = len(dates)
    if total < train_window + 30:
        train_window = max(50, total * 2 // 3)
    remaining = total - train_window
    if remaining < 20:
        return []
    val_size = max(15, remaining // (n_windows + 1))
    splits = []
    for i in range(n_windows):
        vs = train_window + i * val_size
        ve = min(vs + val_size, total)
        te = vs - purge
        if te < 30 or vs + embargo >= total:
            continue
        ts = max(0, te - train_window)
        t_d = dates[ts:te]
        v_d = dates[vs + embargo:ve]
        if len(t_d) >= 30 and len(v_d) >= 5:
            splits.append((t_d, v_d))
    return splits


def apply_optuna_params(cfg, params: Dict):
    for k in ["n_estimators", "max_depth", "learning_rate", "num_leaves",
              "min_child_samples", "subsample", "colsample_bytree",
              "reg_alpha", "reg_lambda", "min_split_gain"]:
        if k in params:
            setattr(cfg, k, params[k])
    return cfg


def _print_summary(study):
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)
    print(f"\n{'='*60}")
    print("OPTUNA OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Total trials: {len(trials)}")
    if trials and trials[0].value is not None:
        print(f"Best score:   {trials[0].value:.4f}")
    print(f"\nTop 5 trials:")
    for t in trials[:5]:
        ic = t.user_attrs.get("mean_ic")
        ir = t.user_attrs.get("ir")
        val = t.value if t.value is not None else float("nan")
        ic_s = f"{ic:.4f}" if isinstance(ic, (int, float)) else "N/A"
        ir_s = f"{ir:.2f}" if isinstance(ir, (int, float)) else "N/A"
        print(f"  #{t.number:>3d}: score={val:.4f}, IC={ic_s}, IR={ir_s}")
    if trials:
        print(f"\nBest hyperparameters:")
        for k, v in trials[0].params.items():
            print(f"  {k:>25s}: {v:.6f}" if isinstance(v, float) else f"  {k:>25s}: {v}")
    print(f"{'='*60}")
