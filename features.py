"""
Integrated feature engine: price/volume, fundamentals, cross-asset, earnings.

Features are organized in groups with MultiIndex columns: (group, feature, ticker)
Then stacked to ML format: (date, ticker) rows × feature columns.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from config import FeatureConfig

logger = logging.getLogger(__name__)


# ===== PRICE / VOLUME FEATURES =====

def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def momentum_features(prices: pd.DataFrame, cfg: FeatureConfig) -> Dict[str, pd.DataFrame]:
    feats = {}
    lr = _log_returns(prices)
    for w in cfg.momentum_windows:
        feats[f"mom_{w}d"] = lr.rolling(w).sum()
        if w > 5:
            feats[f"mom_{w}d_skip1"] = lr.shift(1).rolling(w - 1).sum()
    if 252 in cfg.momentum_windows and 21 in cfg.momentum_windows:
        feats["mom_12m_1m"] = lr.shift(21).rolling(231).sum()
    # Momentum acceleration
    for w in [21, 63]:
        if f"mom_{w}d" in feats:
            feats[f"mom_accel_{w}d"] = feats[f"mom_{w}d"] - feats[f"mom_{w}d"].shift(w)
    return feats


def mean_reversion_features(prices: pd.DataFrame, cfg: FeatureConfig) -> Dict[str, pd.DataFrame]:
    feats = {}
    lr = _log_returns(prices)
    for w in cfg.mean_reversion_windows:
        ma = prices.rolling(w).mean()
        feats[f"price_to_ma_{w}d"] = (prices / ma) - 1
        roll_mean = lr.rolling(w).mean()
        roll_std = lr.rolling(w).std()
        feats[f"ret_zscore_{w}d"] = (lr - roll_mean) / (roll_std + 1e-8)
    for w in [21, 63, 252]:
        feats[f"dist_from_{w}d_high"] = prices / prices.rolling(w).max() - 1
        feats[f"dist_from_{w}d_low"] = prices / prices.rolling(w).min() - 1
    return feats


def volatility_features(prices: pd.DataFrame, cfg: FeatureConfig) -> Dict[str, pd.DataFrame]:
    feats = {}
    lr = _log_returns(prices)
    for w in cfg.volatility_windows:
        feats[f"vol_{w}d"] = lr.rolling(w).std() * np.sqrt(252)
    if 5 in cfg.volatility_windows and 63 in cfg.volatility_windows:
        feats["vol_ratio_5_63"] = (lr.rolling(5).std()) / (lr.rolling(63).std() + 1e-8)
    for w in [21, 63]:
        feats[f"skew_{w}d"] = lr.rolling(w).skew()
        feats[f"kurt_{w}d"] = lr.rolling(w).kurt()
        up = lr.clip(lower=0).rolling(w).std()
        dn = lr.clip(upper=0).rolling(w).std()
        feats[f"updown_vol_{w}d"] = up / (dn + 1e-8)
    return feats


def volume_features(
    prices: pd.DataFrame, volumes: pd.DataFrame, cfg: FeatureConfig,
) -> Dict[str, pd.DataFrame]:
    feats = {}
    dollar_vol = prices * volumes
    lr = _log_returns(prices)
    for w in cfg.volume_windows:
        avg_vol = volumes.rolling(w).mean()
        feats[f"vol_surprise_{w}d"] = volumes / (avg_vol + 1) - 1
        feats[f"log_dollar_vol_{w}d"] = np.log1p(dollar_vol.rolling(w).mean())
    for w in [21, 63]:
        amihud = (lr.abs() / (dollar_vol + 1)).rolling(w).mean()
        feats[f"amihud_{w}d"] = np.log1p(amihud * 1e6)
    feats["vol_trend"] = volumes.rolling(5).mean() / (volumes.rolling(21).mean() + 1) - 1
    return feats


def technical_features(prices: pd.DataFrame, cfg: FeatureConfig) -> Dict[str, pd.DataFrame]:
    feats = {}
    close = prices
    for w in cfg.rsi_windows:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        feats[f"rsi_{w}"] = 100 - 100 / (1 + gain / (loss + 1e-8))
    ma = close.rolling(cfg.bb_window).mean()
    std = close.rolling(cfg.bb_window).std()
    feats["bb_position"] = (close - (ma - cfg.bb_std * std)) / (2 * cfg.bb_std * std + 1e-8)
    feats["bb_width"] = (2 * cfg.bb_std * std) / (ma + 1e-8)
    ema_f = close.ewm(span=cfg.macd_fast).mean()
    ema_s = close.ewm(span=cfg.macd_slow).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=cfg.macd_signal).mean()
    feats["macd_hist"] = (macd - sig) / (close + 1e-8)
    return feats


# ===== CROSS-SECTIONAL RANKING =====

def cross_sectional_ranks(
    feature_dict: Dict[str, pd.DataFrame],
    key_features: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Rank every feature cross-sectionally [0, 1]."""
    if key_features is None:
        key_features = list(feature_dict.keys())
    cs = {}
    for name in key_features:
        if name in feature_dict:
            cs[f"cs_{name}"] = feature_dict[name].rank(axis=1, pct=True)
    return cs


# ===== TARGET =====

def compute_targets(
    prices: pd.DataFrame,
    cfg: FeatureConfig,
    sector_map: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute prediction targets with institutional-grade construction.

    Three target types:
    1. Raw forward return rank (original)
    2. Risk-adjusted forward return (return / trailing vol) — Grinold & Kahn
    3. Industry-relative forward return (stock return - sector avg) — removes sector noise
    """
    lr = _log_returns(prices)
    targets = {}

    for h in cfg.target_horizons:
        fwd = lr.shift(-h).rolling(h).sum()
        targets[f"fwd_ret_{h}d"] = fwd
        targets[f"fwd_rank_{h}d"] = fwd.rank(axis=1, pct=True)

        # Risk-adjusted target: forward return / trailing volatility
        # Grinold & Kahn: predicting risk-adjusted returns is more stable
        trailing_vol = lr.rolling(63).std() * np.sqrt(252)
        trailing_vol = trailing_vol.replace(0, np.nan).fillna(method="ffill")
        risk_adj = fwd / (trailing_vol + 1e-8)
        targets[f"fwd_risk_adj_{h}d"] = risk_adj.rank(axis=1, pct=True)

        # Industry-relative target: stock return minus sector average
        # Removes sector rotation noise — isolates stock-specific alpha
        if sector_map:
            sector_df = pd.Series(sector_map)
            ind_rel = fwd.copy()
            for sector in sector_df.unique():
                sector_stocks = sector_df[sector_df == sector].index
                sector_cols = [c for c in fwd.columns if c in sector_stocks]
                if len(sector_cols) > 1:
                    sector_avg = fwd[sector_cols].mean(axis=1)
                    for col in sector_cols:
                        ind_rel[col] = fwd[col] - sector_avg
            targets[f"fwd_ind_rel_{h}d"] = ind_rel.rank(axis=1, pct=True)

    return targets


# ===== MAIN BUILD =====

def build_all_features(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    cfg: FeatureConfig,
    fundamental_feats: Optional[Dict] = None,
    cross_asset_feats: Optional[Dict] = None,
    insider_feats: Optional[Dict] = None,
    fmp_feats: Optional[Dict] = None,
    openbb_feats: Optional[Dict] = None,
    sector_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Build complete feature panel combining all signal sources.
    All inputs are dicts of tuple_key → DataFrame(dates × tickers).
    Returns: (feature_panel, targets_dict)
    """
    logger.info("Building features...")

    # Price/volume features → dict of DataFrames
    pv_feats = {}
    pv_feats.update(momentum_features(prices, cfg))
    pv_feats.update(mean_reversion_features(prices, cfg))
    pv_feats.update(volatility_features(prices, cfg))
    pv_feats.update(volume_features(prices, volumes, cfg))
    pv_feats.update(technical_features(prices, cfg))

    # Cross-sectional ranks of price/volume features
    cs_feats = cross_sectional_ranks(pv_feats)

    # Build master dict: tuple_key → DataFrame(dates × tickers)
    all_feats = {}
    for name, df in pv_feats.items():
        all_feats[("pv", name)] = df
    for name, df in cs_feats.items():
        all_feats[("pv", name)] = df

    logger.info(f"  Price/volume: {len(all_feats)} features")

    # Add fundamental features (already keyed as tuples)
    if fundamental_feats:
        all_feats.update(fundamental_feats)
        logger.info(f"  + Fundamental: {len(fundamental_feats)} features")

    # Add cross-asset features
    if cross_asset_feats:
        all_feats.update(cross_asset_feats)
        logger.info(f"  + Cross-asset: {len(cross_asset_feats)} features")

    # Add fractionally differentiated price features (López de Prado Ch. 5)
    # These are stationary while preserving memory — better than raw returns
    try:
        from advanced_labeling import add_frac_diff_features
        # Apply to a subset of stocks (top 20 by volume to save computation)
        top_tickers = volumes.mean().nlargest(20).index.tolist()
        frac_prices = prices[top_tickers] if top_tickers else prices.iloc[:, :20]
        fd_feats = add_frac_diff_features(frac_prices, d=0.4)
        if not fd_feats.empty:
            for col in fd_feats.columns:
                all_feats[("pv", col)] = fd_feats[[col]].reindex(prices.index)
            logger.info(f"  + Fractional diff: {len(fd_feats.columns)} features")
    except Exception as e:
        logger.debug(f"Fractional differentiation skipped: {e}")

    # Add insider features (SEC Form 4 data)
    if insider_feats:
        all_feats.update(insider_feats)
        logger.info(f"  + Insider: {len(insider_feats)} features")

    # Add FMP fundamental features (point-in-time, no look-ahead)
    if fmp_feats:
        all_feats.update(fmp_feats)
        logger.info(f"  + FMP fundamentals: {len(fmp_feats)} features")

    # Add OpenBB alternative data (options + short interest)
    if openbb_feats:
        all_feats.update(openbb_feats)
        logger.info(f"  + OpenBB: {len(openbb_feats)} features")

    # Add interaction features (institutional cross-factor combinations)
    # Must come AFTER all base features are computed
    try:
        from interaction_features import build_interaction_features
        # Separate sentiment features if mixed into cross_asset_feats
        sent_feats_dict = {k: v for k, v in (cross_asset_feats or {}).items()
                          if isinstance(k, tuple) and k[0] == "sent"}
        interact_feats = build_interaction_features(
            pv_feats, fundamental_feats, cross_asset_feats, sent_feats_dict,
        )
        if interact_feats:
            all_feats.update(interact_feats)
            logger.info(f"  + Interactions: {len(interact_feats)} features")
    except Exception as e:
        logger.debug(f"Interaction features skipped: {e}")

    # Combine into single DataFrame with MultiIndex columns
    panel = pd.concat(all_feats, axis=1)
    targets = compute_targets(prices, cfg, sector_map=sector_map)

    logger.info(f"Total panel: {panel.shape} ({len(all_feats)} feature groups)")
    return panel, targets


def panel_to_ml_format(
    features: pd.DataFrame,
    target: pd.DataFrame,
    dropna: bool = True,
    max_nan_pct: float = 0.3,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Convert (dates × MultiIndex(group, feat, ticker)) to (date-ticker rows × feature cols).

    Instead of dropping all rows with any NaN (which loses most data),
    we:
    1. Drop features (columns) with >max_nan_pct missing values
    2. Fill remaining NaN with cross-sectional median (per date)
    3. Drop rows where the target is NaN
    """
    if isinstance(features.columns, pd.MultiIndex) and features.columns.nlevels >= 2:
        feat_stacked = features.stack(level=-1)
        feat_stacked.index.names = ["date", "ticker"]
        if isinstance(feat_stacked.columns, pd.MultiIndex):
            feat_stacked.columns = [
                "_".join(str(c) for c in col) for col in feat_stacked.columns
            ]
    else:
        feat_stacked = features.copy()

    tgt_stacked = target.stack()
    tgt_stacked.name = "target"
    tgt_stacked.index.names = ["date", "ticker"]

    common_idx = feat_stacked.index.intersection(tgt_stacked.index)
    X = feat_stacked.loc[common_idx]
    y = tgt_stacked.loc[common_idx]

    # Drop target NaN
    target_valid = y.notna()
    X = X.loc[target_valid]
    y = y.loc[target_valid]

    # Drop feature columns with too many NaN
    nan_pct = X.isna().mean()
    good_cols = nan_pct[nan_pct <= max_nan_pct].index
    dropped = len(X.columns) - len(good_cols)
    if dropped > 0:
        logger.info(f"  Dropped {dropped} features with >{max_nan_pct:.0%} NaN")
    X = X[good_cols]

    # Fill remaining NaN with cross-sectional median per date
    # (better than 0 for ranked features where 0 is far from the 0.5 median)
    if isinstance(X.index, pd.MultiIndex):
        dates_level = X.index.get_level_values(0)
        for col in X.columns:
            if X[col].isna().any():
                medians = X.groupby(dates_level)[col].transform("median")
                X[col] = X[col].fillna(medians)
    X = X.fillna(0)  # Final fallback for any remaining

    logger.info(f"ML dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y
