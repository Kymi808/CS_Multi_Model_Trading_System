"""
Institutional-grade feature interactions for cross-sectional equity ranking.

Manually engineered interaction terms proven in academic research and
used by top quantitative firms. Each interaction captures a well-documented
anomaly that individual features alone miss.

All interactions use cross-sectional RANKS (0-1) to avoid scale issues
and outlier sensitivity. Products of two ranks produce a 0-1 signal
that is highest when BOTH factors are strong.

References:
- Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere"
- Novy-Marx (2013), "The Other Side of Value"
- Loh (2010), "Investor Inattention and the Underreaction to Stock Recommendations"
- Frazzini & Pedersen (2014), "Betting Against Beta"
- Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers"
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _rank_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank (0 to 1) per date."""
    return df.rank(axis=1, pct=True)


def _lookup(feats: dict, *name_parts: str) -> Optional[pd.DataFrame]:
    """
    Find a feature by partial name match across different key conventions.

    Handles both:
    - String keys: "mom_21d" (pv_feats)
    - Tuple keys: ("fund", "value_composite") (fundamental/sentiment/cross-asset)
    """
    for key, df in feats.items():
        key_str = str(key).lower() if isinstance(key, tuple) else key.lower()
        if all(part.lower() in key_str for part in name_parts):
            return df
    return None


def build_interaction_features(
    pv_feats: Dict[str, pd.DataFrame],
    fund_feats: Optional[Dict] = None,
    ca_feats: Optional[Dict] = None,
    sent_feats: Optional[Dict] = None,
) -> Dict[tuple, pd.DataFrame]:
    """
    Build institutional interaction features from base feature sets.

    Each interaction is the PRODUCT of two cross-sectional ranks.
    A stock scoring high on BOTH factors gets a high interaction score.

    Returns dict of ("interact", feature_name) -> DataFrame(dates x tickers)
    """
    fund_feats = fund_feats or {}
    ca_feats = ca_feats or {}
    sent_feats = sent_feats or {}

    # Merge all for unified lookup
    all_feats = {}
    all_feats.update(pv_feats)
    all_feats.update(fund_feats)
    all_feats.update(ca_feats)
    all_feats.update(sent_feats)

    interactions = {}

    # ── 1. Value × Momentum ─────────────────────────────────────────
    # Asness, Moskowitz & Pedersen (2013): "Value and Momentum Everywhere"
    # Cheap stocks with improving price momentum — the classic quant combo
    value = _lookup(all_feats, "value_composite") or _lookup(all_feats, "earnings_yield")
    momentum = _lookup(all_feats, "mom_63d")
    if value is not None and momentum is not None:
        val_rank = _rank_cs(value)
        mom_rank = _rank_cs(momentum)
        common = val_rank.columns.intersection(mom_rank.columns)
        if len(common) > 0:
            interactions[("interact", "value_x_momentum")] = (
                val_rank[common] * mom_rank[common]
            )

    # ── 2. Quality × Value ──────────────────────────────────────────
    # Novy-Marx (2013): "The Other Side of Value"
    # Profitable companies that are also cheap — quality at a discount
    quality = _lookup(all_feats, "quality_composite") or _lookup(all_feats, "rank_roe")
    if quality is not None and value is not None:
        qual_rank = _rank_cs(quality)
        val_rank = _rank_cs(value)
        common = qual_rank.columns.intersection(val_rank.columns)
        if len(common) > 0:
            interactions[("interact", "quality_x_value")] = (
                qual_rank[common] * val_rank[common]
            )

    # ── 3. Momentum × Low Volatility ────────────────────────────────
    # Frazzini & Pedersen (2014): "Betting Against Beta"
    # Strong momentum in low-vol names is more persistent (less noise)
    vol = _lookup(pv_feats, "vol_21d")
    if momentum is not None and vol is not None:
        mom_rank = _rank_cs(momentum)
        # INVERSE vol rank — low vol = high rank
        low_vol_rank = 1 - _rank_cs(vol)
        common = mom_rank.columns.intersection(low_vol_rank.columns)
        if len(common) > 0:
            interactions[("interact", "momentum_x_low_vol")] = (
                mom_rank[common] * low_vol_rank[common]
            )

    # ── 4. Size × Value ─────────────────────────────────────────────
    # Fama & French: small-cap value outperforms
    # Small + cheap = concentrated value factor exposure
    size = _lookup(all_feats, "log_mcap") or _lookup(all_feats, "rank_size")
    if size is not None and value is not None:
        # INVERSE size rank — small = high rank
        small_rank = 1 - _rank_cs(size)
        val_rank = _rank_cs(value)
        common = small_rank.columns.intersection(val_rank.columns)
        if len(common) > 0:
            interactions[("interact", "small_x_value")] = (
                small_rank[common] * val_rank[common]
            )

    # ── 5. Earnings Momentum × Price Momentum ───────────────────────
    # Fundamental confirmation of technical trend
    # When both price AND earnings are improving, the signal is stronger
    earn_mom = _lookup(all_feats, "earn_growth") or _lookup(all_feats, "earn_q_growth")
    if earn_mom is not None and momentum is not None:
        earn_rank = _rank_cs(earn_mom)
        mom_rank = _rank_cs(momentum)
        common = earn_rank.columns.intersection(mom_rank.columns)
        if len(common) > 0:
            interactions[("interact", "earnings_x_price_momentum")] = (
                earn_rank[common] * mom_rank[common]
            )

    # ── 6. Sentiment × Momentum ─────────────────────────────────────
    # News-confirmed trends are more reliable
    # Positive sentiment + upward momentum = strong conviction
    sentiment = _lookup(all_feats, "avg_sentiment") or _lookup(all_feats, "sentiment")
    if sentiment is not None and momentum is not None:
        sent_rank = _rank_cs(sentiment)
        mom_rank = _rank_cs(momentum)
        common = sent_rank.columns.intersection(mom_rank.columns)
        if len(common) > 0:
            interactions[("interact", "sentiment_x_momentum")] = (
                sent_rank[common] * mom_rank[common]
            )

    # ── 7. Credit Stress × Beta ─────────────────────────────────────
    # High-beta stocks are disproportionately affected in credit events
    # When credit tightens, short high-beta names
    beta = _lookup(all_feats, "beta")
    credit = _lookup(all_feats, "credit_spread")
    if beta is not None and credit is not None:
        beta_rank = _rank_cs(beta)
        # Broadcast credit (single column) across all stocks
        if credit.shape[1] == 1 or len(credit.columns) < len(beta.columns):
            credit_val = credit.iloc[:, 0] if credit.shape[1] >= 1 else credit
            credit_rank = pd.DataFrame(
                {col: _rank_single(credit_val) for col in beta_rank.columns},
                index=beta_rank.index,
            )
        else:
            credit_rank = _rank_cs(credit)
        common = beta_rank.columns.intersection(credit_rank.columns)
        if len(common) > 0:
            interactions[("interact", "credit_stress_x_beta")] = (
                credit_rank[common] * beta_rank[common]
            )

    # ── 8. Volume Surprise × Momentum ───────────────────────────────
    # Loh (2010): volume confirms trend — high volume moves are more informative
    vol_surprise = _lookup(pv_feats, "vol_surprise") or _lookup(pv_feats, "vol_trend")
    if vol_surprise is not None and momentum is not None:
        vs_rank = _rank_cs(vol_surprise)
        mom_rank = _rank_cs(momentum)
        common = vs_rank.columns.intersection(mom_rank.columns)
        if len(common) > 0:
            interactions[("interact", "volume_x_momentum")] = (
                vs_rank[common] * mom_rank[common]
            )

    # ── 9. Volatility × Mean Reversion ──────────────────────────────
    # High-vol stocks revert faster to fair value (overreaction)
    zscore = _lookup(pv_feats, "ret_zscore_21d") or _lookup(pv_feats, "zscore")
    if vol is not None and zscore is not None:
        vol_rank = _rank_cs(vol)
        # For mean reversion, we want EXTREME z-scores (both positive and negative)
        # Use absolute z-score rank
        extreme_rank = _rank_cs(zscore.abs())
        common = vol_rank.columns.intersection(extreme_rank.columns)
        if len(common) > 0:
            interactions[("interact", "vol_x_mean_reversion")] = (
                vol_rank[common] * extreme_rank[common]
            )

    # ── 10. Momentum Acceleration × Breadth ─────────────────────────
    # Accelerating momentum with broad market support = stronger signal
    mom_accel = _lookup(pv_feats, "mom_accel")
    breadth = _lookup(all_feats, "breadth")
    if mom_accel is not None:
        accel_rank = _rank_cs(mom_accel)
        if breadth is not None:
            # Broadcast breadth across stocks
            if breadth.shape[1] <= 1:
                breadth_vals = breadth.iloc[:, 0] if breadth.shape[1] == 1 else breadth
                breadth_rank = pd.DataFrame(
                    {col: _rank_single(breadth_vals) for col in accel_rank.columns},
                    index=accel_rank.index,
                )
            else:
                breadth_rank = _rank_cs(breadth)
            common = accel_rank.columns.intersection(breadth_rank.columns)
            if len(common) > 0:
                interactions[("interact", "accel_x_breadth")] = (
                    accel_rank[common] * breadth_rank[common]
                )

    # Cross-sectional rank all interactions for consistency
    cs_interactions = {}
    for key, df in interactions.items():
        cs_interactions[key] = df
        cs_key = (key[0], f"cs_rank_{key[1]}")
        cs_interactions[cs_key] = _rank_cs(df)

    n_feats = len(interactions)
    logger.info(f"Interaction features: {n_feats} interactions ({n_feats * 2} with ranks)")
    return cs_interactions


def _rank_single(series: pd.Series) -> pd.Series:
    """
    Rolling percentile rank of a single time series — PIT-correct.

    Previously used .rank(pct=True), which computed rank over the FULL series,
    including future values. That was look-ahead bias even if the parent feature
    was dead code. Now uses a 252-day rolling window so the feature is safe
    if build_interaction_features is ever re-enabled.
    """
    return series.rolling(252, min_periods=63).apply(
        lambda x: (x <= x[-1]).mean(), raw=True
    )
