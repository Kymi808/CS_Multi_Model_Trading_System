"""
Hidden Markov Model regime detection for equity markets.

Replaces simple moving average crossovers with a proper statistical model
that identifies market regimes (bull/sideways/bear) from multivariate
macro observations.

3-state Gaussian HMM trained on:
1. Market return (21-day SPY log return)
2. Realized volatility (21-day SPY vol, annualized)
3. Credit spread change (HYG - LQD ratio, 21-day)
4. Yield curve slope (TLT - SHV, or 10Y - 3M)
5. VIX level

The HMM provides:
- Current regime classification (bull/sideways/bear)
- Probability of each regime (soft assignment, more informative than hard)
- Transition matrix (probability of switching regimes)
- Regime duration (how long in current regime)

References:
- Hamilton (1989), "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle"
- Ang & Bekaert (2002), "Regime Switches in Interest Rates"
"""
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import hmmlearn; graceful fallback if not installed
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.info("hmmlearn not installed — HMM regime detection unavailable. "
                "Install with: pip install hmmlearn")


@dataclass
class HMMRegimeState:
    """Output of the HMM regime detector."""
    current_regime: str = "unknown"         # "bull", "sideways", "bear"
    regime_probabilities: dict = field(default_factory=lambda: {
        "bull": 0.33, "sideways": 0.34, "bear": 0.33
    })
    transition_matrix: Optional[np.ndarray] = None
    regime_score: float = 0.0               # [-0.1, 0.1] for backward compat
    regime_duration: int = 0                # days in current regime
    confidence: float = 0.0                 # max probability

    def to_dict(self) -> dict:
        return {
            "hmm_regime": self.current_regime,
            "hmm_probabilities": self.regime_probabilities,
            "hmm_confidence": round(self.confidence, 3),
            "hmm_regime_duration": self.regime_duration,
            "regime_score": round(self.regime_score, 4),
        }


# State labels (sorted by mean market return after training)
REGIME_LABELS = {0: "bear", 1: "sideways", 2: "bull"}


class HMMRegimeDetector:
    """
    3-state Gaussian HMM for market regime detection.

    Trains on multivariate macro observations using the EM algorithm.
    Provides soft regime assignments (probabilities) rather than hard labels.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_features: int = 5,
        refit_every: int = 63,
        min_train_days: int = 504,
    ):
        self.n_states = n_states
        self.n_features = n_features
        self.refit_every = refit_every
        self.min_train_days = min_train_days
        self.model: Optional[GaussianHMM] = None
        self.state_order: Optional[np.ndarray] = None  # maps internal -> labeled states
        self.last_fit_length: int = 0
        self._regime_history: list[int] = []

    def prepare_observations(
        self,
        cross_asset_prices: pd.DataFrame,
        zscore_window: int = 252,
    ) -> np.ndarray:
        """
        Extract and z-score the 5 observable features from cross-asset data.

        Expected columns in cross_asset_prices:
        - ^GSPC or SPY: market prices
        - ^VIX or VIXY: volatility index proxy
        - HYG: high yield bonds
        - LQD: investment grade bonds
        - TLT or ^TNX: long-term treasuries
        - SHV or ^IRX: short-term treasuries

        Returns: (n_days, 5) array of z-scored observations
        """
        # Find available columns (handles both yfinance and Alpaca ticker names)
        def _find(names):
            for n in names:
                if n in cross_asset_prices.columns:
                    return cross_asset_prices[n]
            return None

        mkt = _find(["^GSPC", "SPY"])
        vix = _find(["^VIX", "VIXY"])
        hyg = _find(["HYG"])
        lqd = _find(["LQD"])
        tlt = _find(["TLT", "^TNX"])
        shv = _find(["SHV", "^IRX"])

        if mkt is None:
            logger.warning("HMM: no market data available")
            return np.array([])

        # Feature 1: 21-day market log return
        mkt_ret = np.log(mkt / mkt.shift(21)).dropna()

        # Feature 2: 21-day realized volatility (annualized)
        daily_ret = np.log(mkt / mkt.shift(1))
        mkt_vol = daily_ret.rolling(21).std() * np.sqrt(252)

        # Feature 3: Credit spread change (HYG vs LQD, 21-day)
        if hyg is not None and lqd is not None:
            credit = (hyg / lqd).pct_change(21)
        else:
            credit = pd.Series(0.0, index=mkt.index)

        # Feature 4: Yield curve slope (TLT vs SHV)
        if tlt is not None and shv is not None:
            curve = (tlt / shv).pct_change(21)
        else:
            curve = pd.Series(0.0, index=mkt.index)

        # Feature 5: VIX level
        if vix is not None:
            vix_level = vix
        else:
            vix_level = pd.Series(20.0, index=mkt.index)

        # Combine and align
        features = pd.DataFrame({
            "mkt_ret": mkt_ret,
            "mkt_vol": mkt_vol,
            "credit": credit,
            "curve": curve,
            "vix": vix_level,
        }).dropna()

        if len(features) < self.min_train_days:
            return np.array([])

        # Z-score with rolling window (prevents look-ahead)
        obs = features.copy()
        for col in obs.columns:
            rolling_mean = obs[col].rolling(zscore_window, min_periods=63).mean()
            rolling_std = obs[col].rolling(zscore_window, min_periods=63).std()
            obs[col] = (obs[col] - rolling_mean) / (rolling_std + 1e-8)

        # Drop NaN rows from rolling
        obs = obs.dropna()

        return obs.values

    def fit(self, observations: np.ndarray) -> bool:
        """
        Fit the HMM on observations.

        Uses EM algorithm with k-means initialization.
        After fitting, sorts states by mean market return to ensure
        consistent labeling (bear=lowest, bull=highest).

        Returns True if fitting succeeded.
        """
        if not HMM_AVAILABLE:
            logger.warning("hmmlearn not installed — cannot fit HMM")
            return False

        if len(observations) < self.min_train_days:
            logger.warning(f"HMM: insufficient data ({len(observations)} < {self.min_train_days})")
            return False

        try:
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=200,
                tol=1e-4,
                random_state=42,
                init_params="stmc",  # init all parameters
            )

            self.model.fit(observations)

            # Sort states by mean of first feature (market return)
            # This ensures: state 0 = bear (lowest return), state 2 = bull (highest)
            means = self.model.means_[:, 0]  # market return means
            self.state_order = np.argsort(means)  # [bear_idx, sideways_idx, bull_idx]

            self.last_fit_length = len(observations)

            logger.info(
                f"HMM fit: {self.n_states} states on {len(observations)} observations. "
                f"Means: bear={means[self.state_order[0]]:.4f}, "
                f"sideways={means[self.state_order[1]]:.4f}, "
                f"bull={means[self.state_order[2]]:.4f}"
            )
            return True

        except Exception as e:
            logger.error(f"HMM fit failed: {e}")
            return False

    def predict(self, observations: np.ndarray) -> HMMRegimeState:
        """
        Predict current regime with probabilities.

        Uses the forward algorithm to compute P(state | all observations),
        which is more nuanced than Viterbi's hard assignment.
        """
        if self.model is None or self.state_order is None:
            return HMMRegimeState()

        if len(observations) == 0:
            return HMMRegimeState()

        try:
            # Get probabilities for the latest observation
            # predict_proba returns (n_samples, n_states)
            probs = self.model.predict_proba(observations)
            latest_probs = probs[-1]  # last observation

            # Re-order probabilities to match our labeling (bear, sideways, bull)
            ordered_probs = latest_probs[self.state_order]

            # Determine current regime
            regime_idx = np.argmax(ordered_probs)
            regime_name = REGIME_LABELS[regime_idx]

            # Compute regime duration (consecutive same-state observations)
            states = self.model.predict(observations)
            # Map to our ordering
            mapped_states = np.array([
                np.where(self.state_order == s)[0][0] for s in states
            ])
            current = mapped_states[-1]
            duration = 1
            for i in range(len(mapped_states) - 2, -1, -1):
                if mapped_states[i] == current:
                    duration += 1
                else:
                    break

            # Backward-compatible regime_score: P(bull) - P(bear), clipped to [-0.1, 0.1]
            regime_score = float(np.clip(
                ordered_probs[2] - ordered_probs[0],  # P(bull) - P(bear)
                -0.1, 0.1
            ))

            state = HMMRegimeState(
                current_regime=regime_name,
                regime_probabilities={
                    "bull": round(float(ordered_probs[2]), 4),
                    "sideways": round(float(ordered_probs[1]), 4),
                    "bear": round(float(ordered_probs[0]), 4),
                },
                transition_matrix=self.model.transmat_,
                regime_score=regime_score,
                regime_duration=duration,
                confidence=round(float(np.max(ordered_probs)), 4),
            )

            self._regime_history.append(regime_idx)

            return state

        except Exception as e:
            logger.error(f"HMM predict failed: {e}")
            return HMMRegimeState()

    def should_refit(self, current_length: int) -> bool:
        """Check if it's time to refit based on new data."""
        if self.model is None:
            return True
        return (current_length - self.last_fit_length) >= self.refit_every
