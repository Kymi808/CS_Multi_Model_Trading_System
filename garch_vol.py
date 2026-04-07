"""
GARCH(1,1) volatility forecasting for the risk model.

Replaces simple rolling standard deviation with a proper conditional
volatility model that captures volatility clustering (big moves follow
big moves) and mean reversion (high vol reverts to long-run average).

GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
Where:
- ω = long-run variance component
- α = ARCH coefficient (reaction to recent shocks)
- β = GARCH coefficient (persistence of past volatility)
- α + β < 1 for stationarity (typically ~0.95-0.99 for daily equity)

Used for:
1. Better vol targeting in portfolio construction (forward-looking, not backward)
2. Better risk parity weights (GARCH vol > rolling vol for regime transitions)
3. Conditional VaR estimates (fat tails captured by GARCH)

References:
- Bollerslev (1986), "Generalized Autoregressive Conditional Heteroskedasticity"
- Engle (1982), "Autoregressive Conditional Heteroscedasticity"
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class GARCH11:
    """
    GARCH(1,1) model fitted via maximum likelihood.

    Simple but robust implementation. For production, consider
    arch package (pip install arch) which has more model variants.
    This implementation avoids the extra dependency.
    """

    def __init__(self, omega: float = 0.0, alpha: float = 0.1, beta: float = 0.85):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.long_run_var = 0.0
        self._fitted = False

    def fit(self, returns: np.ndarray) -> dict:
        """
        Fit GARCH(1,1) parameters via variance targeting + grid search.

        Variance targeting: ω = (1 - α - β) × unconditional_variance
        Then grid search over (α, β) to maximize log-likelihood.
        """
        returns = np.array(returns, dtype=float)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 50:
            # Not enough data — use simple variance
            self.omega = float(np.var(returns))
            self._fitted = False
            return {"status": "insufficient_data"}

        unconditional_var = float(np.var(returns))
        self.long_run_var = unconditional_var

        # Grid search over α and β
        best_ll = -np.inf
        best_params = (0.1, 0.85)

        for alpha in np.arange(0.02, 0.20, 0.02):
            for beta in np.arange(0.70, 0.97, 0.02):
                if alpha + beta >= 0.999:
                    continue
                omega = unconditional_var * (1 - alpha - beta)
                if omega <= 0:
                    continue
                ll = self._log_likelihood(returns, omega, alpha, beta)
                if ll > best_ll:
                    best_ll = ll
                    best_params = (alpha, beta)

        self.alpha, self.beta = best_params
        self.omega = unconditional_var * (1 - self.alpha - self.beta)
        self._fitted = True

        persistence = self.alpha + self.beta
        half_life = np.log(2) / np.log(1 / persistence) if persistence < 1 else float('inf')

        logger.debug(
            f"GARCH fit: α={self.alpha:.3f}, β={self.beta:.3f}, "
            f"persistence={persistence:.3f}, half-life={half_life:.1f}d"
        )

        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "omega": self.omega,
            "persistence": persistence,
            "half_life_days": half_life,
            "unconditional_vol": np.sqrt(unconditional_var) * np.sqrt(252),
        }

    def _log_likelihood(
        self, returns: np.ndarray, omega: float, alpha: float, beta: float,
    ) -> float:
        """Compute Gaussian log-likelihood for GARCH(1,1)."""
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            sigma2[t] = max(sigma2[t], 1e-10)

        # Gaussian log-likelihood
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
        return float(ll)

    def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Forecast conditional volatility for the next `horizon` days.

        Returns annualized volatility forecast for each day ahead.
        """
        if not self._fitted:
            # Fallback: simple rolling vol
            vol = float(np.std(returns[-21:])) * np.sqrt(252)
            return np.full(horizon, vol)

        # Compute current conditional variance
        sigma2 = np.var(returns)
        for t in range(1, len(returns)):
            sigma2 = self.omega + self.alpha * returns[t-1]**2 + self.beta * sigma2

        # Multi-step forecast
        forecasts = np.zeros(horizon)
        for h in range(horizon):
            if h == 0:
                forecasts[h] = sigma2
            else:
                forecasts[h] = self.omega + (self.alpha + self.beta) * forecasts[h-1]

        # Convert to annualized vol
        return np.sqrt(forecasts * 252)

    def current_vol(self, returns: np.ndarray) -> float:
        """Get current annualized conditional volatility."""
        return float(self.forecast(returns, horizon=1)[0])


def compute_garch_vol_matrix(
    prices: pd.DataFrame,
    lookback: int = 252,
    forecast_horizon: int = 10,
) -> pd.DataFrame:
    """
    Compute GARCH conditional volatility for all stocks.

    Returns DataFrame of annualized vol forecasts (dates × tickers).
    Used for risk parity position sizing (forward-looking vol).
    """
    returns = np.log(prices / prices.shift(1)).dropna()

    vol_matrix = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for ticker in returns.columns:
        rets = returns[ticker].dropna().values

        if len(rets) < 100:
            # Not enough data — use rolling vol
            vol_matrix[ticker] = returns[ticker].rolling(21).std() * np.sqrt(252)
            continue

        garch = GARCH11()

        # Expanding window fit (refit every 63 days for efficiency)
        last_fit = 0
        for i in range(lookback, len(rets)):
            if i - last_fit >= 63 or last_fit == 0:
                garch.fit(rets[max(0, i-lookback):i])
                last_fit = i

            vol = garch.current_vol(rets[max(0, i-lookback):i])
            vol_matrix.iloc[i][ticker] = vol

    return vol_matrix


def garch_risk_parity_weights(
    prices: pd.DataFrame,
    tickers_long: list,
    tickers_short: list,
    lookback: int = 252,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute GARCH-based risk parity weights.

    Instead of using rolling standard deviation (backward-looking),
    uses GARCH conditional volatility (forward-looking, captures
    volatility clustering).

    Returns: (long_weights, short_weights) as pd.Series
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    all_tickers = tickers_long + tickers_short

    vols = {}
    for ticker in all_tickers:
        if ticker not in returns.columns:
            vols[ticker] = 0.20  # default 20% vol
            continue

        rets = returns[ticker].dropna().values
        if len(rets) < 50:
            vols[ticker] = float(np.std(rets) * np.sqrt(252)) if len(rets) > 5 else 0.20
            continue

        garch = GARCH11()
        garch.fit(rets[-lookback:])
        vols[ticker] = garch.current_vol(rets[-lookback:])

    # Risk parity: weight inversely proportional to vol
    def _risk_parity(tickers, sign=1.0):
        inv_vols = {t: 1.0 / max(vols.get(t, 0.20), 0.05) for t in tickers}
        total = sum(inv_vols.values())
        return pd.Series({t: sign * iv / total for t, iv in inv_vols.items()})

    long_w = _risk_parity(tickers_long, sign=1.0)
    short_w = _risk_parity(tickers_short, sign=-1.0)

    return long_w, short_w
