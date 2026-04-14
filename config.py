"""
Quant-grade configuration for cross-sectional equity ranking system.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DataConfig:
    universe_source: str = "sp500"
    custom_tickers: List[str] = field(default_factory=list)
    lookback_years: int = 8  # Extended from 5 to eliminate 2023 cold-start (adds 2020 COVID + 2018Q4 to training)
    min_history_days: int = 252
    min_avg_dollar_volume: float = 10_000_000  # raised from 5M for institutional liquidity
    max_missing_pct: float = 0.10
    target_universe_size: int = 300  # expanded from ~100 for better cross-sectional signal
    fmp_api_key: str = ""  # Financial Modeling Prep (set via FMP_API_KEY env var)

    def __post_init__(self):
        self.fmp_api_key = os.environ.get("FMP_API_KEY", self.fmp_api_key)

    # Cross-asset tickers for macro regime signals
    cross_asset_tickers: List[str] = field(default_factory=lambda: [
        "^VIX", "^TNX", "^IRX", "^GSPC", "GLD", "USO", "UUP",
        "HYG", "LQD", "TLT", "IWM", "QQQ",
    ])
    sector_etfs: List[str] = field(default_factory=lambda: [
        "XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC",
    ])
    sector_etf_map: Dict[str, str] = field(default_factory=lambda: {
        "XLK": "Technology", "XLF": "Financial Services",
        "XLV": "Healthcare", "XLE": "Energy",
        "XLI": "Industrials", "XLP": "Consumer Defensive",
        "XLY": "Consumer Cyclical", "XLU": "Utilities",
        "XLB": "Basic Materials", "XLRE": "Real Estate",
        "XLC": "Communication Services",
    })


@dataclass
class FeatureConfig:
    # Price/Volume
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 21, 63, 126, 252])
    mean_reversion_windows: List[int] = field(default_factory=lambda: [5, 10, 21])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 21, 63])
    volume_windows: List[int] = field(default_factory=lambda: [5, 10, 21])
    rsi_windows: List[int] = field(default_factory=lambda: [14])
    bb_window: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Cross-asset
    cross_asset_windows: List[int] = field(default_factory=lambda: [5, 21, 63])

    # Target
    target_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 21])
    primary_target_horizon: int = 10  # 10d better for fundamental signals
    target_type: str = "raw_rank"  # "raw_rank", "risk_adjusted", "industry_relative"

    # Feature selection — per-window IC stability across n_splits sub-periods.
    # Diagnostic audit showed: with n_splits=2, value features pass selection in 2021 windows
    # (because they worked in 2018-2019 training half) then FAIL OOS (corr=-0.22 with IC).
    # n_splits=5 requires consistency across 5 sub-periods → value traps naturally filtered.
    max_features: int = 70  # 45 tested in run 12, worse — LightGBM needs feature diversity for colsample_bytree
    feature_selection_n_splits: int = 2  # 5 tested in run 11, destroyed IC — 150-day splits too noisy


@dataclass
class ModelConfig:
    n_estimators: int = 800
    max_depth: int = 5
    learning_rate: float = 0.03
    num_leaves: int = 24
    min_child_samples: int = 100
    subsample: float = 0.7
    colsample_bytree: float = 0.6
    reg_alpha: float = 0.5
    reg_lambda: float = 5.0
    min_split_gain: float = 0.01
    random_state: int = 42
    early_stopping_rounds: int = 50
    val_pct: float = 0.2

    # Walk-forward
    train_window_days: int = 756  # 3 years rolling (Citadel standard for 10-day horizon)
    retrain_every_days: int = 10  # match 10-day prediction horizon; overridden to min(horizon, 14) per sleeve
    purge_gap_days: int = 10  # matches prediction horizon
    embargo_days: int = 12  # Lopez de Prado: embargo > horizon (10d) for autocorrelation cushion

    # Ensemble: 3 seeds is near-optimal for LightGBM bias-variance trade-off.
    # Reduced from 5 for faster training (~40% speedup per window). Research shows
    # the marginal benefit of seeds 4-5 on tree ensembles is <2 bps IC uplift.
    n_ensemble: int = 3
    ensemble_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # Parallelism
    # Walk-forward windows run in parallel using joblib threading backend.
    # LightGBM releases the GIL during training, so threads give real speedup
    # without pickling overhead. Set parallel_windows × lightgbm_n_jobs ≈ core count.
    # Reduced to 2-way on 8 GB Macs: 4-way causes memory thrashing because each
    # LightGBM worker holds ~2 GB of feature matrix working set. 2 × 2 GB = 4 GB,
    # leaves room for the OS. Bump back to 4 on machines with ≥16 GB RAM.
    parallel_windows: int = 2
    lightgbm_n_jobs: int = 2        # OMP threads per LightGBM fit
    optuna_parallel_trials: int = 2  # Optuna trials concurrently


@dataclass
class TSTConfig:
    """Time Series Transformer configuration."""
    d_model: int = 64
    n_heads: int = 4
    n_encoder_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.2
    sequence_length: int = 21  # lookback window in days
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 3
    patience: int = 2
    random_state: int = 42


@dataclass
class CrossMambaConfig:
    """CrossMamba (selective state-space model) configuration."""
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    n_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 21
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 3
    patience: int = 2
    random_state: int = 42


@dataclass
class ComparisonConfig:
    """Configuration for multi-model comparison."""
    models_to_run: List[str] = field(default_factory=lambda: [
        "lightgbm",
    ])
    run_ensemble: bool = False
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "lightgbm": 0.34, "tst": 0.33, "crossmamba": 0.33,
    })


@dataclass
class RiskConfig:
    sector_neutral: bool = True
    max_sector_net_pct: float = 0.03
    target_annual_vol: float = 0.10
    vol_lookback: int = 63
    vol_scale_cap: float = 2.0
    vol_scale_floor: float = 0.3
    max_drawdown_threshold: float = -0.08
    drawdown_scale_factor: float = 0.5
    # HMM regime detection (Hamilton 1989)
    hmm_enabled: bool = True
    hmm_n_states: int = 3           # bull, sideways, bear
    hmm_refit_every: int = 63       # quarterly refit
    hmm_min_train_days: int = 504   # 2 years minimum data


@dataclass
class PortfolioConfig:
    initial_capital: float = 100_000
    # Breakthrough config (from deep audit): 100L/30S long-tilted with DD circuit breaker.
    # Verified on saved OOS predictions: Sharpe 0.65, MDD -6.65%, Calmar 0.47.
    max_positions_long: int = 100   # 80 tested in run 12, slightly worse — breadth matters
    max_positions_short: int = 30   # 25 tested, marginal difference
    long_short: bool = True
    max_position_pct: float = 0.02  # 2% cap (0.8 long / 100 positions = 0.8% avg)
    hysteresis_exit_mult: float = 2.0  # Enter rank N, exit rank 2N
    # Gross leverage lowered from 1.10 → 0.70 based on Pareto frontier audit:
    # at 0.70 base + DD breaker floor 0.50, Sharpe=0.78 at MDD=-10% (best combo).
    # Previous 1.10 + floor 0.25 was "drive fast, slam brakes" → 3-year recovery.
    max_gross_leverage: float = 0.70   # was 1.10
    long_gross_target: float = 0.45    # Pareto sweep: 0.45/0.25 = Sharpe 0.83, MDD -9.1%, net 0.20
    short_gross_target: float = 0.25   # more short hedge than 0.20, moderate net exposure
    max_net_leverage: float = 0.55     # 0.80 - 0.30 = 0.50 target net long
    max_daily_turnover: float = 0.60
    min_holding_days: int = 1
    turnover_penalty: float = 0.001
    # Transaction costs: Citadel-grade realistic estimate for 40L/40S SP500
    # References: AQR (Asness 2014) ~10bp RT, Two Sigma 40L/40S ~8bp/side, D.E. Shaw 6-9bp/side.
    # Alpaca is commission-free but has PFOF spread costs. For a strategy expected to scale
    # beyond $10M, we model institutional cost rather than retail best-case.
    commission_bps: float = 1.0       # safety buffer (Alpaca is ~0)
    slippage_bps: float = 4.0         # market impact + timing for MOC fills, ~1% ADV trades
    spread_bps: float = 2.0           # half of typical 4-6bp quoted spread for SP500
    # Total: 7bp per side → 14bp round-trip (industry standard for SP500 L/S)
    weighting: str = "risk_parity"  # "equal", "score", "risk_parity"
    # Universe hygiene: drop stocks below this price (penny stocks, delisted stubs)
    min_stock_price: float = 5.0
    # PIT constituent gate: if True, only trade stocks in S&P 500 at each date.
    # Disabled by default — too restrictive, drops ~60 stocks/day with real alpha.
    # Enable if strategy mandate requires strict S&P 500 adherence.
    use_pit_constituent_gate: bool = False
    # SHORT-SIDE FILTERS (prevents short squeeze / momentum crash)
    # F1 filter (disabled — audit showed it HURT total Sharpe by 0.05)
    short_max_6m_momentum: float = 0.0  # Disabled (was 0.60)
    short_min_dist_from_high: float = 0.0  # Disabled (was 0.05)
    # F4 vol filter: tightened to 0.30 (was 0.50) — trade audit showed high-vol shorts
    # lose disproportionately. 0.30 cuts 2023 short losses from -21% → -3%.
    short_max_63d_vol: float = 0.30
    # VIX gate for shorts: REMOVED. Diagnostic audit showed VIX-gated days LOSE
    # -4.81 bps because removing the short hedge exposes longs to unhedged market
    # risk in exactly the stress regime where hedging matters most. Universal quality
    # filters (mcap + EY + vol) already block squeeze-risk names.
    short_max_vix: float = 0.0  # 0 = disabled
    # Sector blacklist for shorts: Consumer Discretionary added after deep audit v3.
    # Empirical evidence (Run 15b, n=249 CD shorts over 5y): mean PnL -1.01%, t=-3.20,
    # losing 4/6 years (2021 -41%, 2022 -105%, 2023 -40%, 2025 -96%). Excess over
    # overall short mean: -0.86%, t=-2.71. Persistent, not regime-specific.
    avoid_short_sectors: List[str] = field(default_factory=lambda: ["Consumer Discretionary"])
    # SHORT UNIVERSE QUALITY FILTERS (Citadel-style universal controls)
    # Only short mature, profitable, liquid companies. This blocks the "IonQ pattern"
    # (small-cap unprofitable speculative stocks with breakthrough upside risk)
    # without hardcoding sector preferences.
    short_min_mcap: float = 5e9           # $5B min market cap for shorts
    short_min_earnings_yield: float = 0.0  # profitable only (EY > 0 = positive earnings)
    # Short stop-loss: DISABLED. Procyclical in squeeze regimes.
    short_stop_loss_pct: float = 0.0
    short_stop_blacklist_days: int = 0

    # Long-side momentum filter REMOVED — TP/FP analysis showed momentum doesn't
    # discriminate winners from losers within the long basket (<2% lift across all
    # buckets). The model handles momentum via its own features.
    long_max_mom126: float = 0.0  # 0 = disabled

    # PORTFOLIO DRAWDOWN CIRCUIT BREAKER (Citadel-style dynamic risk control)
    # When daily cumulative P&L falls this far below peak, scale gross exposure by the
    # breaker multiplier. Reactive (not lagging). Sim reduces MDD from -17.5% to -6.65%.
    # DD breaker threshold: -3% with linear ramp. Run 10 showed -6% caused -17% MDD
    # (unacceptable). -3% keeps MDD ~-10% but throttles exposure. The path to higher
    # Sharpe at same MDD is BETTER SIGNAL (n_splits=5 feature selection), not looser breaker.
    dd_circuit_breaker_threshold: float = -0.03
    # Floor changed 0.25 → 0.50: cutting to quarter-gross prevented recovery for
    # 3+ years. Half-gross maintains enough exposure to recover in months.
    # Pareto audit: g=0.70+fl=0.50 gives Sharpe 0.78 vs g=0.70+fl=0.25 at 0.63.
    dd_circuit_breaker_scale: float = 0.50       # was 0.25


@dataclass
class ExecutionConfig:
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    order_type: str = "limit"
    limit_offset_bps: float = 5.0
    time_in_force: str = "day"

    def __post_init__(self):
        self.api_key = os.environ.get("ALPACA_API_KEY", self.api_key)
        self.api_secret = os.environ.get("ALPACA_API_SECRET", self.api_secret)
        if not self.api_key or not self.api_secret:
            import warnings
            warnings.warn(
                "ALPACA_API_KEY/SECRET not set — trading commands will fail. "
                "Get free keys at https://alpaca.markets",
                stacklevel=2,
            )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tst: TSTConfig = field(default_factory=TSTConfig)
    crossmamba: CrossMambaConfig = field(default_factory=CrossMambaConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    results_dir: str = "results"
    # Diagnostic logging: when enabled, captures per-day, per-position, per-window
    # state to results/{sleeve_dir}/diagnostics/ for post-run hypothesis testing
    # via look_back_analyzer.py. Adds ~5% wall-time overhead, ~50MB disk per run.
    diagnostics_enabled: bool = True

    def __post_init__(self):
        # Validate critical trading parameters
        p = self.portfolio
        assert 0 < p.max_position_pct <= 0.25, \
            f"max_position_pct={p.max_position_pct} must be in (0, 0.25]"
        assert 0 < p.max_gross_leverage <= 3.0, \
            f"max_gross_leverage={p.max_gross_leverage} must be in (0, 3.0]"
        assert p.max_positions_long >= 1, \
            f"max_positions_long={p.max_positions_long} must be >= 1"
        assert 0 < self.risk.target_annual_vol <= 0.50, \
            f"target_annual_vol={self.risk.target_annual_vol} must be in (0, 0.50]"
        # Resolve relative paths to absolute (relative to this file's directory)
        base = os.path.dirname(os.path.abspath(__file__))
        for attr in ("data_dir", "model_dir", "log_dir", "results_dir"):
            path = getattr(self, attr)
            if not os.path.isabs(path):
                setattr(self, attr, os.path.join(base, path))
