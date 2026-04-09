"""
Quant-grade configuration for cross-sectional equity ranking system.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import os


@dataclass
class DataConfig:
    universe_source: str = "sp500"
    custom_tickers: List[str] = field(default_factory=list)
    lookback_years: int = 5
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

    # Feature selection
    max_features: int = 50  # validated: 50 features produced 3.4 Sharpe. 65 overfits LightGBM.


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
    train_window_days: int = 504
    retrain_every_days: int = 14  # match 10-day prediction horizon (10 trading days = 14 calendar days)
    purge_gap_days: int = 10
    embargo_days: int = 5

    # Ensemble
    n_ensemble: int = 5
    ensemble_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])


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
    max_positions_long: int = 10
    max_positions_short: int = 10
    long_short: bool = True
    max_position_pct: float = 0.10
    max_gross_leverage: float = 1.6
    max_net_leverage: float = 0.15
    max_daily_turnover: float = 0.40
    min_holding_days: int = 1
    turnover_penalty: float = 0.001
    # Realistic transaction costs (conservative estimates for production)
    # Alpaca is commission-free but has payment-for-order-flow spread costs
    commission_bps: float = 1.0       # effective PFOF cost ~1bp
    slippage_bps: float = 8.0        # realistic: 5-15bp depending on urgency/liquidity
    spread_bps: float = 3.0          # typical bid-ask for liquid large-caps
    # Total round-trip cost: ~24bp (12bp each way) — conservative but realistic
    weighting: str = "risk_parity"  # "equal", "score", "risk_parity"


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
