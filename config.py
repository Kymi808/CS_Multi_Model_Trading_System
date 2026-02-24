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
    min_avg_dollar_volume: float = 5_000_000
    max_missing_pct: float = 0.10

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
    retrain_every_days: int = 21
    purge_gap_days: int = 10
    embargo_days: int = 5

    # Ensemble
    n_ensemble: int = 3
    ensemble_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


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
    commission_bps: float = 0.5
    slippage_bps: float = 3.0
    spread_bps: float = 2.0
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


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    results_dir: str = "results"
