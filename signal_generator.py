"""
Signal generation pipeline — identical to backtest Steps 1-7 + 10-12.

Used by:
- cmd_trade (live/paper trading)
- cmd_signal (daily signal generation)
- backtest.py imports the same feature/risk functions

This ensures backtest and live trading use the EXACT same code path.
"""
import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
from typing import Dict, Tuple, Optional

from config import Config
from data_loader import (
    fetch_price_data, fetch_cross_asset_data,
    fetch_fundamental_data, fetch_earnings_dates,
)
from universe import get_universe, filter_universe_by_liquidity, load_sector_map
from fundamental_features import build_fundamental_features
from cross_asset_features import build_cross_asset_features
from sentiment_features import fetch_news_sentiment, build_sentiment_features
from features import build_all_features, panel_to_ml_format
from fmp_features import fetch_fmp_fundamentals, build_fmp_features
from openbb_features import fetch_options_data, fetch_short_interest, build_openbb_features
from model import EnsembleRanker
from risk_model import FactorRiskModel
from portfolio import PortfolioConstructor

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generates trading signals using the same pipeline as the backtest.

    Pipeline:
    1. Fetch universe + prices
    2. Fetch sectors
    3. Fetch fundamentals + earnings
    4. Fetch sentiment
    5. Fetch cross-asset data
    6. Build features
    7. Load model + predict
    8. Construct portfolio (same PortfolioConstructor)
    9. Apply risk scaling (same FactorRiskModel with regime, clip, etc.)
    10. Return target weights
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[EnsembleRanker] = None
        self.risk: Optional[FactorRiskModel] = None
        self.constructor: Optional[PortfolioConstructor] = None
        self.selected_features: list = []
        self.sector_map: dict = {}
        self.prev_weights: pd.Series = pd.Series(dtype=float)
        self.portfolio_returns: pd.Series = pd.Series(dtype=float)

    def load_model(self, model_path: str = "models/latest_model.pkl"):
        """Load trained model from backtest."""
        self.model = EnsembleRanker(self.cfg.model)
        self.model.load(model_path)
        logger.info(f"Loaded model: {len(self.model.models)} models, "
                    f"{len(self.model.feature_names)} features")
        self.selected_features = self.model.feature_names

    def load_feature_list(self, path: str = "results/feature_importance.csv"):
        """Load selected features from backtest."""
        if os.path.exists(path):
            fi = pd.read_csv(path, index_col=0)
            self.selected_features = fi.index.tolist()
            logger.info(f"Loaded {len(self.selected_features)} features from {path}")

    def initialize_risk(self):
        """Initialize risk model and portfolio constructor."""
        self.risk = FactorRiskModel(self.cfg.risk)
        self.constructor = PortfolioConstructor(self.cfg.portfolio)

    def generate_signals(self) -> Tuple[pd.Series, dict]:
        """
        Full signal generation pipeline.

        Returns:
            target_weights: pd.Series of ticker -> weight
            info: dict with diagnostics
        """
        if self.model is None:
            raise ValueError("Call load_model() first")
        if self.risk is None:
            self.initialize_risk()

        info = {}

        # 1. Universe + prices
        logger.info("Fetching universe & prices...")
        tickers = get_universe(self.cfg.data)
        prices, volumes = fetch_price_data(
            tickers, self.cfg.data, cache_dir=self.cfg.data_dir,
        )
        tickers = filter_universe_by_liquidity(tickers, self.cfg.data, prices, volumes)
        prices = prices[[t for t in tickers if t in prices.columns]]
        volumes = volumes[[t for t in tickers if t in volumes.columns]]
        tickers = list(prices.columns)
        info["n_tickers"] = len(tickers)
        logger.info(f"Universe: {len(tickers)} tickers")

        # 2. Sectors
        logger.info("Fetching sectors...")
        self.sector_map = load_sector_map(tickers, cache_dir=self.cfg.data_dir)

        # 3. Fundamentals
        logger.info("Fetching fundamentals...")
        fundamentals = fetch_fundamental_data(tickers, cache_dir=self.cfg.data_dir)
        earnings_dates = fetch_earnings_dates(tickers, cache_dir=self.cfg.data_dir)
        fund_feats = build_fundamental_features(
            fundamentals, prices, earnings_dates, self.sector_map,
        )

        # 4. Sentiment
        logger.info("Fetching sentiment...")
        sentiment_data = fetch_news_sentiment(tickers, cache_dir=self.cfg.data_dir)
        sent_feats = build_sentiment_features(sentiment_data, prices)

        # 5. Cross-asset
        logger.info("Fetching cross-asset data...")
        all_ca = self.cfg.data.cross_asset_tickers + self.cfg.data.sector_etfs
        ca_prices = fetch_cross_asset_data(
            all_ca, prices.index[0].strftime("%Y-%m-%d"),
            prices.index[-1].strftime("%Y-%m-%d"),
            cache_dir=self.cfg.data_dir,
        )
        ca_only = ca_prices[
            [c for c in self.cfg.data.cross_asset_tickers if c in ca_prices.columns]
        ] if not ca_prices.empty else pd.DataFrame()
        sect_etf = ca_prices[
            [c for c in self.cfg.data.sector_etfs if c in ca_prices.columns]
        ] if not ca_prices.empty else pd.DataFrame()
        ca_feats = build_cross_asset_features(
            ca_only, prices, sect_etf, self.sector_map,
            self.cfg.features.cross_asset_windows,
        )

        # 5b. FMP point-in-time fundamentals
        fmp_feats = {}
        try:
            fmp_data = fetch_fmp_fundamentals(tickers, self.cfg.data.fmp_api_key, self.cfg.data_dir)
            fmp_feats = build_fmp_features(fmp_data, prices)
        except Exception as e:
            logger.debug(f"FMP features skipped: {e}")

        # 5c. OpenBB alternative data (live mode — production signals)
        openbb_feats = {}
        try:
            options_data = fetch_options_data(tickers, cache_dir=self.cfg.data_dir, live_mode=True)
            short_data = fetch_short_interest(tickers, cache_dir=self.cfg.data_dir, live_mode=True)
            openbb_feats = build_openbb_features(options_data, short_data, prices)
        except Exception as e:
            logger.debug(f"OpenBB features skipped: {e}")

        # 5d. Insider features (SEC Form 4)
        insider_feats = {}
        try:
            from insider_features import fetch_insider_data, build_insider_features
            insider_data = fetch_insider_data(tickers, cache_dir=self.cfg.data_dir)
            insider_feats = build_insider_features(insider_data, prices, self.fundamentals)
        except Exception as e:
            logger.debug(f"Insider features skipped: {e}")

        # 6. Build features
        logger.info("Building features...")
        features, targets = build_all_features(
            prices, volumes, self.cfg.features,
            fundamental_feats=fund_feats,
            cross_asset_feats={**sent_feats, **ca_feats},
            insider_feats=insider_feats,
            fmp_feats=fmp_feats,
            openbb_feats=openbb_feats,
            sector_map=self.sector_map,
        )

        # Convert panel to ML format (same as backtest's panel_to_ml_format)
        if isinstance(features.columns, pd.MultiIndex) and features.columns.nlevels >= 2:
            feat_stacked = features.stack(level=-1)
            feat_stacked.index.names = ["date", "ticker"]
            if isinstance(feat_stacked.columns, pd.MultiIndex):
                feat_stacked.columns = [
                    "_".join(str(c) for c in col) for col in feat_stacked.columns
                ]
        else:
            feat_stacked = features.copy()

        # Get latest date's features (now a DataFrame: tickers × features)
        latest_date = feat_stacked.index.get_level_values(0).max()
        latest_features = feat_stacked.loc[latest_date]
        info["date"] = str(latest_date)

        # Filter to selected features (from trained model)
        available = [f for f in self.selected_features if f in latest_features.columns]
        missing = [f for f in self.selected_features if f not in latest_features.columns]
        if missing:
            logger.warning(f"Missing {len(missing)} features, filling with NaN")

        X_latest = latest_features[available].copy()
        for f in missing:
            X_latest[f] = np.nan

        # Reorder to match model's expected feature order
        X_latest = X_latest[self.selected_features]

        # Fill NaN with cross-sectional median (same as backtest)
        for col in X_latest.columns:
            if X_latest[col].isna().any():
                X_latest[col] = X_latest[col].fillna(X_latest[col].median())
        X_latest = X_latest.fillna(0)  # Final fallback

        # 7. Predict
        logger.info("Generating predictions...")
        predictions = self.model.predict(X_latest)
        # predictions already has ticker index from X_latest
        info["n_predictions"] = len(predictions)
        info["pred_mean"] = float(predictions.mean())
        info["pred_std"] = float(predictions.std())

        # 8. Risk model — estimate + update regime
        logger.info("Estimating risk model...")
        self.risk.estimate(prices, fundamentals, latest_date, lookback=504)
        self.risk.update_regime(prices)
        info["regime_score"] = self.risk.regime_score

        # 9. Build portfolio
        logger.info("Constructing portfolio...")
        stock_vol = np.log(prices / prices.shift(1)).rolling(63).std() * np.sqrt(252)
        vol_est = stock_vol.iloc[-1] if len(stock_vol) > 63 else None

        target_weights = self.constructor.construct_portfolio(
            predictions=predictions,
            date=latest_date,
            prev_weights=self.prev_weights,
            vol_estimates=vol_est,
        )

        # 10. Apply risk scaling (sector, factor, vol, drawdown, clip, regime)
        n_long = self.cfg.portfolio.max_positions_long
        n_short = self.cfg.portfolio.max_positions_short

        target_weights = self.risk.apply_risk_scaling(
            target_weights, self.portfolio_returns, self.sector_map,
            n_long=n_long, n_short=n_short,
        )

        # Position limits (same as backtest)
        target_weights = target_weights.clip(
            -self.cfg.portfolio.max_position_pct,
            self.cfg.portfolio.max_position_pct,
        )

        # Store for next call
        self.prev_weights = target_weights

        info["n_long"] = int((target_weights > 0).sum())
        info["n_short"] = int((target_weights < 0).sum())
        info["net_exposure"] = float(target_weights.sum())
        info["gross_exposure"] = float(target_weights.abs().sum())

        logger.info(f"Portfolio: {info['n_long']}L / {info['n_short']}S, "
                    f"net={info['net_exposure']:.1%}, gross={info['gross_exposure']:.1%}")

        return target_weights, info

    def update_returns(self, daily_return: float):
        """Call after each trading day with realized return."""
        self.portfolio_returns = pd.concat([
            self.portfolio_returns,
            pd.Series([daily_return], index=[pd.Timestamp.now()]),
        ])
        self.risk.update(daily_return)
