# CS Multi-Model Trading System

Quantitative cross-sectional equity ranking system using three competing ML architectures. Ranks ~460 S&P 500 stocks and constructs long/short portfolios with institutional-grade risk management.

## Models

| Model | Type | Complexity | Strength |
|-------|------|-----------|----------|
| **LightGBM** | Gradient boosting | O(n log n) | Fast, handles tabular data natively |
| **TST** | Time Series Transformer | O(n²) | Multi-head attention captures cross-time dependencies |
| **CrossMamba** | Selective State-Space | O(n) | Linear-time selective scan, long-range memory |
| **Ensemble** | Weighted average | — | Diversification across architectures (the real edge) |

## Key Results (Real Data, 460 Stocks)

- **LightGBM Avg Rank IC**: 0.063 (3.3x improvement over 98-stock baseline)
- **IC Information Ratio**: 0.44
- **Ensemble results**: Pending (Colab comparison in progress)
- **Previous baseline** (Stats486, 98 stocks): Ensemble Sharpe 1.38, 16.1% annual

## Features

### Data Sources
- **Alpaca**: Universe selection (460 tradeable stocks)
- **yfinance**: Historical prices, fundamentals, sentiment (backtest)
- **FMP**: Point-in-time fundamentals (production, $29/mo)
- **OpenBB**: Options IV skew, short interest (when installed)
- **SEC EDGAR**: Insider transactions (free)

### Feature Engineering (200+ raw, 65 selected)
- **Price/Volume**: Momentum (5d-252d), mean reversion, volatility, volume, technicals
- **Fundamental**: PE, ROE, margins, growth, quality/value composites
- **Cross-Asset**: VIX, yields, credit spreads, sector ETFs, breadth
- **Sentiment**: News-based keyword scoring (LLM-enhanced via bridge)
- **Insider**: Net buying ratio, cluster buying, buy-to-market-cap
- **Interactions**: 10 institutional cross-factor combinations (Asness, Novy-Marx, Fama-French)
- **FMP**: Earnings estimate revisions, surprise, beat streak
- **OpenBB**: IV skew, put-call ratio, short interest, days to cover
- **Fractional Differentiation**: Lopez de Prado Ch. 5

### Target Construction
- **Risk-adjusted returns**: forward return / trailing volatility (Grinold & Kahn)
- **Industry-relative returns**: stock return minus sector average
- **Raw forward return rank**: traditional (fallback)

### Risk Model (Barra-style)
- **HMM Regime Detection**: 3-state Gaussian model (Hamilton 1989)
- **GARCH Volatility**: Forward-looking conditional vol for risk parity
- **Factor Neutralization**: Market, size, value, momentum, volatility, quality
- **Sector Neutrality**: Max 3% net sector exposure
- **Tail Risk**: Gap-down halving, vol spike reduction, consecutive loss protection
- **Drawdown Control**: Scale down at -8%, halt at -3% daily

### Institutional Methodology
- Walk-forward validation with purge gap + embargo
- Sample uniqueness weighting (Lopez de Prado)
- Triple barrier labeling for intraday model
- Stability-based feature selection (IC consistency across periods)

## Usage

### Backtest
```bash
python main.py compare                    # all 3 models + ensemble
python main.py compare --models lightgbm  # single model
python main.py backtest --long-biased     # 14L/7S mode
```

### Production Retrain
```bash
python retrain.py --models crossmamba,lightgbm
```

### Live Trading
```bash
python main.py signal      # generate today's signals
python main.py trade       # paper trade via Alpaca
```

## Automated Retraining

GitHub Actions workflow (`.github/workflows/retrain.yml`) runs every 14 days:
1. Fetches fresh data from Alpaca + yfinance
2. Trains CrossMamba + LightGBM (~10 min on Linux)
3. Verifies pickle compatibility
4. Pushes models to repo

## Configuration

Key settings in `config.py`:
```python
PortfolioConfig:
    max_positions_long: 10      # set dynamically by PM agent
    max_positions_short: 10
    weighting: "risk_parity"

RiskConfig:
    target_annual_vol: 0.10
    hmm_enabled: True           # 3-state HMM regime detection

ModelConfig:
    retrain_every_days: 14      # matches 10-day prediction horizon

FeatureConfig:
    target_type: "risk_adjusted"  # Grinold & Kahn
    max_features: 65
```

## Known Limitations

1. CrossMamba segfaults on macOS ARM (train on Linux via GitHub Actions or Colab)
2. yfinance fundamentals have look-ahead bias (use FMP for production)
3. Backtest period (2021-2026) is mostly bullish
4. OpenBB features require separate installation
