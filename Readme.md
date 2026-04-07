# Cross-Sectional Multi-Model Equity Ranking System

## Overview

A quantitative equity ranking system that answers the question: **does diversification across ML architecture types improve cross-sectional stock prediction?**

The system ranks ~460 liquid US equities using three fundamentally different ML architectures — gradient boosting (LightGBM), self-attention (Time Series Transformer), and selective state-space models (CrossMamba) — then passes all predictions through an identical risk, portfolio construction, and transaction cost pipeline. This isolates the effect of model architecture from all other variables.

**Key findings:**

1. With improved feature engineering (risk-adjusted targets, interaction features, expanded universe), individual sequence models dramatically outperform both LightGBM and the ensemble.
2. The ensemble is diluted by LightGBM's weaker signal — TST and CrossMamba alone produce superior risk-adjusted returns.
3. These returns are achieved on a near-neutral portfolio (~10% net exposure), meaning they are driven by stock selection alpha rather than market beta.

### Results (Real Data, 460 Stocks, 10L/10S Neutral, 24bp Costs)

| Model | Annual Return | Sharpe | Sortino | Max Drawdown | Win Rate | Total Return |
|-------|-------------|--------|---------|-------------|----------|-------------|
| **CrossMamba** | 35.38% | 3.393 | 5.310 | -13.66% | 59.25% | 146.62% |
| **TST** | 37.02% | 3.382 | 5.043 | -9.25% | 57.79% | 155.63% |
| Ensemble | 19.53% | 1.878 | 2.608 | -9.84% | 55.13% | 70.19% |
| LightGBM | 13.38% | 1.238 | 1.783 | -10.22% | 55.26% | 45.39% |
| SPY | 20.67% | 1.378 | 1.826 | -18.76% | 57.14% | 74.81% |

LightGBM Avg Rank IC: 0.063 ± 0.141 (IR: 0.44) — 3.3x improvement over the 98-stock baseline of 0.019.

**Caveats:** Fundamental features have look-ahead bias (yfinance). Backtest period (2021-2026) is predominantly bullish. The 3.4 Sharpe will likely degrade 30-40% in live trading.

## Research Questions and Findings

**Q1: Do TST and CrossMamba achieve higher ICs than LightGBM?**
Yes — with improved features (risk-adjusted targets, institutional interactions, expanded universe), the sequence models significantly outperform LightGBM. CrossMamba achieves the highest Sharpe (3.39), TST the lowest drawdown (-9.25%).

**Q2: Does accuracy improvement survive the full pipeline?**
Yes — after Barra-style risk adjustment and realistic transaction costs (24bp round-trip, volatility-dependent), TST and CrossMamba generate 35-37% annual returns on a near-neutral portfolio.

**Q3: Can an ensemble outperform any single model?**
No — in this configuration, the ensemble (1.88 Sharpe) underperforms both TST (3.38) and CrossMamba (3.39) individually. LightGBM dilutes the ensemble. The optimal production configuration uses CrossMamba as primary with TST as secondary.

## Models

### LightGBM (Gradient Boosting)
- **Complexity**: O(n log n) per tree
- **Strengths**: Fast training (seconds), handles tabular data natively, built-in feature importance, GPU-accelerable
- **Architecture**: 3-model ensemble with different random seeds, 800 estimators, max depth 5, 24 leaves

### Time Series Transformer (TST)
- **Complexity**: O(n²) on sequence length (n=21, so effectively negligible)
- **Strengths**: Multi-head self-attention captures cross-time dependencies in rolling feature windows
- **Architecture**: 2-ensemble, 4-head attention, 2 encoder layers, d_model=64, 21-day lookback

### CrossMamba (Selective State-Space Model)
- **Complexity**: O(n) linear-time selective scan
- **Strengths**: Long-range memory without attention's quadratic cost, selective information gating
- **Architecture**: 2-ensemble, 2 SSM blocks, d_model=64, d_state=16, 21-day lookback
- **Note**: Segfaults on macOS ARM. Trains on Linux (GitHub Actions, Colab). GPU optimization via vectorized parallel scan + torch.compile.

## Feature Engineering

200+ raw features, 65 selected via IC stability screening across 3 non-overlapping periods.

**Price/Volume (100 features)**: Momentum at 6 horizons (5d–252d) with skip-1 variants, mean reversion z-scores, distance from highs/lows, Bollinger Bands, realized volatility at 4 windows, volume surprises, Amihud illiquidity, RSI, MACD.

**Fundamental (40 features)**: PE, PB, price-to-sales, EV/EBITDA, earnings yield, ROE, ROA, margins, growth rates, quality/value composites, sector-relative value. All cross-sectionally ranked. Note: yfinance fundamentals have look-ahead bias — FMP integration built but requires API key for point-in-time data.

**Cross-Asset (36 features)**: VIX (level, percentile, changes), yield curve slope, credit spreads (HYG/LQD), dollar/gold/oil momentum, market breadth, sector ETF rotation, risk-on/off indicator.

**Sentiment (10 features)**: Article-level keyword scoring, aggregate/max/min sentiment, article count, positive/negative ratios.

**Insider Trading (8 features)**: Net buying ratio, dollar value, cluster buying count, buy-to-market-cap. Source: FMP or SEC EDGAR.

**FMP Point-in-Time (8 features)**: Earnings estimate revisions, forward/trailing PE ratio, earnings surprise, beat streak. Requires FMP API key ($29/mo).

**OpenBB Alternative Data (12 features)**: Options IV skew, put-call ratio, ATM implied volatility, short interest % of float, days to cover, SI change. Requires OpenBB installation.

**Feature Interactions (20 features)**: 10 institutional cross-factor combinations (each as raw + cross-sectional rank):
Value × Momentum (Asness et al. 2013), Quality × Value (Novy-Marx 2013), Momentum × Low Vol (Frazzini & Pedersen 2014), Size × Value (Fama-French), Earnings × Price Momentum, Sentiment × Momentum, Credit Stress × Beta, Volume × Momentum (Loh 2010), Volatility × Mean Reversion, Momentum Acceleration × Breadth.

**Fractional Differentiation**: Price features at d=0.4 for stationarity with memory preservation (Lopez de Prado, Ch. 5).

### Target Construction
Three variants (configurable via `config.py`):
1. **Risk-adjusted forward return rank** (default): `fwd_return / trailing_vol`, per Grinold & Kahn's Information Ratio concept
2. **Industry-relative forward return rank**: stock return minus sector average, isolates stock-specific alpha
3. **Raw forward return rank**: traditional

## Risk Model

**HMM Regime Detection** (Hamilton 1989): 3-state Gaussian HMM on 5 macro observables. Expanding window, quarterly refits. Probability distribution output. Falls back to multi-speed MA blend.

**GARCH(1,1) Volatility** (Bollerslev 1986): Conditional vol for risk parity. Variance targeting + grid search fit. Captures clustering and mean reversion.

**Barra-Style Factor Model**: 6 factors (market, size, value, momentum, vol, quality). Exponential decay estimation (63-day halflife). Factor exposure clipped ±10%, sector net ±3%.

**Tail Risk**: Gap-down halving, vol spike reduction, consecutive loss protection. Drawdown control at -8%.

**Transaction Costs**: 24bp round-trip (1bp commission + 8bp slippage + 3bp spread), scaled 1–3x by realized vol ratio.

## Walk-Forward Validation

504-day training window, 14-day retrain (matches 10-day horizon), 10-day purge, 5-day embargo. Temporal exponential decay sample weights. ~54 out-of-sample windows over 2021–2026.

## Usage

```bash
python main.py compare                          # all models + ensemble
python main.py compare --models crossmamba      # single model
python main.py backtest --long-biased           # 14L/7S mode
python retrain.py --models crossmamba,lightgbm  # production retrain
python main.py signal                           # today's signals
python main.py trade                            # paper trade
```

## Automated Retraining

GitHub Actions (`.github/workflows/retrain.yml`): runs 1st and 15th monthly on Linux. Trains CrossMamba + LightGBM, verifies pickle compatibility, pushes to repo. Requires `ALPACA_API_KEY`, `ALPACA_API_SECRET` as repository secrets.

## Limitations

1. **Look-ahead bias in fundamentals** — yfinance returns current data for historical dates. FMP integration ready but requires $29/mo key.
2. **Survivorship bias** — current S&P 500 constituents only. Delisted stocks excluded.
3. **Single test period** — 2021–2026, predominantly bullish. No bear market validation.
4. **CrossMamba macOS incompatibility** — segfaults on Apple Silicon, trains on Linux only.
5. **Approximate transaction costs** — not calibrated against actual fills.

## References

- Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere", *Journal of Finance*
- Bollerslev (1986), "Generalized Autoregressive Conditional Heteroskedasticity", *Journal of Econometrics*
- Frazzini & Pedersen (2014), "Betting Against Beta", *Journal of Financial Economics*
- Grinold & Kahn, "Active Portfolio Management", McGraw-Hill
- Hamilton (1989), "A New Approach to Nonstationary Time Series", *Econometrica*
- Lakonishok & Lee (2001), "Are Insider Trades Informative?", *Review of Financial Studies*
- Loh (2010), "Investor Inattention and Stock Recommendations", *Journal of Financial Economics*
- Lopez de Prado (2018), "Advances in Financial Machine Learning", Wiley
- Novy-Marx (2013), "The Other Side of Value", *Journal of Financial Economics*
- Rapach et al. (2016), "Short Interest and Aggregate Stock Returns", *Journal of Financial Economics*
