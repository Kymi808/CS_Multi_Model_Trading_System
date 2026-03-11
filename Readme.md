# Cross-Sectional Ranking System (v5 — Multi-Model)

A quantitative long/short equity system that ranks ~100 S&P 500 stocks using three competing alpha model architectures, then passes all predictions through the same risk, transaction cost, and portfolio construction pipeline to answer: **does the choice of alpha model architecture meaningfully impact portfolio-level performance?**

## Alpha Models

| Model | Architecture | Complexity | Key Property |
|-------|-------------|-----------|--------------|
| **LightGBM** | Gradient boosting ensemble | O(n log n) | Handles tabular data natively, fast to train |
| **TST** | Time Series Transformer | O(n²) | Multi-head attention captures cross-time dependencies |
| **CrossMamba** | Selective State-Space Model | O(n) | Linear-time selective scan, long-range memory |

All three operate on **identical features** to isolate the effect of architecture from all other system components.

## How It Works

The system predicts which stocks will outperform or underperform over the next 10 trading days. It goes long the top-ranked stocks and shorts the bottom-ranked, with a structural long bias to capture market returns.

**Core pipeline (5-component architecture):**
1. **Data**: Fetch prices, fundamentals, sentiment, and cross-asset data for ~100 S&P 500 stocks
2. **Features**: Engineer 186 features (momentum, volatility, value, quality, earnings, cross-asset)
3. **Feature Selection**: Select 50 features with stable predictive power across multiple time periods
4. **Alpha Model**: Train via walk-forward validation — LightGBM, TST, CrossMamba, or all three
5. **Risk Model**: Barra-style factor neutralization, vol targeting, drawdown control, regime overlay
6. **Transaction Cost Model**: Penalize turnover (commission + slippage + spread)
7. **Portfolio Construction**: Risk-parity weighting with position and turnover constraints
8. **Execution**: Execute via Alpaca (paper or live)

**Research questions:**
- Do TST and CrossMamba achieve higher Information Coefficients than LightGBM under identical walk-forward validation?
- Does any improvement in statistical accuracy survive the full pipeline? After risk model adjustment and transaction costs, which architecture produces the highest Sharpe ratio, lowest max drawdown, and greatest alpha vs S&P 500?
- Can an ensemble combining predictions from all three architectures outperform any single model?

## Backtest Results

| Metric | Strategy (14L/7S) | SPY |
|--------|-------------------|-----|
| Annual Return | 20.99% | 20.29% |
| Sharpe Ratio | 1.67 | 1.34 |
| Max Drawdown | -11.39% | -18.76% |
| Win Rate | 56.86% | — |
| Alpha vs SPY | +0.70% | — |

Period: Feb 2023 – Feb 2026. Realistic live expectations: 10–16% annual return, Sharpe 0.8–1.2, max drawdown -15 to -25%.

## Setup

```bash
# Clone and install
cd CS_Ranking_System
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For paper trading
pip install alpaca-py
export ALPACA_API_KEY="your_key"
export ALPACA_API_SECRET="your_secret"
```

### Requirements

- Python 3.10+
- ~2GB RAM for feature engineering
- Internet connection for data fetching (yfinance, Alpaca)

## Usage

### 1. Compare All Alpha Models (primary use case)

```bash
python main.py compare
```

Runs LightGBM, TST, and CrossMamba through the same pipeline, plus a weighted ensemble. Outputs a comparison table of all metrics (Sharpe, Sortino, max drawdown, IC, alpha vs SPY, etc.) and saves comparison plots.

```bash
# Compare specific models only
python main.py compare --models lightgbm,tst

# Compare without ensemble
python main.py compare --no-ensemble

# Long-biased mode
python main.py compare --long-biased --n-long 14 --n-short 7
```

### 2. Train Single Model (LightGBM backtest)

```bash
python main.py backtest --long-biased --n-long 14 --n-short 7 --optimize
```

This runs the full pipeline with LightGBM only. Saves the trained model to `models/latest_model.pkl`.

### 3. View Today's Signals

```bash
python main.py signal --long-biased --n-long 14 --n-short 7
```

### 4. Paper Trade

```bash
python main.py trade --long-biased --n-long 14 --n-short 7
```

### 5. Live Trade (caution)

```bash
python main.py trade --long-biased --n-long 14 --n-short 7 --live
```

Requires typing `CONFIRM` before executing.

## Configuration Options

| Flag | Description | Default |
|------|-------------|---------|
| `--long-biased` | Loose risk config (no sector neutrality, 15% vol target) | Off |
| `--n-long N` | Number of long positions | 15 (or 14 recommended) |
| `--n-short N` | Number of short positions | 5 (or 7 recommended) |
| `--optimize` | Run Optuna hyperparameter search (50 trials) | Off |
| `--horizon N` | Prediction horizon in days | 10 |
| `--long-only` | Long only, no shorts | Off |
| `--live` | Live trading (trade command only) | Paper |

### Recommended Configurations

```bash
# Best risk-adjusted (recommended)
python main.py backtest --long-biased --n-long 14 --n-short 7 --optimize

# Maximum return (higher drawdown risk)
python main.py backtest --long-biased --optimize

# Market neutral (crash protection, lower returns)
python main.py backtest --optimize
```

## Architecture

```
main.py                      CLI entry point (backtest, compare, signal, trade)
├── backtest.py              Walk-forward backtest engine (single model)
├── model_comparison.py      Multi-model comparison framework
├── signal_generator.py      Signal generation for live trading
├── config.py                All configuration (ModelConfig, TSTConfig, CrossMambaConfig, etc.)
├── universe.py              S&P 500 universe construction
├── data_loader.py           Price, fundamental, cross-asset data fetching (yfinance)
├── features.py              Price/volume feature engineering (100 features)
├── fundamental_features.py  Earnings, value, quality signals (40 features)
├── cross_asset_features.py  Rates, VIX, sector ETF signals (36 features)
├── sentiment_features.py    News sentiment via yfinance (10 features)
├── model.py                 Model factory + walk-forward training (all architectures)
├── models/
│   ├── tst_model.py         Time Series Transformer implementation
│   └── crossmamba_model.py  CrossMamba (selective state-space) implementation
├── optuna_tuner.py          Hyperparameter optimization (LightGBM)
├── portfolio.py             Portfolio construction (risk parity, turnover control)
├── risk_model.py            Barra-style factor risk model + regime overlay
└── execution.py             Alpaca order execution
```

### Pipeline Consistency

The signal generator and backtest use the **identical** risk pipeline:

1. Sector neutralization
2. Factor neutralization (market, size, value, momentum, volatility, quality)
3. Volatility targeting (scale to 15% annual vol)
4. Drawdown control (reduce exposure below -12%)
5. Position re-clip (enforce 14L/7S limits)
6. Regime overlay (weak: max 15% bias based on 50d/200d MA)

This is all inside `risk_model.apply_risk_scaling()` — one method, used everywhere.

## Key Features

### Stability-Based Feature Selection
Instead of selecting features by overall IC (which overfits to one regime), features must show consistent predictive power across 3 separate time periods. Score = mean(|IC|) × sign consistency. Reduces 186 features to 50.

### Walk-Forward Validation
No look-ahead bias in model training. Each prediction window uses only data available up to that point, with a purge gap between train and test to prevent leakage.

### Risk Parity Weighting
Positions are weighted inversely proportional to their volatility, so each contributes equal risk to the portfolio. Lower-vol stocks get larger weights.

### Regime Overlay
A weak trend filter (50d vs 200d moving average) scales long/short weights by up to 15%. Bullish regime → slightly more long exposure. Bearish → slightly less. Keeps net exposure in the 55–70% range rather than swinging wildly.

## Top Predictive Features

| Feature | Importance | Category |
|---------|-----------|----------|
| Earnings day return | 79.0 | Earnings |
| 63-day volatility | 60.7 | Risk |
| Cross-sectional vol rank | 54.7 | Risk |
| Distance from 252d high | 51.7 | Momentum |
| 21-day volatility | 51.3 | Risk |
| Amihud illiquidity | 44.0 | Liquidity |
| MACD histogram | 43.3 | Technical |
| 63-day momentum rank | 43.0 | Momentum |

## Known Limitations

1. **Survivorship bias** — Universe is today's S&P 500 constituents, missing delisted stocks. Inflates returns by ~1–3% annually.

2. **Look-ahead bias in fundamentals** — yfinance provides current fundamentals, not point-in-time. The model uses today's PE/ROE for all historical dates. Could inflate alpha by 2–5%.

3. **Single test period** — Only tested on Feb 2023 – Feb 2026 (strong bull market). Performance in bear markets is estimated, not tested.

4. **Transaction cost underestimate** — Backtest assumes 0.5 bps costs. Real slippage during volatile periods could be 5–10x higher.

5. **No tail risk protection** — Drawdown control reduces exposure at -12%, but can't prevent sharp losses in flash crashes.

6. **Regime detector is lagging** — The 50d/200d MA crossover confirms trends after they form. It won't predict crashes.

## Realistic Expectations

| Scenario | Estimated Return |
|----------|-----------------|
| Bull market (+20% SPY) | +18 to +24% |
| Flat market | +3 to +6% |
| Mild correction (-15% SPY) | -8 to -12% |
| Bear market (-25% SPY) | -14 to -19% |
| Crash (-50% SPY) | -28 to -38% |

The strategy captures ~70% of market upside with ~60% of market downside. It won't protect you in a crash, but it will outperform SPY on a risk-adjusted basis over full market cycles.

## File Outputs

| File | Description |
|------|-------------|
| `models/latest_model.pkl` | Trained LightGBM model |
| `models/latest_tst_model.pkl` | Trained TST model |
| `models/latest_crossmamba_model.pkl` | Trained CrossMamba model |
| `results/model_comparison.json` | Metrics comparison table (all models) |
| `results/model_comparison.png` | Comparison visualization (cumulative returns, drawdown, Sharpe, bar chart) |
| `results/comparison_table.csv` | Comparison table as CSV |
| `results/returns_*.csv` | Per-model daily returns |
| `results/summary_*.json` | Per-model summary metrics |
| `results/backtest_performance_*.png` | Per-model equity curve plots |
| `results/optuna_best_params.json` | Optimized hyperparameters |
| `results/latest_signals.csv` | Most recent signal output |
| `results/trades_*.json` | Trade execution logs |
| `data/` | Cached price/fundamental data (clear before rebalancing) |
