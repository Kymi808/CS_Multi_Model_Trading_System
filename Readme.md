# Cross-Sectional Ranking System (v4)

A quantitative long/short equity system that ranks S&P 500 stocks using machine learning and trades via Alpaca.

## How It Works

The system predicts which stocks will outperform or underperform over the next 10 trading days using a LightGBM ensemble trained on 50 stability-selected features. It goes long the top-ranked stocks and shorts the bottom-ranked, with a structural long bias to capture market returns.

**Core pipeline:**
1. Fetch prices, fundamentals, sentiment, and cross-asset data for 102 S&P 500 stocks
2. Engineer 186 features (momentum, volatility, value, quality, earnings, cross-asset)
3. Select 50 features with stable predictive power across multiple time periods
4. Train an ensemble of 3 LightGBM models via walk-forward validation
5. Rank all stocks, go long the top 14 and short the bottom 7
6. Apply risk management: factor neutralization, vol targeting, drawdown control, regime overlay
7. Execute via Alpaca (paper or live)

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

### 1. Train the Model (run once)

```bash
python main.py backtest --long-biased --n-long 14 --n-short 7 --optimize
```

This runs the full pipeline: data fetch → feature engineering → feature selection → Optuna hyperparameter optimization → walk-forward training → backtest simulation. Takes ~5 minutes. Saves the trained model to `models/latest_model.pkl` and Optuna params to `results/optuna_best_params.json`.

**Don't run `--optimize` again** unless you want to retrain — it overwrites your saved model.

### 2. View Today's Signals

```bash
python main.py signal --long-biased --n-long 14 --n-short 7
```

Shows current portfolio recommendations without trading. Useful for reviewing before executing.

### 3. Paper Trade

```bash
python main.py trade --long-biased --n-long 14 --n-short 7
```

Generates signals and executes via Alpaca paper trading. Shows the full trade plan and asks for confirmation before executing. Run during market hours (9:30 AM – 4:00 PM ET).

### 4. Rebalance

Run the trade command **once per week** (e.g., every Monday morning). The system compares your current Alpaca positions to the new target portfolio and only trades the difference. Clear the data cache first to fetch fresh data:

```bash
rm -rf data/
python main.py trade --long-biased --n-long 14 --n-short 7
```

### 5. Live Trade (caution)

```bash
python main.py trade --long-biased --n-long 14 --n-short 7 --live
```

Requires typing `CONFIRM` before executing. Paper trade for at least 2–3 months first.

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
main.py                 CLI entry point, argument parsing
├── backtest.py         Walk-forward backtest engine
├── signal_generator.py Signal generation for live trading (same pipeline as backtest)
├── config.py           All configuration parameters
├── universe.py         S&P 500 universe construction
├── data_loader.py      Price, fundamental, cross-asset data fetching (yfinance)
├── features.py         Price/volume feature engineering (100 features)
├── fundamental_features.py  Earnings, value, quality signals (40 features)
├── cross_asset_features.py  Rates, VIX, sector ETF signals (36 features)
├── sentiment_features.py    News sentiment via yfinance (10 features)
├── model.py            LightGBM ensemble + walk-forward training
├── optuna_tuner.py     Hyperparameter optimization
├── portfolio.py        Portfolio construction (risk parity, turnover control)
├── risk_model.py       Barra-style factor risk model + regime overlay
└── execution.py        Alpaca order execution
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
| `models/latest_model.pkl` | Trained ensemble model |
| `results/optuna_best_params.json` | Optimized hyperparameters |
| `results/backtest_performance.png` | Equity curve plot |
| `results/latest_signals.csv` | Most recent signal output |
| `results/trades_*.json` | Trade execution logs |
| `data/` | Cached price/fundamental data (clear before rebalancing) |
