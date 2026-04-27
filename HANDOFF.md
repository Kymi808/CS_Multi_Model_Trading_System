# CS Multi-Model Trading System Handoff

## Role In The Duo Repo

This repository is the model and signal-generation dependency for
`openclaw-fintech`. OpenClaw owns scheduling, approvals, execution, alerting, and
operator workflows. This repo owns model classes, trained artifacts, feature
generation, retraining, and standalone research/backtest commands.

OpenClaw expects this repo to be mounted or cloned and referenced through:

```bash
export CS_SYSTEM_PATH=/absolute/path/to/CS_Multi_Model_Trading_System
```

For Dockerized OpenClaw scheduled trading:

```bash
CS_SYSTEM_PATH_HOST=/absolute/path/to/CS_Multi_Model_Trading_System \
  docker compose -f docker-compose.yml -f docker-compose.models.yml \
  up -d --build trading-scheduler
```

## Required Shipped Artifacts

Keep these artifacts versioned because OpenClaw loads them directly:

- `models/latest_lightgbm_model.pkl`
- `models/latest_crossmamba_model.pkl`
- `models/latest_tst_model.pkl`
- `models/latest_model.pkl`

Generated data caches, logs, presentation exports, and ad hoc backtest outputs are
not part of the runtime contract. Regenerate them when needed.

## Fresh Setup

```bash
git clone https://github.com/Kymi808/CS_Multi_Model_Trading_System.git
cd CS_Multi_Model_Trading_System
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with Alpaca paper keys before running live data or trade commands.
Install `requirements-training.txt` only for CrossMamba/TST retraining. The base
runtime intentionally omits Torch so Docker and OpenClaw inference stay lean.

## Release Gates

Run before pushing handoff changes:

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest
python -m compileall *.py alpaca_adapter models tests
python main.py --help
python tests/smoke_openclaw_contract.py
docker build -t cs-multi-model:handoff .
docker run --rm cs-multi-model:handoff
```

GitHub Actions runs the same core checks on push and pull request.

## Production Rules

- Use `TRADING_ENV=paper` for scheduled paper trading.
- Do not set live credentials or live endpoint URLs without an explicit risk review.
- Do not commit `.env`, local DBs, API output caches, logs, or generated result sweeps.
- CrossMamba/TST are Linux-oriented. Use LightGBM for Apple Silicon local inference.
- OpenClaw should mount this repo read-only in Docker.

## Known Limits

- The published backtests are research evidence, not a live track record.
- Fundamental data quality depends on FMP availability for point-in-time fields.
- Transaction-cost and slippage assumptions must be recalibrated from paper fills.
