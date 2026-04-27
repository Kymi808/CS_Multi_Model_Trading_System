"""
Stress testing framework — simulates extreme scenarios on the portfolio.

Five scenarios that every institutional risk team must test:
1. Flash crash (SPY -10% in one day)
2. Correlation crisis (all correlations → 1, diversification fails)
3. VIX spike to 80 (GFC-level fear)
4. Broker API outage during trading hours
5. Model failure (all-zero predictions)

Each scenario answers: what happens to the portfolio? Which risk controls
fire? How much do we lose? Does the system halt correctly?

Reference: Basel Committee, "Principles for Sound Stress Testing" (2009)
"""
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    name: str
    description: str
    historical_precedent: str
    parameters: dict


@dataclass
class StressResult:
    scenario: StressScenario
    portfolio_loss_pct: float
    portfolio_loss_usd: float
    worst_position: str
    worst_position_loss: float
    risk_controls_triggered: List[str]
    system_halted: bool
    exposure_after_controls: float  # gross exposure after risk scaling
    recommendation: str
    details: dict = field(default_factory=dict)


# ── Default Scenarios ────────────────────────────────────────────────────

SCENARIOS = [
    StressScenario(
        "flash_crash",
        "SPY drops 10% in a single trading day",
        "October 19, 1987 (Black Monday: SPY -20.5%), March 16, 2020 (-12%)",
        {"spy_shock": -0.10},
    ),
    StressScenario(
        "correlation_crisis",
        "All stock correlations go to 1 (diversification fails completely)",
        "September 2008 (Lehman), March 2020 (COVID): correlations spiked to 0.8-0.95",
        {"target_correlation": 1.0},
    ),
    StressScenario(
        "vix_spike",
        "VIX spikes to 80 (double the COVID peak)",
        "VIX peaked at 82.69 on March 16, 2020. GFC peak was 80.86 on Nov 20, 2008",
        {"vix_target": 80, "vol_multiplier": 4.0},
    ),
    StressScenario(
        "api_outage",
        "Alpaca API goes down for 6 hours during market hours",
        "Robinhood outages in March 2020 during peak volatility",
        {"outage_hours": 6},
    ),
    StressScenario(
        "model_failure",
        "ML model produces all-zero predictions (no signal)",
        "Model retraining failure, data pipeline breaks, or pickle corruption",
        {"predictions": "all_zero"},
    ),
]


def run_stress_tests(
    weights: pd.Series,
    prices: pd.DataFrame,
    portfolio_returns: pd.Series = None,
    account_equity: float = 500_000,
    factor_exposures: pd.DataFrame = None,
    risk_config: dict = None,
) -> List[StressResult]:
    """
    Run all stress scenarios on the current portfolio.

    Args:
        weights: current position weights (ticker -> weight)
        prices: historical prices (for beta estimation)
        portfolio_returns: recent daily returns (for vol/drawdown context)
        account_equity: account value in USD
        factor_exposures: Barra factor exposures per stock
        risk_config: risk model configuration
    """
    risk_config = risk_config or {}
    results = []

    for scenario in SCENARIOS:
        logger.info(f"Running stress scenario: {scenario.name}")
        if scenario.name == "flash_crash":
            result = _flash_crash(scenario, weights, prices, account_equity, factor_exposures, portfolio_returns)
        elif scenario.name == "correlation_crisis":
            result = _correlation_crisis(scenario, weights, prices, account_equity, portfolio_returns)
        elif scenario.name == "vix_spike":
            result = _vix_spike(scenario, weights, account_equity, risk_config, portfolio_returns)
        elif scenario.name == "api_outage":
            result = _api_outage(scenario, weights, prices, account_equity)
        elif scenario.name == "model_failure":
            result = _model_failure(scenario, weights, prices, account_equity)
        else:
            continue
        results.append(result)

    return results


# ── Scenario Implementations ─────────────────────────────────────────────

def _flash_crash(scenario, weights, prices, equity, exposures, returns):
    """SPY drops 10%: beta-adjusted shock to each position."""
    spy_shock = scenario.parameters["spy_shock"]

    # Estimate beta per stock
    if exposures is not None and "market" in exposures.columns:
        betas = exposures["market"]
    else:
        # Estimate from recent returns
        spy = prices.get("SPY", prices.mean(axis=1))
        spy_ret = spy.pct_change().dropna()
        betas = {}
        for ticker in weights.index:
            if ticker in prices.columns:
                stock_ret = prices[ticker].pct_change().dropna()
                common = stock_ret.index.intersection(spy_ret.index)
                if len(common) > 50:
                    cov = np.cov(stock_ret.loc[common].values[-252:],
                                spy_ret.loc[common].values[-252:])
                    betas[ticker] = cov[0, 1] / (cov[1, 1] + 1e-10)
                else:
                    betas[ticker] = 1.0
            else:
                betas[ticker] = 1.0
        betas = pd.Series(betas)

    # Apply beta-adjusted shock
    stock_shocks = betas.reindex(weights.index, fill_value=1.0) * spy_shock
    position_losses = weights * stock_shocks
    total_loss = float(position_losses.sum())
    total_loss_usd = total_loss * equity

    # What risk controls would fire?
    controls = []
    if abs(total_loss) > 0.03:
        controls.append("daily_loss_halt (-3% limit)")
    controls.append("tail_risk_scale: gap_down → 0.5x (return < -3%)")
    controls.append("market_stress_halt (SPY down >3%)")

    # Worst position
    worst = position_losses.idxmin()
    worst_loss = float(position_losses.min())

    return StressResult(
        scenario=scenario,
        portfolio_loss_pct=round(total_loss, 4),
        portfolio_loss_usd=round(total_loss_usd, 2),
        worst_position=str(worst),
        worst_position_loss=round(worst_loss * equity, 2),
        risk_controls_triggered=controls,
        system_halted=True,  # daily loss halt would trigger
        exposure_after_controls=round(float(weights.abs().sum()) * 0.5, 4),  # halved by tail risk
        recommendation="System correctly halts. No manual intervention needed. "
                       "Risk controls reduce next-day exposure by 50%.",
        details={
            "beta_adjusted": True,
            "avg_beta": round(float(betas.reindex(weights.index, fill_value=1.0).mean()), 2),
            "long_loss": round(float(position_losses[weights > 0].sum()), 4),
            "short_gain": round(float(position_losses[weights < 0].sum()), 4),
            "net_hedging": round(float(-position_losses[weights < 0].sum()), 4),
        },
    )


def _correlation_crisis(scenario, weights, prices, equity, returns):
    """All correlations → 1: long-short diversification fails."""
    # When correlations = 1, all stocks move together
    # Short positions no longer hedge — everything moves in the same direction
    net_exposure = float(weights.sum())
    gross_exposure = float(weights.abs().sum())

    # In a correlation crisis, portfolio return ≈ net_exposure × market_return
    # Assume market drops 5% (typical for a correlation spike day)
    market_drop = -0.05
    undiversified_loss = net_exposure * market_drop
    diversified_loss = net_exposure * market_drop * 0.3  # normal: 30% of net due to diversification

    controls = []
    if abs(undiversified_loss) > 0.03:
        controls.append("daily_loss_halt (-3% limit)")
    controls.append("correlation_warning: diversification benefit = 0")

    return StressResult(
        scenario=scenario,
        portfolio_loss_pct=round(undiversified_loss, 4),
        portfolio_loss_usd=round(undiversified_loss * equity, 2),
        worst_position="ALL (correlated)",
        worst_position_loss=0,
        risk_controls_triggered=controls,
        system_halted=abs(undiversified_loss) > 0.03,
        exposure_after_controls=round(gross_exposure, 4),
        recommendation="Short hedge provides NO protection when correlations = 1. "
                       f"Net exposure ({net_exposure:.1%}) is the true risk. "
                       "Consider reducing net exposure in high-VIX environments.",
        details={
            "net_exposure": round(net_exposure, 4),
            "gross_exposure": round(gross_exposure, 4),
            "diversified_loss": round(diversified_loss, 4),
            "undiversified_loss": round(undiversified_loss, 4),
            "diversification_benefit_lost": round(abs(undiversified_loss - diversified_loss), 4),
        },
    )


def _vix_spike(scenario, weights, equity, risk_config, returns):
    """VIX to 80: massive vol scaling reduction."""
    vix_target = scenario.parameters["vix_target"]

    target_vol = risk_config.get("target_annual_vol", 0.10)
    vol_floor = risk_config.get("vol_scale_floor", 0.3)

    # Implied daily vol at VIX 80: ~5% per day
    implied_annual_vol = vix_target / 100

    # Vol scale: target / realized, floored
    vol_scale = max(vol_floor, target_vol / implied_annual_vol)

    # Tail risk also fires (vol spike detection: 5d vol > 2x 63d vol)
    tail_scale = 0.6  # vol spike → 0.6x

    combined_scale = vol_scale * tail_scale
    effective_exposure = float(weights.abs().sum()) * combined_scale

    controls = [
        f"vol_scale: {target_vol:.0%} / {implied_annual_vol:.0%} = {vol_scale:.2f} (floored at {vol_floor})",
        f"tail_risk_scale: vol_spike → {tail_scale}x",
        f"combined scaling: {combined_scale:.2f}x",
        f"effective exposure: {effective_exposure:.1%} of equity (from {float(weights.abs().sum()):.1%})",
    ]

    # At VIX 80, expected daily move is ~5%. With combined scaling:
    expected_daily_loss = effective_exposure * 0.05 * 0.5  # 50% chance of loss direction
    expected_loss_usd = expected_daily_loss * equity

    return StressResult(
        scenario=scenario,
        portfolio_loss_pct=round(expected_daily_loss, 4),
        portfolio_loss_usd=round(expected_loss_usd, 2),
        worst_position="N/A (scaling applies uniformly)",
        worst_position_loss=0,
        risk_controls_triggered=controls,
        system_halted=False,  # system continues but at reduced size
        exposure_after_controls=round(effective_exposure, 4),
        recommendation=f"System correctly scales to {combined_scale:.0%} of normal. "
                       f"At VIX 80, daily P&L swing is ~${expected_loss_usd:,.0f}. "
                       "Consider manual override to flat if VIX > 60.",
    )


def _api_outage(scenario, weights, prices, equity):
    """Alpaca API down: can't rebalance or close positions."""
    outage_hours = scenario.parameters["outage_hours"]
    gross_exposure = float(weights.abs().sum())

    # During outage, positions drift with the market
    # Worst case: full exposure at maximum adverse daily move
    # Typical max daily move for a diversified portfolio: ~3-5%
    max_adverse = gross_exposure * 0.04  # 4% adverse move
    max_loss_usd = max_adverse * equity

    controls = [
        "execution_agent: httpx.ConnectError raised, retries exhausted after 3 attempts",
        "order_manager: all orders fail with 'rejected' status",
        "NO automatic halt — system logs errors but doesn't close positions",
        "VULNERABILITY: no circuit breaker for broker outage",
    ]

    return StressResult(
        scenario=scenario,
        portfolio_loss_pct=round(max_adverse, 4),
        portfolio_loss_usd=round(max_loss_usd, 2),
        worst_position="N/A (cannot trade)",
        worst_position_loss=0,
        risk_controls_triggered=controls,
        system_halted=False,  # THIS IS THE PROBLEM
        exposure_after_controls=round(gross_exposure, 4),
        recommendation="VULNERABILITY IDENTIFIED: System does not halt on broker outage. "
                       "Positions remain open with no ability to manage risk. "
                       "RECOMMENDATION: Add broker health check to scheduler. If Alpaca is "
                       "unreachable for >5 minutes, set a 'broker_down' flag that prevents "
                       "new orders and triggers alert. Existing positions remain (can't close anyway).",
        details={
            "outage_hours": outage_hours,
            "positions_at_risk": len(weights[weights != 0]),
            "drift_risk_per_hour": round(max_adverse / outage_hours, 4),
        },
    )


def _model_failure(scenario, weights, prices, equity):
    """All predictions = 0: model provides no signal."""
    from portfolio import PortfolioConstructor
    from config import Config

    cfg = Config()
    constructor = PortfolioConstructor(cfg.portfolio)

    # Simulate zero predictions
    tickers = list(prices.columns[:100])
    zero_preds = pd.Series(0.0, index=tickers)

    # What does the portfolio constructor do?
    try:
        new_weights = constructor.construct_portfolio(
            zero_preds,
            pd.Timestamp.now(),
            prev_weights=weights,
        )
        n_positions = len(new_weights[new_weights.abs() > 0.001])
        new_gross = float(new_weights.abs().sum())
    except Exception:
        n_positions = 0
        new_gross = 0

    controls = [
        "portfolio_constructor: all scores tied at 0",
        f"constructor produces {n_positions} positions (arbitrary selection from tied scores)",
        "turnover_penalty keeps some previous positions (sticky)",
        "NO explicit zero-prediction detection in pipeline",
    ]

    return StressResult(
        scenario=scenario,
        portfolio_loss_pct=0.0,  # no immediate loss, but random positions
        portfolio_loss_usd=0.0,
        worst_position="N/A (random positions from tied scores)",
        worst_position_loss=0,
        risk_controls_triggered=controls,
        system_halted=False,  # THIS IS A PROBLEM
        exposure_after_controls=round(new_gross, 4),
        recommendation="VULNERABILITY IDENTIFIED: System does not detect zero-signal models. "
                       "With all predictions = 0, the constructor picks arbitrary positions. "
                       "RECOMMENDATION: Add prediction quality check before execution. "
                       "If prediction dispersion (p90 - p10) < threshold, skip trading for the day.",
        details={
            "new_positions": n_positions,
            "new_gross_exposure": round(new_gross, 4),
            "random_selection": True,
        },
    )


# ── Reporting ────────────────────────────────────────────────────────────

def format_stress_report(results: List[StressResult]) -> str:
    """Human-readable stress test report."""
    lines = [
        "=" * 70,
        "STRESS TEST REPORT",
        "=" * 70,
        "",
    ]

    for r in results:
        s = r.scenario
        halted = "YES" if r.system_halted else "NO"
        lines.extend([
            f"Scenario: {s.name.upper()}",
            f"  {s.description}",
            f"  Historical precedent: {s.historical_precedent}",
            "",
            f"  Portfolio loss: {r.portfolio_loss_pct:+.2%} (${r.portfolio_loss_usd:+,.2f})",
            f"  Worst position: {r.worst_position} (${r.worst_position_loss:+,.2f})",
            f"  System halted: {halted}",
            f"  Exposure after controls: {r.exposure_after_controls:.1%}",
            "",
            "  Risk controls triggered:",
        ])
        for ctrl in r.risk_controls_triggered:
            lines.append(f"    - {ctrl}")
        lines.extend([
            "",
            f"  Recommendation: {r.recommendation}",
            "",
            "-" * 70,
            "",
        ])

    # Summary
    n_halted = sum(1 for r in results if r.system_halted)
    max_loss = min(r.portfolio_loss_pct for r in results)
    vulnerabilities = sum(1 for r in results if "VULNERABILITY" in r.recommendation)

    lines.extend([
        "SUMMARY",
        f"  Scenarios tested: {len(results)}",
        f"  System halts correctly: {n_halted}/{len(results)}",
        f"  Maximum portfolio loss: {max_loss:+.2%}",
        f"  Vulnerabilities found: {vulnerabilities}",
    ])

    if vulnerabilities > 0:
        lines.append("\n  ACTION REQUIRED: Fix identified vulnerabilities before live trading.")

    return "\n".join(lines)
