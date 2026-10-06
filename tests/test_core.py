import json
from pathlib import Path

import numpy as np
import pytest

from allocation_frontier.backtest import walk_forward
from allocation_frontier.moments.covariance import LedoitWolfShrinkage
from allocation_frontier.metrics import max_drawdown, sharpe_ratio, spectral_effective_bets
from allocation_frontier.moments.returns import (
    absolute_view,
    black_litterman,
    implied_equilibrium_returns,
)
from allocation_frontier.optimize import PortfolioResult, equal_weight, max_sharpe, min_variance

RNG = np.random.default_rng(0)


# ---------- Black-Litterman ----------

def _toy_sigma(n: int = 4) -> np.ndarray:
    a = RNG.normal(size=(n, n)) * 0.1
    return a @ a.T + 0.02 * np.eye(n)


def test_bl_collapses_to_prior_with_uninformative_views():
    """Con P=I, Q=Π y Ω→∞ el posterior debe colapsar al prior Π."""
    sigma = _toy_sigma()
    n = sigma.shape[0]
    pi = implied_equilibrium_returns(sigma)
    p = np.eye(n)
    omega = np.eye(n) * 1e12  # views con confianza nula
    res = black_litterman(sigma, p, pi, omega=omega)
    assert np.allclose(res.posterior_mean, pi, atol=1e-8)


def test_bl_accepts_singular_sigma_when_view_system_is_defined():
    """La forma Woodbury no exige invertir una Sigma semidefinida."""
    sigma = np.array([[0.04, 0.04], [0.04, 0.04]])  # rango 1
    p = np.array([[1.0, 0.0]])
    q = np.array([0.08])
    omega = np.array([[0.01]])

    res = black_litterman(sigma, p, q, omega=omega)
    assert np.isfinite(res.posterior_mean).all()
    assert np.isfinite(res.posterior_cov).all()
    assert np.allclose(res.posterior_cov, res.posterior_cov.T, atol=1e-12)


def test_bl_view_moves_posterior_toward_q():
    sigma = _toy_sigma()
    n = sigma.shape[0]
    pi = implied_equilibrium_returns(sigma)
    row, val = absolute_view(n, asset=0, value=pi[0] + 0.05)
    res = black_litterman(sigma, row, np.array([val]))
    assert res.posterior_mean[0] > pi[0]  # la view empuja en su dirección
    assert res.posterior_mean[0] < val    # sin llegar a creerla del todo


# ---------- Optimización ----------

def test_weights_sum_to_one_and_long_only():
    sigma = _toy_sigma(6)
    mu = RNG.uniform(0.02, 0.10, 6)
    for res in (min_variance(sigma), max_sharpe(mu, sigma)):
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
        assert (res.weights >= -1e-9).all()


def test_min_variance_two_assets_closed_form():
    """Con dos activos no correlacionados, la solución cerrada es
    w1 = σ2²/(σ1²+σ2²)."""
    s1sq, s2sq = 0.04, 0.09
    sigma = np.diag([s1sq, s2sq])
    res = min_variance(sigma)
    w1_expected = s2sq / (s1sq + s2sq)
    assert np.isclose(res.weights[0], w1_expected, atol=1e-6)


def test_max_sharpe_two_assets_closed_form():
    """Sin correlación y sin rf, w ∝ μ/σ² (tangency sin cortos activos)."""
    mu = np.array([0.08, 0.06])
    sigma = np.diag([0.04, 0.09])
    raw = mu / np.diag(sigma)
    expected = raw / raw.sum()
    res = max_sharpe(mu, sigma)
    assert np.allclose(res.weights, expected, atol=1e-4)


# ---------- Métricas ----------

def test_max_drawdown_hand_computed():
    # riqueza: 1 → 1.10 → 0.88 → 0.968 ; pico 1.10, valle 0.88 → −20%
    r = np.array([0.10, -0.20, 0.10])
    assert np.isclose(max_drawdown(r), -0.20, atol=1e-12)


def test_max_drawdown_counts_loss_from_initial_capital():
    r = np.array([-0.10, 0.05])
    assert np.isclose(max_drawdown(r), -0.10, atol=1e-12)


def test_sharpe_hand_computed():
    r = np.array([0.01, -0.01, 0.01, -0.01])
    mean_ann = 0.0
    assert np.isclose(sharpe_ratio(r), mean_ann, atol=1e-12)


# ---------- Backtest: no-lookahead (CRÍTICO) ----------

def test_backtest_no_lookahead():
    """Inyectar un shock brutal en el futuro no puede cambiar los pesos
    decididos antes del shock. Si este test falla, el backtest está
    fugando información futura y todos sus resultados son inválidos."""
    t, n = 600, 5
    base = RNG.normal(0.0003, 0.01, (t, n))

    shocked = base.copy()
    shocked[450:] += 0.50  # shock absurdo, imposible de no detectar

    def strat(insample: np.ndarray):
        sigma = np.cov(insample, rowvar=False)
        return min_variance(sigma)

    res_base = walk_forward(base, strat, window=250, step=21)
    res_shock = walk_forward(shocked, strat, window=250, step=21)

    for i, t_reb in enumerate(res_base.rebalance_dates):
        if t_reb <= 450:  # decisiones tomadas con ventana previa al shock
            assert np.allclose(
                res_base.weight_history[i], res_shock.weight_history[i], atol=1e-12
            ), f"Fuga de información futura en el rebalanceo t={t_reb}"


def test_backtest_oos_length():
    t, n = 500, 4
    r = RNG.normal(0, 0.01, (t, n))
    res = walk_forward(r, lambda ins: equal_weight(n), window=200, step=20)
    assert len(res.oos_returns) == t - 200


def test_backtest_window_too_large_raises():
    r = RNG.normal(0, 0.01, (100, 3))
    with pytest.raises(ValueError):
        walk_forward(r, lambda ins: equal_weight(3), window=100, step=10)


def test_backtest_weights_drift_inside_oos_block():
    """Sin rebalanceo intra-bloque, un ganador gana peso antes del día siguiente."""
    r = np.zeros((6, 2))
    r[2] = [0.10, 0.0]
    r[3] = [0.10, 0.0]

    res = walk_forward(r, lambda ins: equal_weight(2), window=2, step=2)

    assert np.isclose(res.gross_oos_returns[0], 0.05)

    w_after_first = np.array([0.5 * 1.10, 0.5]) / 1.05
    expected_second = float(w_after_first @ r[3])
    assert np.isclose(res.gross_oos_returns[1], expected_second)
    assert expected_second > 0.05  # prueba que no se restauró 50/50 diariamente

    values_before_second_rebalance = np.array([0.5 * 1.10**2, 0.5])
    expected_pretrade = values_before_second_rebalance / values_before_second_rebalance.sum()
    assert np.allclose(res.pre_trade_weight_history[1], expected_pretrade)

    expected_turnover = np.abs(np.array([0.5, 0.5]) - expected_pretrade).sum() / 2
    assert np.isclose(res.turnover_history[1], expected_turnover)


def test_backtest_transaction_cost_is_pathwise_at_rebalance():
    """El coste real se descuenta cuando ocurre el trade, no ex-post del Sharpe."""
    r = np.zeros((6, 2))
    r[2] = [0.10, 0.0]
    r[3] = [0.10, 0.0]
    rate = 0.01

    res = walk_forward(
        r,
        lambda ins: equal_weight(2),
        window=2,
        step=2,
        transaction_cost_rate=rate,
    )

    turnover = res.turnover_history[1]
    expected_cost = rate * turnover
    assert np.isclose(res.cost_history[1], expected_cost)
    # t=4 tiene retorno gross cero: el retorno neto es exactamente -coste.
    assert np.isclose(res.gross_oos_returns[2], 0.0)
    assert np.isclose(res.oos_returns[2], -expected_cost)
    assert res.oos_returns[2] < res.gross_oos_returns[2]
    summary = res.summary(periods_per_year=252)
    expected_drag = 1.0 - np.prod(1.0 - res.cost_history)
    assert np.isclose(summary["cumulative_rebalance_cost_drag"], expected_drag)
    assert np.isclose(summary["sum_rebalance_cost_fraction"], res.cost_history.sum())


def test_backtest_initial_trade_cost_is_explicit_opt_in():
    r = np.zeros((5, 2))
    res_free = walk_forward(
        r,
        lambda ins: equal_weight(2),
        window=2,
        step=2,
        transaction_cost_rate=0.01,
        charge_initial_trade=False,
    )
    res_paid = walk_forward(
        r,
        lambda ins: equal_weight(2),
        window=2,
        step=2,
        transaction_cost_rate=0.01,
        charge_initial_trade=True,
    )

    assert np.isclose(res_free.turnover_history[0], 0.0)
    assert np.isclose(res_free.oos_returns[0], 0.0)
    assert np.isclose(res_paid.turnover_history[0], 1.0)
    assert np.isclose(res_paid.oos_returns[0], -0.01)


def test_backtest_rejects_invalid_execution_inputs():
    r = np.zeros((10, 2))
    with pytest.raises(ValueError):
        walk_forward(r, lambda ins: equal_weight(2), window=3, step=0)
    with pytest.raises(ValueError):
        walk_forward(r, lambda ins: equal_weight(2), window=3, step=2, transaction_cost_rate=-0.1)

    broken = r.copy()
    broken[5, 0] = -1.0
    with pytest.raises(ValueError):
        walk_forward(broken, lambda ins: equal_weight(2), window=3, step=2)


def test_backtest_initial_trade_uses_gross_notional_for_long_short():
    r = np.zeros((5, 2))

    def long_short(_ins):
        return PortfolioResult(
            weights=np.array([1.5, -0.5]),
            expected_return=np.nan,
            expected_vol=np.nan,
        )

    res = walk_forward(
        r,
        long_short,
        window=2,
        step=2,
        transaction_cost_rate=0.01,
        charge_initial_trade=True,
    )

    assert np.isclose(res.turnover_history[0], 2.0)
    assert np.isclose(res.oos_returns[0], -0.02)


def test_implied_equilibrium_returns_rejects_invalid_market_weights():
    sigma = np.eye(3)
    with pytest.raises(ValueError, match="una entrada por activo"):
        implied_equilibrium_returns(sigma, np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="sumar 1"):
        implied_equilibrium_returns(sigma, np.array([0.2, 0.2, 0.2]))
    with pytest.raises(ValueError, match="pesos negativos"):
        implied_equilibrium_returns(sigma, np.array([0.8, 0.3, -0.1]))
    with pytest.raises(ValueError, match="risk_aversion"):
        implied_equilibrium_returns(sigma, risk_aversion=0.0)


# ---------- Diversificación espectral / paridad SATOR ----------

def test_spectral_effective_bets_equal_independent_assets():
    sigma = np.eye(4)
    w = np.full(4, 0.25)
    assert np.isclose(spectral_effective_bets(w, sigma), 4.0, atol=1e-12)


def test_spectral_effective_bets_perfectly_correlated_assets_is_one():
    sigma = np.ones((4, 4))
    w = np.full(4, 0.25)
    assert np.isclose(spectral_effective_bets(w, sigma), 1.0, atol=1e-10)


def test_spectral_effective_bets_respects_sator_weight_threshold():
    sigma = np.eye(3)
    w = np.array([0.994, 0.003, 0.003])
    assert np.isclose(spectral_effective_bets(w, sigma, min_weight=0.005), 1.0)


def test_sator_allocation_parity_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "allocation_parity.v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    r = np.asarray(fixture["returns"], dtype=float)
    w = np.asarray(fixture["weights"], dtype=float)
    expected = fixture["expected"]

    est = LedoitWolfShrinkage()
    sigma = est.estimate(r)

    assert np.isclose(est.shrinkage_, expected["shrinkage"], atol=1e-12)
    assert np.allclose(sigma, np.asarray(expected["covariance"]), atol=1e-12)

    risk = float(np.sqrt(w @ sigma @ w))
    assert np.isclose(risk, expected["risk"], atol=1e-12)
    assert np.isclose(
        spectral_effective_bets(w, sigma),
        expected["spectral_effective_bets"],
        atol=1e-10,
    )
