import numpy as np
import pytest

from allocation_frontier.backtest import walk_forward
from allocation_frontier.metrics import max_drawdown, sharpe_ratio
from allocation_frontier.moments.returns import (
    absolute_view,
    black_litterman,
    implied_equilibrium_returns,
)
from allocation_frontier.optimize import equal_weight, max_sharpe, min_variance

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
