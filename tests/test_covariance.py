import numpy as np
import pytest
from sklearn.covariance import LedoitWolf as SkLedoitWolf

from allocation_frontier.moments.covariance import (
    LedoitWolfShrinkage,
    RMTDenoisedCovariance,
    SampleCovariance,
    ensure_psd,
)

RNG = np.random.default_rng(42)
EPS = 1e-8


def _factor_returns(t: int = 1000, n: int = 15) -> np.ndarray:
    beta = RNG.uniform(0.5, 1.5, n)
    f = RNG.normal(0, 0.01, t)
    eps = RNG.normal(0, 0.015, (t, n))
    return beta * f[:, None] + eps


ESTIMATORS = [SampleCovariance(), LedoitWolfShrinkage(), RMTDenoisedCovariance()]


@pytest.mark.parametrize("est", ESTIMATORS, ids=lambda e: type(e).__name__)
def test_symmetric_and_psd(est):
    sigma = est.estimate(_factor_returns())
    assert np.allclose(sigma, sigma.T, atol=1e-12)
    assert np.linalg.eigvalsh(sigma).min() >= -EPS


def test_ensure_psd_clamps():
    bad = np.array([[1.0, 0.999], [0.999, 1.0]]) - 1.001 * np.eye(2) * 0.5
    bad[0, 0] = -0.1  # forzar autovalor negativo
    fixed = ensure_psd(bad)
    assert np.linalg.eigvalsh(fixed).min() >= 0


def test_ledoit_wolf_delta_in_unit_interval():
    lw = LedoitWolfShrinkage()
    lw.estimate(_factor_returns())
    assert 0.0 <= lw.shrinkage_ <= 1.0


def test_ledoit_wolf_vs_sklearn_oracle():
    """sklearn contrae hacia identidad escalada (LW 2003); nuestro target
    es correlación constante (LW 2004). No se exige igualdad exacta sino
    coherencia: misma escala de traza y distancia razonable frente al
    tamaño de las matrices."""
    r = _factor_returns(t=2000, n=10)
    ours = LedoitWolfShrinkage().estimate(r)
    theirs = SkLedoitWolf().fit(r).covariance_
    assert np.isclose(np.trace(ours), np.trace(theirs), rtol=0.05)
    rel = np.linalg.norm(ours - theirs) / np.linalg.norm(theirs)
    assert rel < 0.30


def test_rmt_pure_noise_detects_no_signal():
    """Sobre ruido i.i.d. puro, ningún autovalor debe clasificarse como
    señal más allá de fluctuaciones de borde (tolerancia: ≤1 autovalor
    marginal por efectos de muestra finita)."""
    t, n = 2000, 50  # q = 0.025
    noise = RNG.standard_normal((t, n)) * 0.01
    est = RMTDenoisedCovariance(fit_noise_variance=False)
    est.estimate(noise)
    assert est.n_signal_ <= 1


def test_rmt_recovers_signal_in_factor_model():
    r = _factor_returns(t=1500, n=30)
    est = RMTDenoisedCovariance()
    est.estimate(r)
    assert est.n_signal_ >= 1  # el factor de mercado debe sobrevivir


def test_rmt_preserves_total_variance():
    r = _factor_returns()
    s = SampleCovariance().estimate(r)
    den = RMTDenoisedCovariance().estimate(r)
    assert np.isclose(np.trace(den), np.trace(s), rtol=1e-6)
