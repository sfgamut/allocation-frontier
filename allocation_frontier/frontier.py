"""Frontera eficiente clásica y frontera resampleada (Michaud 1998).

La frontera clásica es determinista dado (μ, Σ) — y por eso hereda todo
su error de estimación: dos muestras del mismo proceso generan fronteras
visiblemente distintas, y los portafolios de la punta agresiva son los
más inestables.

La frontera resampleada trata (μ, Σ) como estimaciones con ruido:
se muestrean M escenarios (μ_m, Σ_m) por Monte Carlo paramétrico
(el mismo motor vectorizado del módulo de simulación), se re-optimiza la
frontera completa en cada escenario, y se promedian los pesos *por rango*
(el k-ésimo portafolio de cada frontera con el k-ésimo de las demás).
El promedio sobre escenarios es un suavizado bayesiano de facto: las
posiciones extremas que solo aparecen en algunos escenarios se diluyen,
y sobreviven las asignaciones robustas al error de estimación. El costo
es sesgo hacia la diversificación (la frontera resampleada nunca alcanza
la punta teórica de la clásica — que de todos modos era ilusoria).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optimize import target_return


@dataclass
class Frontier:
    returns: np.ndarray   # (K,)
    vols: np.ndarray      # (K,)
    weights: np.ndarray   # (K, N)


def simulate_moments(
    mu: np.ndarray,
    sigma: np.ndarray,
    t_obs: int,
    n_scenarios: int,
    periods_per_year: int = 252,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo paramétrico vectorizado de (μ, Σ).

    Genera ``n_scenarios`` historias gaussianas de ``t_obs`` períodos con
    los momentos anuales dados y devuelve los momentos re-estimados de
    cada historia: exactamente la distribución de muestreo que la
    frontera clásica ignora.
    """
    rng = rng or np.random.default_rng()
    n = len(mu)
    mu_p = np.asarray(mu, float) / periods_per_year
    sigma_p = np.asarray(sigma, float) / periods_per_year

    chol = np.linalg.cholesky(sigma_p + 1e-12 * np.eye(n))
    z = rng.standard_normal((n_scenarios, t_obs, n))
    paths = mu_p + z @ chol.T  # (M, T, N) — una sola operación vectorizada

    mus = paths.mean(axis=1) * periods_per_year
    centered = paths - paths.mean(axis=1, keepdims=True)
    sigmas = np.einsum("mti,mtj->mij", centered, centered) / (t_obs - 1) * periods_per_year
    return mus, sigmas


def efficient_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_points: int = 25,
    long_only: bool = True,
) -> Frontier:
    """Frontera clásica por barrido de target-return."""
    mu = np.asarray(mu, dtype=float)
    targets = np.linspace(mu.min(), mu.max(), n_points)
    ws, rets, vols = [], [], []
    for tr in targets:
        try:
            res = target_return(mu, sigma, tr, long_only=long_only)
        except RuntimeError:
            continue
        ws.append(res.weights)
        rets.append(res.expected_return)
        vols.append(res.expected_vol)
    return Frontier(np.array(rets), np.array(vols), np.array(ws))


def resampled_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    t_obs: int,
    n_scenarios: int = 200,
    n_points: int = 25,
    long_only: bool = True,
    rng: np.random.Generator | None = None,
) -> Frontier:
    """Frontera resampleada de Michaud: promedio de pesos por rango."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    mus, sigmas = simulate_moments(mu, sigma, t_obs, n_scenarios, rng=rng)

    acc = np.zeros((n_points, len(mu)))
    counts = np.zeros(n_points)
    for m in range(n_scenarios):
        fr = efficient_frontier(mus[m], sigmas[m], n_points, long_only)
        k = len(fr.weights)
        if k == 0:
            continue
        # alinear por rango aunque algún escenario pierda puntos
        idx = np.linspace(0, n_points - 1, k).round().astype(int)
        acc[idx] += fr.weights
        counts[idx] += 1

    valid = counts > 0
    w_avg = acc[valid] / counts[valid, None]
    rets = w_avg @ mu
    vols = np.sqrt(np.einsum("ki,ij,kj->k", w_avg, sigma, w_avg))
    order = np.argsort(rets)
    return Frontier(rets[order], vols[order], w_avg[order])
