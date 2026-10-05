"""Estimadores de retornos esperados.

El retorno esperado es el input *más* ruidoso del problema de Markowitz:
el error estándar de una media con vol anual del 20% y 10 años de datos
es ~6.3% anual — del mismo orden que la prima de riesgo que se intenta
estimar (Merton 1980). Peor: la solución de Markowitz es casi linealmente
sensible a μ, así que ese ruido se transfiere íntegro a los pesos.
Black-Litterman ataca exactamente esto anclando μ a un prior de
equilibrio y dejando que las views lo desplacen solo en proporción a la
confianza declarada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


def historical_mean(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> np.ndarray:
    """Media histórica anualizada.

    Baseline honesto pero ruidoso: es insesgada, pero su varianza de
    muestreo domina cualquier señal en horizontes realistas. Se incluye
    precisamente para que el backtest exhiba ese problema.
    """
    r = np.asarray(returns, dtype=float)
    return r.mean(axis=0) * periods_per_year


@dataclass
class BlackLittermanResult:
    posterior_mean: np.ndarray
    prior_mean: np.ndarray
    posterior_cov: np.ndarray


def implied_equilibrium_returns(
    sigma: np.ndarray,
    market_weights: np.ndarray | None = None,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """Retornos de equilibrio por reverse-optimization: ``Π = δ·Σ·w_mkt``.

    Lógica: si el mercado en agregado sostiene los pesos ``w_mkt``, y el
    agregado optimiza media-varianza con aversión ``δ``, entonces los
    retornos que *racionalizan* esos pesos son Π. Es el μ "que no
    sorprende a nadie": el punto de partida neutral.

    Sin datos de capitalización se usa el portafolio equiponderado como
    proxy de mercado — decisión documentada, no silenciosa: sesga Π
    hacia activos de alta covarianza con el 1/N.
    """
    sigma = np.asarray(sigma, dtype=float)
    n = sigma.shape[0]
    w = np.full(n, 1.0 / n) if market_weights is None else np.asarray(market_weights, float)
    return risk_aversion * sigma @ w


def black_litterman(
    sigma: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray | None = None,
    market_weights: np.ndarray | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> BlackLittermanResult:
    """Posterior de Black-Litterman.

    ``E[R] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]``

    * ``τ`` escala la incertidumbre del prior: (τΣ) es la covarianza del
      *error de estimación* de Π, no del retorno. τ chico ⇒ prior rígido,
      las views casi no mueven μ; τ grande ⇒ el posterior persigue las
      views. Convención de He-Litterman: τ ∈ [0.01, 0.1].
    * ``P`` (K×N) selecciona/combina activos por view; ``Q`` (K,) es el
      valor esperado de cada view. Views absolutas: fila de P con un 1.
      Views relativas: fila con +1/−1 (activo A superará a B por q).
    * ``Ω`` (K×K) es la covarianza del error de las views. Default de
      He-Litterman: ``Ω = diag(P(τΣ)Pᵀ)`` — confianza proporcional a la
      incertidumbre del prior en esa combinación.

    Por qué estabiliza: el posterior es un promedio ponderado por
    precisiones entre Π (ancla de equilibrio) y Q (opinión). El μ
    resultante vive cerca del equilibrio salvo donde hay views con
    confianza explícita, eliminando los extremos espurios de la media
    histórica que Markowitz amplifica en posiciones absurdas.
    """
    sigma = np.asarray(sigma, dtype=float)
    p = np.atleast_2d(np.asarray(p, dtype=float))
    q = np.atleast_1d(np.asarray(q, dtype=float))

    pi = implied_equilibrium_returns(sigma, market_weights, risk_aversion)
    tau_sigma = tau * sigma

    if omega is None:
        omega = np.diag(np.diag(p @ tau_sigma @ p.T))
    omega = np.atleast_2d(np.asarray(omega, dtype=float))

    if p.shape[1] != sigma.shape[0]:
        raise ValueError("P debe tener una columna por activo")
    if len(q) != p.shape[0]:
        raise ValueError("Q debe tener una entrada por view")
    if omega.shape != (p.shape[0], p.shape[0]):
        raise ValueError("Omega debe tener forma (K, K)")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau debe ser finito y > 0")
    if not np.isfinite(sigma).all() or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Black-Litterman recibió valores no finitos")
    if not np.isfinite(omega).all():
        raise ValueError("Omega contiene valores no finitos")

    # Forma algebraicamente equivalente vía Woodbury:
    # M = τΣ
    # μ_post = Π + M P' (P M P' + Ω)^-1 (Q - PΠ)
    # Σ_post = M - M P' (P M P' + Ω)^-1 P M
    #
    # Evita invertir τΣ y Ω por separado. Esto es más estable y, además,
    # admite Σ semidefinida siempre que el sistema en el espacio de views
    # esté bien definido.
    view_system = p @ tau_sigma @ p.T + omega
    try:
        solved = np.linalg.solve(view_system, p @ tau_sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError("el sistema de views de Black-Litterman es singular") from exc

    gain = solved.T
    posterior_mean = pi + gain @ (q - p @ pi)
    posterior_cov = tau_sigma - gain @ p @ tau_sigma
    posterior_cov = (posterior_cov + posterior_cov.T) / 2.0

    return BlackLittermanResult(posterior_mean, pi, posterior_cov)


def absolute_view(n_assets: int, asset: int, value: float) -> tuple[np.ndarray, float]:
    """View absoluta: 'el activo i rendirá `value`'."""
    row = np.zeros(n_assets)
    row[asset] = 1.0
    return row, value


def relative_view(
    n_assets: int, long: int, short: int, spread: float
) -> tuple[np.ndarray, float]:
    """View relativa: 'el activo `long` superará a `short` por `spread`'."""
    row = np.zeros(n_assets)
    row[long], row[short] = 1.0, -1.0
    return row, spread
