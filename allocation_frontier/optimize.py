"""Optimización media-varianza.

Se usa ``scipy.optimize.minimize`` (SLSQP) en lugar de cvxpy: los tres
problemas (min-var, max-Sharpe vía target-return, target-return) son QP
chicos con restricciones lineales y de caja, donde SLSQP converge en
milisegundos sin arrastrar un solver cónico completo como dependencia.
Si el proyecto incorporara restricciones de cardinalidad o cónicas
(tracking error, CVaR), cvxpy pasaría a justificarse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .moments.covariance import ensure_psd


@dataclass
class PortfolioResult:
    weights: np.ndarray
    expected_return: float
    expected_vol: float

    @property
    def sharpe(self) -> float:
        return self.expected_return / self.expected_vol if self.expected_vol > 0 else np.nan


def _prepare_sigma(sigma: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Regularización ridge opcional para Σ mal condicionada:
    ``Σ + ridge·mean(diag(Σ))·I`` — desplaza todo el espectro sin tocar
    las direcciones propias."""
    sigma = ensure_psd(sigma)
    if ridge > 0:
        sigma = sigma + ridge * np.mean(np.diag(sigma)) * np.eye(sigma.shape[0])
    return sigma


def _solve(
    mu: np.ndarray | None,
    sigma: np.ndarray,
    objective,
    long_only: bool,
    target_return: float | None = None,
) -> np.ndarray:
    n = sigma.shape[0]
    w0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if target_return is not None and mu is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target_return})

    res = minimize(
        objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"SLSQP no convergió: {res.message}")
    return res.x


def min_variance(
    sigma: np.ndarray, long_only: bool = True, ridge: float = 0.0
) -> PortfolioResult:
    sigma = _prepare_sigma(sigma, ridge)
    w = _solve(None, sigma, lambda w: w @ sigma @ w, long_only)
    vol = float(np.sqrt(w @ sigma @ w))
    return PortfolioResult(w, np.nan, vol)


def target_return(
    mu: np.ndarray, sigma: np.ndarray, target: float,
    long_only: bool = True, ridge: float = 0.0,
) -> PortfolioResult:
    mu = np.asarray(mu, dtype=float)
    sigma = _prepare_sigma(sigma, ridge)
    w = _solve(mu, sigma, lambda w: w @ sigma @ w, long_only, target_return=target)
    return PortfolioResult(w, float(w @ mu), float(np.sqrt(w @ sigma @ w)))


def max_sharpe(
    mu: np.ndarray, sigma: np.ndarray, rf: float = 0.0,
    long_only: bool = True, ridge: float = 0.0,
) -> PortfolioResult:
    """Max-Sharpe por minimización directa del Sharpe negativo.

    El problema no es convexo en w, pero sobre el simplex (long-only,
    suma 1) es cuasi-cóncavo y SLSQP con arranque 1/N lo resuelve de
    forma robusta; se verifica contra la solución cerrada en tests.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = _prepare_sigma(sigma, ridge)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(w @ sigma @ w)
        return -(w @ mu - rf) / vol if vol > 0 else 1e9

    w = _solve(mu, sigma, neg_sharpe, long_only)
    return PortfolioResult(w, float(w @ mu), float(np.sqrt(w @ sigma @ w)))


def equal_weight(n: int) -> PortfolioResult:
    """1/N. La vara de medir (DeMiguel, Garlappi y Uppal 2009)."""
    w = np.full(n, 1.0 / n)
    return PortfolioResult(w, np.nan, np.nan)
