"""Métricas out-of-sample."""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def annualized_return(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    return float(r.mean() * periods_per_year)


def annualized_vol(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: np.ndarray, rf: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    vol = annualized_vol(returns, periods_per_year)
    if vol == 0:
        return float("nan")
    return (annualized_return(returns, periods_per_year) - rf) / vol


def max_drawdown(returns: np.ndarray) -> float:
    """Máxima caída pico-a-valle incluyendo el capital inicial como pico.

    Devuelve un número negativo (p. ej. -0.35 = −35%). Incluir riqueza
    inicial 1.0 es esencial: una pérdida en el primer período ya es drawdown.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def average_turnover(weight_history: np.ndarray) -> float:
    """Cambio target-to-target promedio, no turnover ejecutado con weight drift.

    Se conserva por compatibilidad y para estudiar inestabilidad de targets:
    mean_t(sum_i |w_target,t - w_target,t-1|) / 2.

    Para costes de ejecución usa BacktestResult.turnover_history, que compara
    cada target contra los pesos pre-trade realmente derivados.
    """
    w = np.atleast_2d(np.asarray(weight_history, dtype=float))
    if len(w) < 2:
        return 0.0
    return float(np.abs(np.diff(w, axis=0)).sum(axis=1).mean() / 2.0)


def spectral_effective_bets(
    weights: np.ndarray,
    sigma: np.ndarray,
    min_weight: float = 0.005,
) -> float:
    """Número efectivo de apuestas sobre factores PCA de la covarianza.

    Replica la definición usada actualmente por SATOR/TU PORTAFOLIO:

        q_k = lambda_k * (v_k^T w)^2
        p_k = q_k / sum(q)
        ENB = exp(-sum_k p_k log p_k)

    donde (lambda_k, v_k) son autovalores/autovectores de la covarianza
    restringida a activos con peso > min_weight.

    Importante: esto es un ENB espectral/PCA. No se etiqueta como el
    minimum-torsion ENB de Meucci, que usa otra elección de factores.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("sigma debe ser cuadrada")
    if len(w) != s.shape[0]:
        raise ValueError("weights y sigma no tienen la misma dimensión")
    if not np.isfinite(w).all() or not np.isfinite(s).all():
        raise ValueError("weights/sigma contiene NaN/Inf")
    if not np.isfinite(min_weight) or min_weight < 0:
        raise ValueError("min_weight debe ser finito y >= 0")
    if (w < 0).any():
        raise ValueError("spectral_effective_bets espera pesos long-only")
    total_w = float(w.sum())
    if total_w <= 0:
        raise ValueError("weights debe tener suma positiva")
    w = w / total_w

    idx = np.flatnonzero(w > min_weight)
    if len(idx) == 0:
        return 1.0

    ws = w[idx]
    sub = s[np.ix_(idx, idx)]
    sub = 0.5 * (sub + sub.T)
    vals, vecs = np.linalg.eigh(sub)
    if vals.min(initial=0.0) < -1e-10:
        raise ValueError("sigma debe ser PSD")
    vals = np.clip(vals, 0.0, None)

    exposures = vecs.T @ ws
    contributions = vals * exposures**2
    total = float(contributions.sum())
    if total <= 0:
        return 1.0
    p = contributions / total
    nz = p > 1e-12
    return float(np.exp(-np.sum(p[nz] * np.log(p[nz]))))
