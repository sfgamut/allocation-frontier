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
