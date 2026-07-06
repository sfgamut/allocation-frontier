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
    """Máxima caída pico-a-valle de la curva de riqueza compuesta.
    Devuelve un número negativo (p. ej. -0.35 = −35%)."""
    r = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def average_turnover(weight_history: np.ndarray) -> float:
    """Turnover one-way promedio por rebalanceo:
    ``mean_t( Σ_i |w_{t,i} − w_{t−1,i}| ) / 2``.

    Proxy de costo de transacción: dos estrategias con el mismo Sharpe
    bruto y turnover 5× distinto no son equivalentes netas de costos.
    """
    w = np.atleast_2d(np.asarray(weight_history, dtype=float))
    if len(w) < 2:
        return 0.0
    return float(np.abs(np.diff(w, axis=0)).sum(axis=1).mean() / 2.0)
