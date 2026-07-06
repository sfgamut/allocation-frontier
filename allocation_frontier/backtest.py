"""Backtest walk-forward rolling, sin lookahead.

Diseño anti-fuga
----------------
La única fuente de información en la fecha de rebalanceo ``t`` es la
ventana ``returns[t−W : t]`` — el slicing termina *estrictamente antes*
de ``t``, así que ni el retorno del propio día de rebalanceo entra a la
estimación. Los pesos decididos en ``t`` se aplican a los retornos de
``[t, t+step)``: primero se decide, después se observa. Cualquier
información futura inyectada más allá de ``t`` no puede alterar los
pesos en ``t`` — y eso es exactamente lo que verifica el test de
no-lookahead con un shock artificial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics
from .optimize import PortfolioResult

# Una estrategia recibe la ventana in-sample (W×N) y devuelve pesos.
Strategy = Callable[[np.ndarray], PortfolioResult]


@dataclass
class BacktestResult:
    name: str
    oos_returns: np.ndarray
    weight_history: np.ndarray
    rebalance_dates: list = field(default_factory=list)

    def summary(self, periods_per_year: int = metrics.TRADING_DAYS) -> dict:
        r = self.oos_returns
        return {
            "strategy": self.name,
            "ann_return": metrics.annualized_return(r, periods_per_year),
            "ann_vol": metrics.annualized_vol(r, periods_per_year),
            "sharpe": metrics.sharpe_ratio(r, periods_per_year=periods_per_year),
            "max_drawdown": metrics.max_drawdown(r),
            "avg_turnover": metrics.average_turnover(self.weight_history),
        }


def walk_forward(
    returns: np.ndarray,
    strategy: Strategy,
    window: int,
    step: int,
    name: str = "strategy",
) -> BacktestResult:
    """Corre una estrategia en walk-forward.

    Parameters
    ----------
    returns : (T, N) retornos simples por período.
    strategy : callable que mapea la ventana in-sample a pesos.
    window : W, largo de la ventana de estimación.
    step : períodos entre rebalanceos (los pesos se mantienen fijos
        dentro del bloque OOS; sin rebalanceo intra-bloque).
    """
    r = np.asarray(returns, dtype=float)
    t_total = len(r)
    if window >= t_total:
        raise ValueError("window >= T: no queda tramo out-of-sample")

    oos: list[np.ndarray] = []
    weights_hist: list[np.ndarray] = []
    dates: list[int] = []

    for t in range(window, t_total, step):
        insample = r[t - window : t]          # ← termina antes de t
        res = strategy(insample)
        w = res.weights
        block = r[t : min(t + step, t_total)]  # ← empieza en t
        oos.append(block @ w)
        weights_hist.append(w)
        dates.append(t)

    return BacktestResult(name, np.concatenate(oos), np.array(weights_hist), dates)


def run_comparison(
    returns: np.ndarray,
    strategies: dict[str, Strategy],
    window: int,
    step: int,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Corre todas las estrategias sobre los mismos datos y devuelve el
    cuadro comparativo ordenado por Sharpe (desc) + los resultados
    crudos para graficar."""
    results = {
        name: walk_forward(returns, strat, window, step, name)
        for name, strat in strategies.items()
    }
    df = (
        pd.DataFrame([res.summary() for res in results.values()])
        .set_index("strategy")
        .sort_values("sharpe", ascending=False)
    )
    return df, results
