"""Backtest walk-forward rolling, sin lookahead y con holdings reales.

Diseño anti-fuga
----------------
La única fuente de información en la fecha de rebalanceo t es la ventana
returns[t-W:t]. Los pesos decididos en t se aplican desde t en adelante.

Diseño de ejecución
-------------------
Entre rebalanceos NO se restauran los pesos objetivo. Las posiciones derivan
con los retornos de los activos:

    w_i,t+1 = w_i,t * (1+r_i,t) / (1+r_p,t)

En el rebalanceo siguiente el turnover se calcula contra esos pesos pre-trade,
no contra los targets del rebalanceo anterior.

transaction_cost_rate es el coste proporcional por unidad de turnover one-way,
donde turnover = 0.5 * sum(abs(w_target - w_pretrade)). El coste se paga antes
del retorno del primer período del bloque:

    1 + r_net = (1 - coste) * (1 + r_gross)

La entrada inicial no se cobra por defecto para mantener comparabilidad con
versiones anteriores; charge_initial_trade=True la registra como turnover 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics
from .optimize import PortfolioResult

Strategy = Callable[[np.ndarray], PortfolioResult]


def _empty_matrix() -> np.ndarray:
    return np.empty((0, 0), dtype=float)


def _validate_returns(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[1] < 1:
        raise ValueError("returns debe tener forma (T, N) con N >= 1")
    if not np.isfinite(r).all():
        raise ValueError("returns contiene NaN/Inf")
    if (r <= -1.0).any():
        raise ValueError("un retorno simple de activo no puede ser <= -1")
    return r


def _validate_target(weights: np.ndarray, n_assets: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(w) != n_assets:
        raise ValueError(f"strategy devolvió {len(w)} pesos para {n_assets} activos")
    if not np.isfinite(w).all():
        raise ValueError("strategy devolvió pesos no finitos")
    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError(f"los pesos deben sumar 1; sum={w.sum():.12g}")
    return w.copy()


def _drift_one_period(
    weights: np.ndarray, asset_returns: np.ndarray
) -> tuple[float, np.ndarray]:
    """Retorno gross y pesos al cierre después de una observación."""
    gross = float(weights @ asset_returns)
    if not np.isfinite(gross) or 1.0 + gross <= 0.0:
        raise ValueError("el portafolio perdió 100% o más; no se pueden propagar pesos")
    ending = weights * (1.0 + asset_returns) / (1.0 + gross)
    ending = ending / ending.sum()
    return gross, ending


@dataclass
class BacktestResult:
    name: str
    oos_returns: np.ndarray
    weight_history: np.ndarray
    rebalance_dates: list = field(default_factory=list)
    gross_oos_returns: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    pre_trade_weight_history: np.ndarray = field(default_factory=_empty_matrix)
    ending_weight_history: np.ndarray = field(default_factory=_empty_matrix)
    turnover_history: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    cost_history: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    transaction_cost_rate: float = 0.0
    charge_initial_trade: bool = False

    @property
    def average_rebalance_turnover(self) -> float:
        """Turnover promedio excluyendo la entrada inicial."""
        if len(self.turnover_history) <= 1:
            return 0.0
        return float(np.mean(self.turnover_history[1:]))

    def summary(self, periods_per_year: int = metrics.TRADING_DAYS) -> dict:
        r = self.oos_returns
        gross = self.gross_oos_returns if len(self.gross_oos_returns) else r
        return {
            "strategy": self.name,
            "ann_return": metrics.annualized_return(r, periods_per_year),
            "ann_return_gross": metrics.annualized_return(gross, periods_per_year),
            "ann_vol": metrics.annualized_vol(r, periods_per_year),
            "sharpe": metrics.sharpe_ratio(r, periods_per_year=periods_per_year),
            "sharpe_gross": metrics.sharpe_ratio(gross, periods_per_year=periods_per_year),
            "max_drawdown": metrics.max_drawdown(r),
            "avg_turnover": self.average_rebalance_turnover,
            "initial_turnover": (
                float(self.turnover_history[0]) if len(self.turnover_history) else 0.0
            ),
            "sum_rebalance_cost_fraction": float(np.sum(self.cost_history)),
            "cumulative_rebalance_cost_drag": (
                float(1.0 - np.prod(1.0 - self.cost_history))
                if len(self.cost_history) else 0.0
            ),
        }


def walk_forward(
    returns: np.ndarray,
    strategy: Strategy,
    window: int,
    step: int,
    name: str = "strategy",
    transaction_cost_rate: float = 0.0,
    charge_initial_trade: bool = False,
) -> BacktestResult:
    """Corre una estrategia en walk-forward con posiciones que derivan.

    transaction_cost_rate es el coste proporcional por unidad de turnover
    one-way. 0.002 significa 20 bps por unidad de turnover.
    """
    r = _validate_returns(returns)
    t_total, n_assets = r.shape

    if window < 1:
        raise ValueError("window debe ser >= 1")
    if step < 1:
        raise ValueError("step debe ser >= 1")
    if window >= t_total:
        raise ValueError("window >= T: no queda tramo out-of-sample")
    if not np.isfinite(transaction_cost_rate) or transaction_cost_rate < 0:
        raise ValueError("transaction_cost_rate debe ser finito y >= 0")

    net_oos: list[float] = []
    gross_oos: list[float] = []
    target_hist: list[np.ndarray] = []
    pre_trade_hist: list[np.ndarray] = []
    ending_hist: list[np.ndarray] = []
    turnovers: list[float] = []
    costs: list[float] = []
    dates: list[int] = []

    pre_trade_weights: np.ndarray | None = None

    for t in range(window, t_total, step):
        insample = r[t - window : t]
        target = _validate_target(strategy(insample).weights, n_assets)

        if pre_trade_weights is None:
            pre_trade = np.zeros(n_assets, dtype=float)
            turnover = float(np.abs(target).sum()) if charge_initial_trade else 0.0
        else:
            pre_trade = pre_trade_weights.copy()
            turnover = float(np.abs(target - pre_trade).sum() / 2.0)

        cost_fraction = transaction_cost_rate * turnover
        if cost_fraction >= 1.0:
            raise ValueError("el coste de rebalanceo consume 100% o más del portafolio")

        current = target.copy()
        block = r[t : min(t + step, t_total)]

        for j, asset_r in enumerate(block):
            gross, ending = _drift_one_period(current, asset_r)
            if j == 0 and cost_fraction:
                net = (1.0 - cost_fraction) * (1.0 + gross) - 1.0
            else:
                net = gross

            gross_oos.append(gross)
            net_oos.append(float(net))
            current = ending

        pre_trade_weights = current.copy()

        target_hist.append(target)
        pre_trade_hist.append(pre_trade)
        ending_hist.append(current)
        turnovers.append(turnover)
        costs.append(cost_fraction)
        dates.append(t)

    return BacktestResult(
        name=name,
        oos_returns=np.asarray(net_oos, dtype=float),
        weight_history=np.asarray(target_hist, dtype=float),
        rebalance_dates=dates,
        gross_oos_returns=np.asarray(gross_oos, dtype=float),
        pre_trade_weight_history=np.asarray(pre_trade_hist, dtype=float),
        ending_weight_history=np.asarray(ending_hist, dtype=float),
        turnover_history=np.asarray(turnovers, dtype=float),
        cost_history=np.asarray(costs, dtype=float),
        transaction_cost_rate=float(transaction_cost_rate),
        charge_initial_trade=charge_initial_trade,
    )


def run_comparison(
    returns: np.ndarray,
    strategies: dict[str, Strategy],
    window: int,
    step: int,
    transaction_cost_rate: float = 0.0,
    charge_initial_trade: bool = False,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Corre estrategias sobre idénticos datos y convenciones de ejecución."""
    results = {
        name: walk_forward(
            returns,
            strat,
            window,
            step,
            name,
            transaction_cost_rate=transaction_cost_rate,
            charge_initial_trade=charge_initial_trade,
        )
        for name, strat in strategies.items()
    }
    df = (
        pd.DataFrame([res.summary() for res in results.values()])
        .set_index("strategy")
        .sort_values("sharpe", ascending=False)
    )
    return df, results
