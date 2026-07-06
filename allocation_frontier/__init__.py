"""allocation-frontier: optimización de portafolios research-grade.

Estimadores de covarianza (muestral, Ledoit-Wolf, RMT), retornos
(histórico, Black-Litterman), fronteras (clásica, Michaud) y backtest
walk-forward sin lookahead.
"""

from .moments.covariance import (
    LedoitWolfShrinkage,
    RMTDenoisedCovariance,
    SampleCovariance,
    ensure_psd,
)
from .moments.returns import black_litterman, historical_mean, implied_equilibrium_returns
from .optimize import equal_weight, max_sharpe, min_variance, target_return
from .frontier import efficient_frontier, resampled_frontier, simulate_moments
from .backtest import run_comparison, walk_forward
from . import data, metrics, plotting

__all__ = [
    "SampleCovariance", "LedoitWolfShrinkage", "RMTDenoisedCovariance", "ensure_psd",
    "historical_mean", "black_litterman", "implied_equilibrium_returns",
    "min_variance", "max_sharpe", "target_return", "equal_weight",
    "efficient_frontier", "resampled_frontier", "simulate_moments",
    "walk_forward", "run_comparison", "data", "metrics", "plotting",
]
