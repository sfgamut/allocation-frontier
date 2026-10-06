"""Demo end-to-end: por qué el Markowitz ingenuo falla fuera de muestra.

Corre el walk-forward comparativo sobre un mercado sintético con
estructura factorial conocida (12 activos, 10 años). Con datos reales:
reemplazá `synthetic_market()` por `data.load_prices_yfinance([...])`
+ `data.to_simple_returns(...)` — el resto del pipeline es idéntico.

Ejecutar:  python notebooks/demo.py   (desde la raíz del repo)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from allocation_frontier import (
    LedoitWolfShrinkage,
    RMTDenoisedCovariance,
    SampleCovariance,
    black_litterman,
    data,
    efficient_frontier,
    equal_weight,
    historical_mean,
    max_sharpe,
    min_variance,
    resampled_frontier,
    run_comparison,
)
from allocation_frontier.moments.returns import relative_view
from allocation_frontier import plotting

FIGDIR = Path(__file__).parent / "figures"

# ---------------------------------------------------------------- datos
# El demo completo es deliberadamente pesado. CI usa el mismo pipeline en modo
# smoke para no convertir un benchmark Monte Carlo en un gate de ~20 minutos.
SMOKE = os.environ.get("ALLOC_FRONTIER_SMOKE") == "1"
N_ASSETS = 12 if SMOKE else 40
N_DAYS = 756 if SMOKE else 2520
N_SECTORS = 3 if SMOKE else 5
mkt = data.synthetic_market(
    n_assets=N_ASSETS, n_days=N_DAYS, n_sectors=N_SECTORS, seed=11
)
R = mkt.returns.to_numpy()
names = list(mkt.returns.columns)
print(f"Mercado sintético: {R.shape[0]} días × {R.shape[1]} activos")

# ------------------------------------------------------- estrategias
SAMPLE, LW, RMT = SampleCovariance(), LedoitWolfShrinkage(), RMTDenoisedCovariance()


def _minvar(cov_est):
    return lambda ins: min_variance(cov_est.estimate(ins))


def _maxsharpe_hist(cov_est):
    def s(ins):
        return max_sharpe(historical_mean(ins), cov_est.estimate(ins) * 252)
    return s


def _maxsharpe_bl(cov_est):
    def s(ins):
        sigma = cov_est.estimate(ins) * 252
        n = sigma.shape[0]
        # view relativa suave de ejemplo: A00 superará a A11 por 2% anual
        p, q = relative_view(n, long=0, short=n - 1, spread=0.02)
        mu = black_litterman(sigma, p, np.array([q])).posterior_mean
        return max_sharpe(mu, sigma)
    return s


strategies = {
    "1/N (equiponderado)": lambda ins: equal_weight(ins.shape[1]),
    "MinVar · muestral": _minvar(SAMPLE),
    "MinVar · Ledoit-Wolf": _minvar(LW),
    "MinVar · RMT": _minvar(RMT),
    "MaxSharpe · hist · muestral": _maxsharpe_hist(SAMPLE),
    "MaxSharpe · hist · Ledoit-Wolf": _maxsharpe_hist(LW),
    "MaxSharpe · hist · RMT": _maxsharpe_hist(RMT),
    "MaxSharpe · BL · Ledoit-Wolf": _maxsharpe_bl(LW),
    "MaxSharpe · BL · RMT": _maxsharpe_bl(RMT),
}

# ------------------------------------------------- walk-forward (OOS)
W, STEP = (126, 21) if SMOKE else (252, 21)  # smoke ~6 meses; demo ~1 año
COST_RATE = 0.0020  # 20 bps por unidad de turnover one-way
table, results = run_comparison(
    R,
    strategies,
    window=W,
    step=STEP,
    transaction_cost_rate=COST_RATE,
)

# ann_return/sharpe ya son NETOS: el coste se paga en cada rebalanceo.
# Conservamos estos aliases para que el CSV sea legible frente a v0.2.
table["ann_return_net"] = table["ann_return"]
table["sharpe_net"] = table["sharpe"]
table = table.sort_values("sharpe", ascending=False)

print("\n=== Cuadro comparativo out-of-sample (costes pathwise, Sharpe neto) ===")
print(table.round(4).to_string())
table.to_csv(FIGDIR.parent / "comparison_oos.csv")

# ------------------------------------------------------------ figuras
ins = R[:W]
mu_hat = historical_mean(ins)
sig_hat = SAMPLE.estimate(ins) * 252
frontier_points = 8 if SMOKE else 25
resample_scenarios = 8 if SMOKE else 120
fr_classic = efficient_frontier(mu_hat, sig_hat, n_points=frontier_points)
fr_resampled = resampled_frontier(
    mu_hat, sig_hat, t_obs=W, n_scenarios=resample_scenarios,
    n_points=frontier_points, rng=np.random.default_rng(1),
)
plotting.plot_frontiers(fr_classic, fr_resampled, FIGDIR)

s_raw = SAMPLE.estimate(ins)
s_clean = RMT.estimate(ins)
plotting.plot_covariance_heatmaps(s_raw, s_clean, FIGDIR)
plotting.plot_equity_curves(results, FIGDIR)
plotting.plot_weights(results, names, FIGDIR)

print(f"\nFiguras guardadas en {FIGDIR}/")
print(f"RMT detectó {RMT.n_signal_} autovalores de señal (esperado: mercado + sectores)")
