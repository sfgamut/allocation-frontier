"""Gráficas del proyecto. Matplotlib, estilo sobrio, figuras a disco."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .backtest import BacktestResult
from .frontier import Frontier

FIGDIR = Path("notebooks/figures")


def _save(fig: plt.Figure, name: str, figdir: Path | str = FIGDIR) -> Path:
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_frontiers(
    classic: Frontier, resampled: Frontier, figdir: Path | str = FIGDIR
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(classic.vols, classic.returns, "o-", lw=1.5, ms=3, label="Clásica")
    ax.plot(resampled.vols, resampled.returns, "s-", lw=1.5, ms=3, label="Resampleada (Michaud)")
    ax.set_xlabel("Volatilidad anual")
    ax.set_ylabel("Retorno esperado anual")
    ax.set_title("Frontera eficiente: clásica vs. resampleada")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "frontier_classic_vs_resampled", figdir)


def plot_covariance_heatmaps(
    s: np.ndarray, c_clean: np.ndarray, figdir: Path | str = FIGDIR
) -> Path:
    """Correlación muestral vs. denoised: el ruido fuera de la estructura
    de bloques se apaga visiblemente tras el clipping."""

    def _corr(m: np.ndarray) -> np.ndarray:
        d = np.sqrt(np.diag(m))
        return m / np.outer(d, d)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title in zip(
        axes, [_corr(s), _corr(c_clean)], ["Correlación muestral", "Denoised (RMT)"]
    ):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8)
    return _save(fig, "covariance_heatmaps", figdir)


def plot_equity_curves(
    results: dict[str, BacktestResult], figdir: Path | str = FIGDIR
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, res in results.items():
        wealth = np.cumprod(1.0 + res.oos_returns)
        ax.plot(wealth, lw=1.3, label=name)
    ax.set_xlabel("Días out-of-sample")
    ax.set_ylabel("Riqueza (base 1)")
    ax.set_title("Equity curves out-of-sample (walk-forward)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "equity_curves_oos", figdir)


def plot_weights(
    results: dict[str, BacktestResult], asset_names: list[str], figdir: Path | str = FIGDIR
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(asset_names)
    x = np.arange(n)
    width = 0.8 / len(results)
    for i, (name, res) in enumerate(results.items()):
        ax.bar(x + i * width, res.weight_history.mean(axis=0), width, label=name)
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(asset_names, rotation=45, fontsize=7)
    ax.set_ylabel("Peso promedio")
    ax.set_title("Pesos promedio por estrategia (todo el walk-forward)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "average_weights", figdir)
