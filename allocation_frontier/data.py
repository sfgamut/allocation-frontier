"""Carga y preparación de datos de precios.

Dos vías:
* ``load_prices_yfinance`` / ``load_prices_csv`` para datos reales.
* ``synthetic_market`` para un mercado sintético con estructura factorial
  realista (mercado + sectores + idiosincrático), útil para tests,
  desarrollo offline y para validar los estimadores contra una
  covarianza *verdadera* conocida — lujo que los datos reales no dan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Precios → retornos log, alineando fechas y descartando filas
    incompletas (activos con historias desalineadas)."""
    prices = prices.sort_index().dropna(how="any")
    return np.log(prices / prices.shift(1)).dropna(how="any")


def to_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index().dropna(how="any")
    return prices.pct_change().dropna(how="any")


def load_prices_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col)
    return df.sort_index()


def load_prices_yfinance(
    tickers: list[str], start: str, end: str | None = None
) -> pd.DataFrame:
    """Descarga precios ajustados. Requiere red; import diferido para no
    volver yfinance una dependencia dura del núcleo numérico."""
    import yfinance as yf

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    return prices.dropna(how="any")


@dataclass
class SyntheticMarket:
    returns: pd.DataFrame       # T×N retornos simples
    true_mu: np.ndarray         # anual
    true_sigma: np.ndarray      # anual


def synthetic_market(
    n_assets: int = 12,
    n_days: int = 2520,
    n_sectors: int = 3,
    seed: int = 7,
) -> SyntheticMarket:
    """Mercado sintético de 1 factor de mercado + factores sectoriales.

    r_i = β_i·f_mkt + γ_i·f_sector(i) + ε_i, calibrado a magnitudes
    realistas (vol de mercado ~16% anual, idiosincrática 15–30%).
    La estructura factorial produce el espectro típico que RMT explota:
    un autovalor dominante (mercado), unos pocos intermedios (sectores)
    y un bulk de ruido.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS

    beta = rng.uniform(0.7, 1.4, n_assets)
    sector_of = np.arange(n_assets) % n_sectors
    gamma = rng.uniform(0.4, 0.9, n_assets)
    idio_vol = rng.uniform(0.15, 0.30, n_assets)

    mkt_vol, sec_vol = 0.16, 0.10
    mkt_prem = 0.06
    sec_prem = np.linspace(0.02, -0.02, n_sectors)  # dispersión de primas sectoriales

    f_mkt = rng.normal(mkt_prem * dt, mkt_vol * np.sqrt(dt), n_days)
    f_sec = rng.normal(
        sec_prem * dt, sec_vol * np.sqrt(dt), size=(n_days, n_sectors)
    )
    eps = rng.normal(0.0, idio_vol * np.sqrt(dt), size=(n_days, n_assets))

    r = beta * f_mkt[:, None] + gamma * f_sec[:, sector_of] + eps

    # Momentos verdaderos (anuales) implicados por la estructura factorial
    load_sec = np.zeros((n_assets, n_sectors))
    load_sec[np.arange(n_assets), sector_of] = gamma
    true_mu = beta * mkt_prem + load_sec @ sec_prem
    true_sigma = (
        mkt_vol**2 * np.outer(beta, beta)
        + sec_vol**2 * load_sec @ load_sec.T
        + np.diag(idio_vol**2)
    )

    dates = pd.bdate_range("2016-01-04", periods=n_days)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return SyntheticMarket(pd.DataFrame(r, index=dates, columns=cols), true_mu, true_sigma)
