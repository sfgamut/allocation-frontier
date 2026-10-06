"""Estimadores de la matriz de covarianza.

Por qué esto es el núcleo del problema
--------------------------------------
La covarianza muestral ``S`` es el estimador de máxima verosimilitud bajo
normalidad, pero para portafolios es un estimador *malo* fuera de muestra.
Con N activos y T observaciones, ``S`` tiene O(N²) parámetros libres
estimados con N·T datos: cuando ``q = N/T`` no es despreciable, los
autovalores de ``S`` se dispersan artificialmente alrededor de los
verdaderos (los chicos se subestiman, los grandes se sobreestiman).
El optimizador de Markowitz es una máquina de maximizar error: asigna
peso máximo justo a las direcciones donde la covarianza fue peor estimada
(autovalores espuriamente chicos → varianza aparente casi nula).

Los dos remedios implementados atacan el mismo mal por caminos distintos:

* **Ledoit-Wolf** contrae ``S`` hacia un target estructurado ``F`` de
  pocos parámetros (correlación constante). El trade-off sesgo-varianza
  se resuelve analíticamente: ``δ*`` minimiza la pérdida esperada de
  Frobenius entre el estimador combinado y la covarianza verdadera.
* **RMT / Marchenko-Pastur** usa el resultado exacto de matrices
  aleatorias: si los retornos fueran ruido i.i.d. puro, los autovalores
  de la correlación muestral caerían en ``[λ₋, λ₊]`` con
  ``λ± = σ²(1 ± √q)²``. Todo autovalor dentro de esa banda es
  estadísticamente indistinguible de ruido y se colapsa a su promedio
  (clipping que preserva la traza); los que la exceden son señal
  (mercado, sectores) y se conservan intactos.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

_EPS_PSD = 1e-10


class CovarianceEstimator(Protocol):
    """Interfaz común: recibe retornos T×N, devuelve covarianza N×N."""

    def estimate(self, returns: np.ndarray) -> np.ndarray: ...


def _validate_returns(returns: np.ndarray, *, min_obs: int = 2) -> np.ndarray:
    """Valida la matriz T×N antes de cualquier estimación."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[1] < 1:
        raise ValueError("returns debe tener forma (T, N) con N >= 1")
    if r.shape[0] < min_obs:
        raise ValueError(f"se requieren al menos {min_obs} observaciones")
    if not np.isfinite(r).all():
        raise ValueError("returns contiene NaN/Inf")
    return r


def _require_positive_variances(var: np.ndarray, *, estimator: str) -> None:
    bad = np.flatnonzero(~np.isfinite(var) | (var <= 0.0))
    if len(bad):
        indices = ", ".join(map(str, bad.tolist()))
        raise ValueError(
            f"{estimator} requiere varianza positiva por activo; "
            f"activos constantes/no válidos en índices: {indices}"
        )


def _symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def ensure_psd(sigma: np.ndarray, eps: float = _EPS_PSD) -> np.ndarray:
    """Proyecta a la matriz PSD más cercana clampando autovalores.

    La reconstrucción espectral (RMT) y la aritmética en float64 pueden
    producir autovalores levemente negativos (~ -1e-14). Clampearlos a
    ``eps`` es un ajuste numérico documentado, no una corrección
    estadística: la magnitud del clamp es órdenes por debajo del ruido
    de estimación.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma debe ser una matriz cuadrada")
    if not np.isfinite(sigma).all():
        raise ValueError("sigma contiene NaN/Inf")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps debe ser finito y >= 0")
    sigma = _symmetrize(sigma)
    vals, vecs = np.linalg.eigh(sigma)
    if vals.min() >= eps:
        return sigma
    vals = np.clip(vals, eps, None)
    return _symmetrize((vecs * vals) @ vecs.T)


class SampleCovariance:
    """Covarianza muestral (insesgada, ddof=1). Base de comparación."""

    def estimate(self, returns: np.ndarray) -> np.ndarray:
        r = _validate_returns(returns)
        sigma = np.atleast_2d(np.cov(r, rowvar=False, ddof=1))
        return _symmetrize(sigma)


class LedoitWolfShrinkage:
    """Shrinkage hacia target de correlación constante (Ledoit-Wolf 2004,
    "Honey, I Shrunk the Sample Covariance Matrix").

    Estimador: ``Σ = δ*·F + (1−δ*)·S`` donde ``F`` comparte las varianzas
    de ``S`` pero reemplaza todas las correlaciones por su promedio
    ``r̄``. La intensidad óptima es ``δ* = clip(κ/T, 0, 1)`` con
    ``κ = (π − ρ)/γ``:

    * ``π``: suma de varianzas asintóticas de las entradas de ``S``
      (cuánto ruido de muestreo hay que matar),
    * ``ρ``: covarianza entre el error de ``S`` y el error de ``F``
      (la parte del ruido que el target hereda y no se puede eliminar),
    * ``γ``: distancia Frobenius²  entre ``F`` y ``S``
      (cuánto sesgo se paga por contraer).

    Implementación propia según el paper; sklearn se usa solo como
    oráculo en tests (sklearn contrae hacia un target distinto —
    identidad escalada, LW 2003 — por lo que el test cruzado compara
    propiedades, no igualdad exacta).
    """

    def __init__(self) -> None:
        self.shrinkage_: float | None = None

    def estimate(self, returns: np.ndarray) -> np.ndarray:
        x = _validate_returns(returns)
        t, n = x.shape
        x = x - x.mean(axis=0)

        s = _symmetrize((x.T @ x) / t)  # MLE, como en el paper
        var = np.diag(s).copy()
        _require_positive_variances(var, estimator="Ledoit-Wolf")
        sd = np.sqrt(var)
        outer_sd = np.outer(sd, sd)

        # Con un solo activo no existe correlación fuera de diagonal que
        # estimar. El target de correlación constante degenera exactamente
        # en la varianza muestral; no debe producir mean(empty) -> NaN.
        if n == 1:
            self.shrinkage_ = 0.0
            return ensure_psd(s)

        corr = s / outer_sd
        mask = ~np.eye(n, dtype=bool)
        r_bar = corr[mask].mean()

        f = r_bar * outer_sd
        np.fill_diagonal(f, var)

        # π: varianza asintótica de cada entrada de S
        y = x**2
        pi_mat = (y.T @ y) / t - s**2
        pi_hat = pi_mat.sum()

        # ρ: término diagonal + término fuera de diagonal del paper
        theta_ii = ((x**3).T @ x) / t - var[:, None] * s
        theta_jj = theta_ii.T
        rho_off = (
            0.5
            * r_bar
            * ((sd[None, :] / sd[:, None]) * theta_ii
               + (sd[:, None] / sd[None, :]) * theta_jj)
        )
        rho_hat = np.trace(pi_mat) + rho_off[mask].sum()

        gamma_hat = np.sum((f - s) ** 2)

        kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
        delta = float(np.clip(kappa / t, 0.0, 1.0))
        self.shrinkage_ = delta

        sigma = delta * f + (1.0 - delta) * s
        return ensure_psd(sigma)


class RMTDenoisedCovariance:
    """Denoising espectral vía Marchenko-Pastur (Laloux et al. 1999).

    Procedimiento sobre la matriz de correlación ``C``:

    1. ``q = N/T``; el borde superior de puro ruido es ``λ₊ = σ²(1+√q)²``.
    2. Eigendescomposición de ``C``. Autovalores ``≤ λ₊`` → ruido.
    3. Clipping: los autovalores de ruido se reemplazan por su promedio
       (preserva la traza ⇒ preserva la varianza total; no se "inventa"
       ni destruye riesgo agregado, solo se redistribuye la parte
       indistinguible de ruido de forma isótropa).
    4. Reconstrucción, diagonal forzada a 1, y retorno a covarianza con
       las volatilidades muestrales originales.

    Ajuste de Laloux: con ``fit_noise_variance=True`` (default), σ² no se
    fija en 1 sino en la fracción de varianza *no* explicada por los
    autovalores de señal, ``σ² = 1 − Σ_señal λᵢ / N``, iterando una vez.
    Esto reconoce que el mercado (autovalor dominante) absorbe parte de
    la varianza y desplaza el borde de la banda de ruido hacia abajo.
    """

    def __init__(self, fit_noise_variance: bool = True) -> None:
        self.fit_noise_variance = fit_noise_variance
        self.lambda_plus_: float | None = None
        self.n_signal_: int | None = None

    def estimate(self, returns: np.ndarray) -> np.ndarray:
        r = _validate_returns(returns)
        t, n = r.shape
        q = n / t

        s = _symmetrize(np.cov(r, rowvar=False, ddof=1))
        var = np.diag(s)
        _require_positive_variances(var, estimator="RMT")
        sd = np.sqrt(var)
        c = s / np.outer(sd, sd)
        np.fill_diagonal(c, 1.0)

        vals, vecs = np.linalg.eigh(c)  # ascendente

        sigma2 = 1.0
        lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
        if self.fit_noise_variance:
            signal = vals > lam_plus
            sigma2 = max(1.0 - vals[signal].sum() / n, 1e-8)
            lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2

        noise = vals <= lam_plus
        self.lambda_plus_ = float(lam_plus)
        self.n_signal_ = int((~noise).sum())

        cleaned = vals.copy()
        if noise.any():
            cleaned[noise] = vals[noise].mean()  # preserva la traza

        c_clean = _symmetrize((vecs * cleaned) @ vecs.T)
        d = np.sqrt(np.diag(c_clean))
        c_clean = c_clean / np.outer(d, d)
        np.fill_diagonal(c_clean, 1.0)

        sigma = c_clean * np.outer(sd, sd)
        return ensure_psd(sigma)
