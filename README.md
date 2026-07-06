# allocation-frontier

Librería de optimización de portafolios orientada a una sola pregunta empírica: **¿cuánto del fracaso del Markowitz clásico fuera de muestra es error de estimación, y cuánto se recupera con estimadores robustos?** Implementa tres estimadores de covarianza (muestral, shrinkage de Ledoit-Wolf, denoising espectral por teoría de matrices aleatorias), dos estimadores de retorno esperado (media histórica, Black-Litterman), la frontera eficiente clásica y la resampleada de Michaud, y un backtest walk-forward diseñado explícitamente para no fugar información futura.

## Por qué la covarianza muestral falla fuera de muestra

La covarianza muestral `S` es insesgada, pero para el uso que le da un optimizador es un estimador hostil. Con `N` activos y `T` observaciones, `S` estima O(N²) parámetros; cuando `q = N/T` no es despreciable, el espectro de `S` se dispersa artificialmente: los autovalores pequeños se subestiman y los grandes se sobreestiman (resultado exacto de Marchenko-Pastur). El optimizador de media-varianza es, en la frase de Michaud, una *máquina de maximizar error*: concentra peso precisamente en las direcciones de varianza aparentemente mínima, que son las peor estimadas. El resultado son portafolios extremos, inestables entre rebalanceos, y con riesgo realizado muy superior al prometido.

Los dos remedios implementados atacan el mismo mal por vías complementarias. **Ledoit-Wolf** contrae `S` hacia un target estructurado de correlación constante, con intensidad `δ*` derivada analíticamente del trade-off sesgo-varianza (implementación propia según el paper de 2004; sklearn se usa solo como oráculo en tests). **RMT** compara el espectro de la correlación muestral contra la banda de puro ruido `[λ₋, λ₊]` de Marchenko-Pastur, colapsa al promedio los autovalores indistinguibles de ruido preservando la traza, y conserva intactos los de señal (mercado, sectores), con el ajuste de Laloux et al. para la varianza de la fracción de ruido.

Del lado de los retornos, la media histórica se incluye como baseline honesto pero ruidoso —el error estándar de una media con vol del 20% y diez años de datos es del orden de la prima que se intenta medir— y **Black-Litterman** como remedio: ancla el retorno esperado a un prior de equilibrio por reverse-optimization y solo lo desplaza en la dirección de views con confianza explícita.

## Qué muestra el cuadro comparativo

El backtest walk-forward (ventana rodante de estimación, rebalanceo periódico, retornos acumulados estrictamente fuera de muestra) corre cada combinación de estimador de covarianza × estimador de retorno × objetivo, contra la vara de medir que la literatura exige: el portafolio **1/N equiponderado**, que DeMiguel, Garlappi y Uppal (2009) mostraron sorprendentemente difícil de batir una vez descontado el error de estimación.

La corrida de demostración (mercado sintético factorial de 40 activos, `q ≈ 0.16`) deja dos lecciones que conviene leer con honestidad. Primero, en un mundo gaussiano y estacionario —el más amable posible para Markowitz— la media histórica sí contiene señal y el max-Sharpe ingenuo gana en Sharpe bruto; la patología clásica se manifiesta no en el retorno sino en el **turnover**, un orden de magnitud mayor que el de Black-Litterman (0.29 vs 0.01 por rebalanceo), que es exactamente el costo del ruido de estimación convertido en rotación de cartera. Segundo, las estrategias min-varianza con estimadores robustos entregan la menor volatilidad y el menor drawdown realizados, que es lo que prometen. Sobre datos reales —colas pesadas, regímenes, primas inestables— la ventaja de la media histórica típicamente se evapora y el ranking se invierte; el pipeline es idéntico, basta sustituir la fuente de datos.

## Uso

```bash
uv sync
uv run pytest          # 19 tests, incluido el de no-lookahead
uv run python notebooks/demo.py
```

Con datos reales: `data.load_prices_yfinance(["SPY", "TLT", ...], start="2015-01-01")` seguido de `data.to_simple_returns(...)`; el resto del pipeline no cambia.

## Garantías de corrección

Toda covarianza devuelta es simétrica y PSD (los autovalores levemente negativos por reconstrucción numérica se clampean a ε documentadamente). Ledoit-Wolf se verifica contra sklearn como oráculo; RMT se verifica sobre ruido i.i.d. puro (debe detectar ~cero señal) y sobre un modelo factorial (debe conservar el factor de mercado) preservando la varianza total. Black-Litterman pasa el test de colapso al prior con views no informativas. Y el test crítico del backtest inyecta un shock artificial en el futuro de la serie y verifica que ningún peso decidido antes del shock cambia: si ese test falla, todos los resultados del backtest son inválidos por construcción.

## Arquitectura

`moments/covariance.py` y `moments/returns.py` contienen los estimadores con interfaz uniforme; `optimize.py` resuelve min-varianza, max-Sharpe y target-return con SLSQP (QP chicos con restricciones lineales no justifican cvxpy; la decisión está documentada en el módulo); `frontier.py` implementa la frontera clásica y la resampleada de Michaud sobre un Monte Carlo paramétrico vectorizado; `backtest.py` y `metrics.py` producen el cuadro comparativo OOS; `plotting.py` genera las cuatro familias de figuras (fronteras, heatmaps de denoising, equity curves, pesos promedio).
