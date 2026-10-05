# allocation-frontier

Librería de optimización de portafolios orientada a una sola pregunta empírica: **¿cuánto del fracaso del Markowitz clásico fuera de muestra es error de estimación, y cuánto se recupera con estimadores robustos?** Implementa tres estimadores de covarianza (muestral, shrinkage de Ledoit-Wolf, denoising espectral por teoría de matrices aleatorias), dos estimadores de retorno esperado (media histórica, Black-Litterman), la frontera eficiente clásica y la resampleada de Michaud, y un backtest walk-forward diseñado explícitamente para no fugar información futura.

## Por qué la covarianza muestral falla fuera de muestra

La covarianza muestral `S` es insesgada, pero para el uso que le da un optimizador es un estimador hostil. Con `N` activos y `T` observaciones, `S` estima O(N²) parámetros; cuando `q = N/T` no es despreciable, el espectro de `S` se dispersa artificialmente: los autovalores pequeños se subestiman y los grandes se sobreestiman (resultado exacto de Marchenko-Pastur). El optimizador de media-varianza es, en la frase de Michaud, una *máquina de maximizar error*: concentra peso precisamente en las direcciones de varianza aparentemente mínima, que son las peor estimadas. El resultado son portafolios extremos, inestables entre rebalanceos, y con riesgo realizado muy superior al prometido.

Los dos remedios implementados atacan el mismo mal por vías complementarias. **Ledoit-Wolf** contrae `S` hacia un target estructurado de correlación constante, con intensidad `δ*` derivada analíticamente del trade-off sesgo-varianza (implementación propia según el paper de 2004; sklearn se usa solo como oráculo en tests). **RMT** compara el espectro de la correlación muestral contra la banda de puro ruido `[λ₋, λ₊]` de Marchenko-Pastur, colapsa al promedio los autovalores indistinguibles de ruido preservando la traza, y conserva intactos los de señal (mercado, sectores), con el ajuste de Laloux et al. para la varianza de la fracción de ruido.

Del lado de los retornos, la media histórica se incluye como baseline honesto pero ruidoso —el error estándar de una media con vol del 20% y diez años de datos es del orden de la prima que se intenta medir— y **Black-Litterman** como remedio: ancla el retorno esperado a un prior de equilibrio por reverse-optimization y solo lo desplaza en la dirección de views con confianza explícita.

## Qué mide el cuadro comparativo

El backtest walk-forward usa una ventana rodante de estimación y decide pesos
solo con `returns[t-window:t]`. Desde v0.3, los holdings **derivan realmente**
entre rebalanceos: no se restauran los pesos objetivo cada día. En el siguiente
rebalanceo el turnover se mide contra los pesos pre-trade que resultaron de esa
deriva, y los costes se descuentan en la trayectoria cuando ocurre el trade.

El cuadro compara cada combinación de estimador de covarianza × estimador de
retorno × objetivo contra el portafolio **1/N equiponderado**. Publica retorno y
Sharpe netos, sus equivalentes gross, drawdown, turnover ejecutado y drag de
costes. Esto separa tres preguntas que v0.2 mezclaba: qué target decidió el
optimizador, cómo evolucionaron las posiciones al mantenerlas y cuánto costó
volver al target.

Los resultados numéricos de v0.2 no se presentan como evidencia de v0.3. El
demo regenerará `comparison_oos.csv` y la curva de riqueza usando la nueva
semántica; hasta entonces es preferible no publicar un ranking a conservar uno
calculado con una convención de ejecución incorrecta.

## Uso

```bash
uv sync
uv run pytest          # incluye no-lookahead, weight drift y costes pathwise
uv run python notebooks/demo.py
```

Con datos reales: `data.load_prices_yfinance(["SPY", "TLT", ...], start="2015-01-01")` seguido de `data.to_simple_returns(...)`; el resto del pipeline no cambia.

## Garantías de corrección

Las covarianzas se validan antes de estimar: NaN/Inf y muestras demasiado
cortas fallan explícitamente; los estimadores basados en correlación rechazan
activos de varianza cero en vez de propagar divisiones por cero. Toda
covarianza reconstruida se mantiene simétrica y PSD.

Ledoit-Wolf se contrasta con sklearn por propiedades; RMT se prueba sobre ruido
i.i.d. y sobre un modelo factorial preservando varianza total. Black-Litterman
usa sistemas lineales en lugar de inversiones explícitas y se prueba incluso
con una covarianza PSD singular cuando el sistema de views está definido.

El backtest conserva como invariantes separados: no-lookahead, deriva exacta de
holdings, turnover contra pesos pre-trade y costes pathwise. Un shock inyectado
en el futuro no puede cambiar decisiones anteriores; una estrategia 50/50 con
un activo ganador debe dejar de ser 50/50 al período siguiente si no hubo
rebalanceo.

## Arquitectura

`moments/covariance.py` y `moments/returns.py` contienen los estimadores con interfaz uniforme; `optimize.py` resuelve min-varianza, max-Sharpe y target-return con SLSQP (QP chicos con restricciones lineales no justifican cvxpy; la decisión está documentada en el módulo); `frontier.py` implementa la frontera clásica y la resampleada de Michaud sobre un Monte Carlo paramétrico vectorizado; `backtest.py` modela holdings, rebalanceos y costes; `metrics.py` produce las métricas OOS; `plotting.py` genera las cuatro familias de figuras (fronteras, heatmaps de denoising, equity curves, pesos promedio).
