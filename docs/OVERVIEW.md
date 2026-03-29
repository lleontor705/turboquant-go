# Overview — turboquant-go

**turboquant-go** es una librería Go de compresión vectorial que implementa algoritmos de investigación recientes para reducir el tamaño de embeddings sin perder demasiada fidelidad. Está diseñada para ser **pura Go**, **determinística** y **concurrent-safe**.

## Algoritmos incluidos

### PolarQuant (recomendado para fidelidad)
- **Idea**: rotación aleatoria → descomposición polar recursiva → cuantización independiente por nivel.
- **Ventaja clave**: la distribución conjunta se factoriza, por lo que la cuantización por nivel es **óptima**.
- **Fidelidad**: almacena la **norma original** y reescala al decodificar.
- **Uso ideal**: cuando quieres la mejor calidad por bit (MSE y coseno altos) y un buen ratio de compresión.

### TurboQuant_mse (ratio/calidad balanceado)
- **Idea**: rotación aleatoria → cuantización Lloyd‑Max con codebooks Beta(d).
- **Fidelidad**: almacena la **norma original** y reescala en la reconstrucción.
- **Uso ideal**: cuando necesitas alta compresión con buena precisión y un pipeline simple.

### TurboQuant_prod (inner product unbiased)
- **Idea**: cuantización MSE (b‑1 bits) + sketch QJL de 1 bit del residual.
- **Ventaja clave**: **estimación de inner product UNBIASED** (E[estimate] = IP real).
- **Fidelidad**: almacena norma y residual norm para corrección.
- **Uso ideal**: ANN search y ranking donde el sesgo en IP distorsiona resultados.

### QJL Sketching (1‑bit sketches)
- **Idea**: proyección JL → cuantización de signo → bit‑packing.
- **Ventaja clave**: compresión extrema y **Hamming distance** muy rápida.
- **Uso ideal**: búsqueda ANN rápida, filtros o pre‑ranking con memoria mínima.

### Cuantización Escalar Uniforme (baseline)
- **Idea**: cuantización uniforme 4‑bit / 8‑bit con empaquetado de nibbles.
- **Ventaja clave**: velocidad muy alta y simplicidad.
- **Uso ideal**: línea base, validación o sistemas con CPU muy restringida.

## ¿Cuál usar?

| Objetivo | Mejor opción | Motivo |
|---|---|---|
| Máxima fidelidad por bit | **PolarQuant** | Cuantización óptima por nivel + norma almacenada |
| Ratio/precisión equilibrado | **TurboQuant_mse** | Rotación + Lloyd‑Max con codebooks Beta(d) |
| Ranking por inner product sin sesgo | **TurboQuant_prod** | Estimación IP **unbiased** |
| Velocidad extrema / muy poca memoria | **QJL** | Sketch 1‑bit + Hamming ultra‑rápido |
| Baseline simple | **UniformQuantizer** | 4‑bit / 8‑bit sin dependencias |

## Propiedades comunes

- **Determinismo**: semillas controlan rotaciones y proyecciones.
- **Concurrency‑safe**: todos los tipos son inmutables tras construcción.
- **Wire formats versionados**: `CompressedVector` y sketches se serializan en binario con versión.
