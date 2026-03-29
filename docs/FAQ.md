# FAQ

## ¿Por qué se almacena la norma del vector?

Los algoritmos TurboQuant_mse, TurboQuant_prod y PolarQuant **normalizan** el vector a norma unitaria antes de cuantizar. Para reconstruir la magnitud original, se guarda la norma y se reescala durante la dequantización. Esto reduce el error absoluto y mejora la fidelidad global.

## ¿Cómo se almacenan los radios en PolarQuant?

Los radios finales se cuantizan en **uint16** (`RadiusBits = 16`). Esto reduce el overhead frente a `float64` y acerca los ratios de compresión a los valores teóricos. La cuantización es lineal en [0,1] y se aplica tras la descomposición polar.

## ¿Qué significa que TurboQuant_prod sea “unbiased”?

Significa que el estimador de inner product cumple:

> **E[estimate] = inner product real**

Esto es crítico en ANN search: un sesgo sistemático puede distorsionar el ranking. TurboQuant_prod corrige el error MSE con un sketch QJL de 1 bit del residual.

## ¿Qué pasa con outliers en QJL?

QJL permite dos enfoques:

- **OutlierIndices**: índices fijos almacenados en precisión completa (recomendado).
- **OutlierK**: selecciona dinámicamente los top‑K de mayor magnitud (deprecado).

En ambos casos, esos canales se guardan en `BitVector.OutlierValues` y se excluyen del packing de bits.

## ¿La cuantización escalar admite 4‑bit y 8‑bit?

Sí. `UniformQuantizer` soporta 4‑bit (dos dimensiones por byte) y 8‑bit (un byte por dimensión). Es la opción más simple para baseline o para entornos con CPU muy limitada.
