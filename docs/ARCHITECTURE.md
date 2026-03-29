# Arquitectura

Este documento describe la estructura de paquetes, el flujo de datos y los formatos binarios (wire formats) de **turboquant-go**.

## Estructura de paquetes

```
turboquant-go/
├── quantize/   # cuantización vectorial (TurboQuant, PolarQuant, Uniform)
├── sketch/     # QJL 1‑bit sketching + Hamming
├── rotate/     # rotaciones aleatorias y FWHT
├── internal/   # helpers internos (bits, serial)
└── examples/   # demos y casos de uso
```

### Paquete `quantize/`
- `Quantizer` interface: `Quantize`, `Dequantize`, `Bits`.
- `CompressedVector`: estructura común con `Data`, `Dim`, `Min`, `Max`, `BitsPer`.
- Implementaciones:
  - `TurboQuantizer` (TurboQuant_mse)
  - `TurboProdQuantizer` (TurboQuant_prod, unbiased IP)
  - `PolarQuantizer` (PolarQuant)
  - `UniformQuantizer` (4‑bit / 8‑bit)

### Paquete `sketch/`
- `QJLSketcher`: proyección JL + cuantización de signo.
- `BitVector`: bits empaquetados + opcionales outliers.
- `HammingDistance` / `EstimateInnerProduct` para similitud rápida.

### Paquete `rotate/`
- `RandomOrthogonal`: matriz ortogonal aleatoria (QR de Gaussiana).
- `FWHT`: Fast Walsh‑Hadamard Transform in‑place.

### Paquete `internal/`
- `internal/bits`: pack/unpack de signos en `[]uint64`.
- `internal/serial`: utilidades para versionado y slices binarios.

## Flujo de datos (alto nivel)

### TurboQuant_mse
1. **Normalizar** vector a norma unitaria.
2. **Rotación aleatoria** con matriz ortogonal.
3. **Cuantización Lloyd‑Max** usando codebooks Beta(d).
4. **Empaquetado** de índices en bits.
5. **Wire format** incluye versión + norma original.

### PolarQuant
1. **Normalizar** vector.
2. **Rotación aleatoria**.
3. **Transformación polar** recursiva (ángulos + radios).
4. **Cuantización por nivel** con codebooks específicos.
5. **Radii** cuantizados a `uint16`.
6. **Wire format** con parámetros de config + radii + índices.

### TurboQuant_prod
1. **Normalizar** vector.
2. **MSE cuantización** (b‑1 bits) → reconstrucción.
3. **Residual** = x_unit − x̂_mse, guardar norma residual `γ`.
4. **QJL sketch** 1‑bit del residual.
5. **Wire format** incluye norma original, `γ`, MSE bytes y sketch bits.
6. **EstimateInnerProduct** aplica corrección QJL para estimador unbiased.

### QJL Sketching
1. **Proyección** JL (Gaussiana o SRHT).
2. **Opcional**: outliers (indices fijos o top‑K).
3. **Sign quantization** y empaquetado en `[]uint64`.
4. **Hamming distance** para similitud rápida.

## Wire formats

### `CompressedVector` (común)

```
[version:1B]
[dim:uint32]
[min:float64]
[max:float64]
[bitsPer:uint32]
[data_len:uint32]
[data:bytes]
```

### TurboQuant_mse (`turbo.go`)

```
[version:1B]
[norm:float64]
[packed indices: bytes]
```

### PolarQuant (`polar.go`)

```
[version:1B]
[dim:uint32]
[levels:1B]
[bitsLevel1:1B]
[bitsRest:1B]
[radiusBits:1B]
[norm:float64]
[numRadii:uint32]
[radii: numRadii × uint16]
[level1 packed indices]
[level2 packed indices]
...
[levelL packed indices]
```

### TurboQuant_prod (`turbo_prod.go`)

```
[version:1B]
[sketchDim:uint32]
[norm:float64]
[gamma:float64]
[mseDataLen:uint32]
[mseData bytes]
[numSketchWords:uint32]
[sketchBits: uint64 × numSketchWords]
```

### QJL BitVector (`sketch/qjl.go`)

```
[version:1B]
[dim:uint32]
[bits: uint64 slice]
[has_outliers:1B]
[outlier_indices: uint64 slice] (opcional)
[outlier_values: float64 slice] (opcional)
```

## Notas de diseño

- **Norma almacenada**: TurboQuant_mse, TurboQuant_prod y PolarQuant reescalan con la norma original para mejorar fidelidad.
- **Radii en uint16**: PolarQuant reduce el overhead y acerca la compresión a ratios teóricos.
- **Determinismo**: semillas para rotación y proyección hacen reproducibles los resultados.
