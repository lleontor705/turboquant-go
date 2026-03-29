# turboquant

**Librería Go pura de compresión vectorial basada en los algoritmos TurboQuant y PolarQuant de Google Research**

[![Go Version](https://img.shields.io/badge/Go-1.24+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

---

## Descripción General

**turboquant** es una librería Go que implementa los algoritmos de compresión vectorial más avanzados de la literatura de investigación, incluyendo:

- **PolarQuant** (arXiv:2502.02617, AISTATS 2026) — Descomposición polar recursiva con cuantización independiente por nivel. Alcanza 9.1× de compresión con similaritud coseno de 0.984.
- **TurboQuant_mse** (arXiv:2504.19874, ICLR 2026) — Rotación aleatoria + cuantización Lloyd-Max con codebooks Beta(d). Hasta 21.3× de compresión.
- **QJL Sketching** (arXiv:2406.03482) — Proyección Johnson-Lindenstrauss + cuantización a 1 bit. 192× de compresión con estimación de similaritud vía distancia Hamming.
- **TurboQuant_prod** (arXiv:2504.19874, ICLR 2026) — Dos etapas: cuantización MSE + sketch QJL del residual. **ÚNICO algoritmo con estimación de inner product UNBIASED** — E[estimate] = IP verdadero. Ideal para ANN search.
- **Cuantización Escalar Uniforme** — Línea base de 4-bit y 8-bit con empaquetamiento de nibbles.

### ¿Por qué Go?

- **Zero CGO** — Compilación cruzada limpia, binario único, sin dependencias nativas.
- **`math/bits.OnesCount64`** — Hamming distance cercana al óptimo hardware vía instrucciones POPCNT.
- **Safe for concurrency** — Todos los tipos son inmutables tras construcción, sin locks necesarios.
- **Rellena un vacío** — No existe otra librería Go de cuantización vectorial con estos algoritmos.

### Papers Referenciados

| Algoritmo | Paper | Venue |
|-----------|-------|-------|
| PolarQuant | [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) | AISTATS 2026 |
| TurboQuant | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | ICLR 2026 |
| QJL | [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) | — |

Código de referencia QJL: [github.com/amirzandieh/QJL](https://github.com/amirzandieh/QJL)

---

## Instalación

```bash
go get github.com/lleontor705/turboquant-go
```

Requiere Go 1.24+. Única dependencia externa: [gonum](https://gonum.org/).

---

## Algoritmos Implementados

| Algoritmo | Bits/coord | Compresión | MSE (d=768) | Cosine Sim | Encode |
|-----------|-----------|------------|-------------|------------|--------|
| **PolarQuant** (default) | 3.875 | 9.1× | 0.000042 | 0.984 | 624 µs |
| **TurboQuant_mse** 3-bit | 3 | 21.3× | 0.000045 | — | 724 µs |
| **TurboQuant_mse** 4-bit | 4 | 16.0× | 0.000012 | — | 724 µs |
| **TurboQuant_prod** 3-bit | 3 | 20× | bias≈0.001 | **unbiased** | 2.6 ms |
| **TurboQuant_prod** 4-bit | 4 | 16× | bias≈0.001 | **unbiased** | 2.6 ms |
| **QJL Sketch** 256-bit | 0.33 | 192× | — | — | 251 µs |
| **Scalar** 8-bit | 8 | 8× | 0.098% | — | 2.9 µs |

> Benchmarks en AMD Ryzen 7 5800X3D, Go 1.24, Windows/amd64.

---

## Quick Start — PolarQuant (recomendado)

PolarQuant ofrece la mejor calidad de reconstrucción por bit. Usa descomposición polar recursiva que factoriza la distribución conjunta, haciendo la cuantización independiente por nivel óptima (no una aproximación).

```go
package main

import (
    "fmt"
    "math"

    "github.com/lleontor705/turboquant-go/quantize"
)

func main() {
    // Configuración por defecto: 4 niveles, 4 bits nivel-1, 2 bits niveles 2-4
    config := quantize.DefaultPolarConfig(768) // dim=768
    pq, err := quantize.NewPolarQuantizer(config)
    if err != nil {
        panic(err)
    }

    // Vector de ejemplo (768-dimensional)
    vec := make([]float64, 768)
    for i := range vec {
        vec[i] = math.Sin(float64(i) * 0.01)
    }

    // Codificar → CompressedVector
    cv, err := pq.Quantize(vec)
    if err != nil {
        panic(err)
    }

    // Decodificar → float64 vector
    recovered, err := pq.Dequantize(cv)
    if err != nil {
        panic(err)
    }

    // Reportar métricas
    origBytes := 768 * 8 // float64
    compressedBytes := len(cv.Data)
    ratio := float64(origBytes) / float64(compressedBytes)

    fmt.Printf("Bits/coord:     %.3f\n", pq.Bits())
    fmt.Printf("Compresión:     %.1f× (%d → %d bytes)\n", ratio, origBytes, compressedBytes)

    // Calcular MSE
    var mse float64
    for i := range vec {
        d := vec[i] - recovered[i]
        mse += d * d
    }
    mse /= float64(len(vec))
    fmt.Printf("MSE:            %.6f\n", mse)
}
```

---

## Quick Start — TurboQuant_mse

TurboQuant aplica una rotación aleatoria seguida de cuantización Lloyd-Max usando codebooks derivados de la distribución Beta. Más simple que PolarQuant, con excelente ratio compresión/calidad.

```go
package main

import (
    "fmt"
    "math"

    "github.com/lleontor705/turboquant-go/quantize"
)

func main() {
    // Crear quantizer: dim=768, 4 bits por coordenada, seed=42
    tq, err := quantize.NewTurboQuantizer(768, 4, 42)
    if err != nil {
        panic(err)
    }

    vec := make([]float64, 768)
    for i := range vec {
        vec[i] = math.Cos(float64(i) * 0.02)
    }

    // Codificar
    cv, err := tq.Quantize(vec)
    if err != nil {
        panic(err)
    }

    // Decodificar
    recovered, err := tq.Dequantize(cv)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Bits/coord:  %d\n", tq.Bits())
    fmt.Printf("Dimensión:   %d\n", tq.Dim())
    fmt.Printf("Compresión:  %.1f×\n",
        float64(768*8)/float64(len(cv.Data)))
    fmt.Printf("Recuperado:  %.4f...\n", recovered[0])
}
```

---

## Quick Start — TurboQuant_prod

**TurboQuant_prod es el ÚNICO algoritmo que proporciona estimación de inner product UNBIASED** — E[estimate] = IP verdadero. Esto es crítico para ANN search donde el ranking precisa ser preciso.

Combina dos etapas: cuantización MSE (b-1 bits) para reconstrucción + sketch QJL de 1 bit del residual para corrección de inner product. La corrección QJL compensa el error de cuantización MSE, produciendo un estimador insesgado.

```go
package main

import (
    "fmt"
    "math"
    "math/rand"

    "github.com/lleontor705/turboquant-go/quantize"
)

func main() {
    const dim = 768
    const bits = 3  // 2-bit MSE + 1-bit QJL residual

    // Crear quantizer para estimación de inner product UNBIASED
    q, err := quantize.NewTurboProdQuantizer(dim, bits, dim, 42)
    if err != nil {
        panic(err)
    }

    // Generar vectores aleatorios
    rng := rand.New(rand.NewSource(123))
    x := randomUnitVector(rng, dim)
    query := randomUnitVector(rng, dim)

    // Codificar
    compressed, err := q.Quantize(x)
    if err != nil {
        panic(err)
    }

    // Decodificar (solo parte MSE — el residual es sketch irreversible)
    reconstructed, err := q.Dequantize(compressed)
    if err != nil {
        panic(err)
    }
    mseReconstruction := mse(x, reconstructed)
    fmt.Printf("MSE reconstrucción: %.6f\n", mseReconstruction)

    // Estimación de inner product UNBIASED
    ipEstimated, err := q.EstimateInnerProduct(query, compressed)
    if err != nil {
        panic(err)
    }
    ipExact := dotProduct(query, x)
    fmt.Printf("IP exacto:     %.6f\n", ipExact)
    fmt.Printf("IP estimado:   %.6f\n", ipEstimated)
    fmt.Printf("Error:         %.6f\n", math.Abs(ipEstimated-ipExact))
    fmt.Printf("Bits/coord:    %d (%d MSE + 1 QJL)\n", bits, bits-1)
}

// helpers
func randomUnitVector(rng *rand.Rand, dim int) []float64 {
    v := make([]float64, dim)
    for i := range v {
        v[i] = rng.NormFloat64()
    }
    norm := 0.0
    for _, x := range v {
        norm += x * x
    }
    norm = math.Sqrt(norm)
    for i := range v {
        v[i] /= norm
    }
    return v
}

func dotProduct(a, b []float64) float64 {
    s := 0.0
    for i := range a {
        s += a[i] * b[i]
    }
    return s
}

func mse(orig, rec []float64) float64 {
    s := 0.0
    for i := range orig {
        d := orig[i] - rec[i]
        s += d * d
    }
    return s / float64(len(orig))
}
```

**¿Por qué unbiased importa?** En ANN search, los resultados se ordenan por similaritud. Un estimador sesgado sistemáticamente sobre- o sub-estima ciertos vectores, distorsionando el ranking. TurboQuant_prod garantiza que el error promedio es cero — el ranking preserva la calidad real de los vecinos.

---

## Quick Start — QJL Sketching

QJL comprime vectores en sketches de 1 bit. La distancia Hamming entre sketches estima la similaritud coseno del vector original — ideal para búsqueda ANN rápida.

```go
package main

import (
    "fmt"
    "math"

    "github.com/lleontor705/turboquant-go/sketch"
)

func main() {
    // Crear sketcher: 768-dim → 256-bit sketches
    sketcher, err := sketch.NewQJLSketcher(sketch.QJLOptions{
        Dim:       768,
        SketchDim: 256,
        Seed:      42,
    })
    if err != nil {
        panic(err)
    }

    // Generar dos vectores de ejemplo
    vecA := make([]float64, 768)
    vecB := make([]float64, 768)
    for i := range vecA {
        vecA[i] = math.Sin(float64(i) * 0.01)
        vecB[i] = math.Cos(float64(i) * 0.01)
    }

    // Crear sketches
    a, err := sketcher.Sketch(vecA)
    if err != nil {
        panic(err)
    }
    b, err := sketcher.Sketch(vecB)
    if err != nil {
        panic(err)
    }

    // Distancia Hamming (panics on mismatch — usar en hot paths)
    dist := sketch.HammingDistance(*a, *b)
    fmt.Printf("Hamming distance: %d / %d bits\n", dist, a.Dim)

    // Estimar inner product en [-1, 1]
    ip, err := sketch.EstimateInnerProduct(*a, *b)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Similaritud estimada: %.4f\n", ip)
}
```

**Con manejo de outliers** — almacenar las K proyecciones más grandes en precisión completa:

```go
sketcher, _ := sketch.NewQJLSketcher(sketch.QJLOptions{
    Dim:       768,
    SketchDim: 256,
    Seed:      42,
    OutlierK:  16, // 16 proyecciones outlier en precisión completa
})
```

**Con SRHT** — proyecciones O(n log n) (requiere que Dim sea potencia de 2):

```go
sketcher, _ := sketch.NewQJLSketcher(sketch.QJLOptions{
    Dim:       1024,
    SketchDim: 256,
    Seed:      42,
    UseSRHT:   true,
})
```

---

## Quick Start — Cuantización Escalar Uniforme

Cuantización escalar simple a 4-bit o 8-bit. Ideal como línea base o cuando se necesita codificación ultra-rápida.

```go
package main

import (
    "fmt"

    "github.com/lleontor705/turboquant-go/quantize"
)

func main() {
    // 8-bit quantizer para valores en [0, 10]
    q, err := quantize.NewUniformQuantizer(0, 10, 8)
    if err != nil {
        panic(err)
    }

    vec := []float64{1.2, 3.4, 5.6, 7.8, 9.9}

    // Codificar → CompressedVector
    cv, err := q.Quantize(vec)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Comprimido: %d bytes (era %d float64s)\n", len(cv.Data), len(vec))

    // Decodificar → float64
    recovered, err := q.Dequantize(cv)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Recuperado:  %.4v\n", recovered)
    fmt.Printf("Error máx:   %.6f\n", q.MaxError())
}
```

Cuantización 4-bit empaqueta dos dimensiones por byte (16× compresión):

```go
q4, _ := quantize.NewUniformQuantizer(-1.0, 1.0, 4)
cv, _ := q4.Quantize(embedding) // 768-dim → 384 bytes
```

---

## Quick Start — FWHT

Transformada Walsh-Hadamard rápida in-place. O(n log n). La longitud debe ser potencia de 2.

```go
package main

import (
    "fmt"

    "github.com/lleontor705/turboquant-go/rotate"
)

func main() {
    x := []float64{1, 2, 3, 4}
    err := rotate.FWHT(x) // in-place; x = [10, -2, -4, 0]
    if err != nil {
        panic(err)
    }
    fmt.Printf("Transformado: %.0v\n", x)
}
```

---

## Estructura del Proyecto

```
turboquant-go/
├── quantize/                        # Cuantización vectorial
│   ├── quantize.go                  # Quantizer interface, CompressedVector, errores
│   ├── scalar.go                    # UniformQuantizer (4-bit / 8-bit)
│   ├── turbo.go                     # TurboQuantizer (TurboQuant_mse)
│   ├── turbo_prod.go                # TurboProdQuantizer (TurboQuant_prod — unbiased IP)
│   ├── lloydmax.go                  # Lloyd-Max solver, NearestCentroid, QuantizeWithCodebook
│   ├── codebooks.go                 # BetaPDF, TurboCodebook
│   ├── polar.go                     # PolarQuantizer
│   ├── polar_types.go               # PolarVector, PolarConfig, DefaultPolarConfig
│   ├── polar_transform.go           # PolarTransform, InversePolarTransform
│   └── polar_codebooks.go           # sinPowerPDF, LevelCodebook, PolarSinExponent
├── sketch/                          # 1-bit sketching (QJL)
│   ├── types.go                     # BitVector, errores centinela
│   ├── qjl.go                       # QJLSketcher, QJLOptions, serialización
│   ├── jl.go                        # Projector, NewGaussianProjection, NewSRHT
│   └── hamming.go                   # HammingDistance, HammingDistanceSafe, EstimateInnerProduct
├── rotate/                          # Rotaciones y transformadas
│   ├── hadamard.go                  # FWHT, IsPowerOfTwo
│   └── orthogonal.go                # RandomOrthogonal (QR)
├── internal/
│   ├── bits/                        # Empaquetamiento de bits
│   │   └── pack.go                  # Pack, Unpack, PopCount
│   └── serial/                      # Helpers serialización binaria
├── cmd/
│   └── genfixtures/                 # Generador de fixtures para tests
├── examples/
│   ├── ann_search/                  # Ejemplo búsqueda ANN con QJL
│   ├── polar_search/                # Ejemplo búsqueda con PolarQuant
│   └── turbo_search/                # Ejemplo búsqueda con TurboQuant
├── testdata/                        # Fixtures JSON para golden tests
├── go.mod
└── go.sum
```

---

## Benchmarks

Medidos en AMD Ryzen 7 5800X3D, Go 1.24, Windows/amd64.

### Operaciones Core

| Operación | Dimensión | Tiempo | Memoria | Allocs |
|-----------|-----------|--------|---------|--------|
| Hamming Distance | 256-bit | **7.5 ns** | 0 B | 0 |
| FWHT | 256-dim | **1.0 µs** | 0 B | 0 |
| FWHT | 1024-dim | **3.8 µs** | 0 B | 0 |
| FWHT | 4096-dim | **20.8 µs** | 0 B | 0 |

### Cuantización (d=768)

| Operación | Tiempo | Memoria | Allocs |
|-----------|--------|---------|--------|
| PolarQuant Encode | **624 µs** | 41 KB | 27 |
| PolarQuant Decode | **1.1 ms** | 30 KB | 19 |
| TurboQuant Encode 3-bit | **724 µs** | 19 KB | 4 |
| TurboQuant Decode 3-bit | **1.3 ms** | 18 KB | 3 |
| TurboQuant Encode 4-bit | **724 µs** | 19 KB | 4 |
| TurboQuant Decode 4-bit | **1.3 ms** | 18 KB | 3 |
| TurboProd Encode 768-dim 3-bit | **2.6 ms** | 57 KB | 15 |
| TurboProd Decode 768-dim 3-bit | **1.2 ms** | 19 KB | 7 |
| TurboProd EstimateIP 768-dim | **1.9 ms** | 26 KB | 9 |
| Scalar Quantize 8-bit | **2.9 µs** | 768 B | 1 |
| Scalar Quantize 4-bit | **3.3 µs** | 384 B | 1 |
| Scalar Dequantize 8-bit | **1.1 µs** | 6 KB | 1 |
| Scalar Dequantize 4-bit | **1.4 µs** | 6 KB | 1 |

### QJL Sketching (768-dim → 256-bit)

| Configuración | Tiempo | Allocs |
|--------------|--------|--------|
| Sin outliers | **251 µs** | 4 |
| Con outliers (K=16) | **258 µs** | 10 |

### Hamming Distance

| Sketch Dim | Tiempo | Allocs |
|-----------|--------|--------|
| 256-bit | **7.5 ns** | 0 |
| 1024-bit | **10.8 ns** | 0 |
| 4096-bit | **45.8 ns** | 0 |

> Todos los paquetes son seguros para uso concurrente sin sincronización externa. Los tipos son inmutables tras la construcción.

---

## Publicación / Releases

Las versiones públicas se publican como **GitHub Releases** usando tags semánticos:

- `v0.1.0`, `v0.1.1`, ...

Cada tag `v*` activa el workflow de release, ejecuta tests, y genera un release en GitHub.
Para ver versiones disponibles, visita la página de releases del repositorio.

---

## Referencia de API

### `quantize/` — Cuantización Vectorial

#### Interfaces y Tipos Core

| Tipo | Descripción |
|------|-------------|
| `Quantizer` | Interfaz: `Quantize(vec []float64) (CompressedVector, error)`, `Dequantize(cv CompressedVector) ([]float64, error)`, `Bits() int` |
| `CompressedVector` | Vector comprimido con metadata: `Data []byte`, `Dim int`, `Min float64`, `Max float64`, `BitsPer int`. Implementa `encoding.BinaryMarshaler` / `BinaryUnmarshaler`. |

#### Errores

| Error | Descripción |
|-------|-------------|
| `ErrInvalidConfig` | Parámetros de construcción inválidos (min >= max, bits no soportados, dim <= 0) |
| `ErrDimensionMismatch` | Longitud del vector no coincide con la dimensión esperada |
| `ErrNaNInput` | El vector de entrada contiene valores NaN |

#### UniformQuantizer

```go
func NewUniformQuantizer(min, max float64, bits int) (*UniformQuantizer, error)
```
- `bits`: 4 o 8
- 4-bit: dos dimensiones empaquetadas por byte (nibble high + low)
- 8-bit: una dimensión por byte

```go
func (q *UniformQuantizer) Quantize(vec []float64) (CompressedVector, error)
func (q *UniformQuantizer) Dequantize(cv CompressedVector) ([]float64, error)
func (q *UniformQuantizer) Bits() int
func (q *UniformQuantizer) MaxError() float64  // (max - min) / (2 * levels)
```

#### TurboQuantizer

```go
func NewTurboQuantizer(dim int, bits int, seed int64) (*TurboQuantizer, error)
```
- `dim`: dimensión del vector (>= 2)
- `bits`: bits por coordenada (1, 2, 3, o 4)
- `seed`: semilla determinística para la rotación

```go
func (tq *TurboQuantizer) Quantize(vec []float64) (CompressedVector, error)
func (tq *TurboQuantizer) Dequantize(cv CompressedVector) ([]float64, error)
func (tq *TurboQuantizer) Bits() int
func (tq *TurboQuantizer) Dim() int
func (tq *TurboQuantizer) RotationMatrix() *mat.Dense
```

#### TurboProdQuantizer

**Estimación de inner product UNBIASED** — combina cuantización MSE (b-1 bits) + sketch QJL de 1 bit del residual para corrección.

```go
func NewTurboProdQuantizer(dim int, bits int, sketchDim int, seed int64) (*TurboProdQuantizer, error)
```
- `dim`: dimensión del vector (>= 2)
- `bits`: bits totales por coordenada (2, 3, o 4). Usa (bits-1) para MSE + 1 para QJL.
- `sketchDim`: dimensión del sketch QJL (típicamente = dim para calidad óptima, debe ser <= dim)
- `seed`: semilla determinística

```go
func (tpq *TurboProdQuantizer) Quantize(vec []float64) (CompressedVector, error)
func (tpq *TurboProdQuantizer) Dequantize(cv CompressedVector) ([]float64, error)
func (tpq *TurboProdQuantizer) EstimateInnerProduct(query []float64, cv CompressedVector) (float64, error)  // ← KEY METHOD
func (tpq *TurboProdQuantizer) Bits() int
func (tpq *TurboProdQuantizer) Dim() int
func (tpq *TurboProdQuantizer) SketchDim() int
func (tpq *TurboProdQuantizer) ParseProdVector(cv CompressedVector) (ProdVector, error)
```

##### ProdVector

```go
type ProdVector struct {
    MSEData       CompressedVector   // sub-vector MSE cuantizado (b-1 bits)
    Residual      sketch.BitVector   // sketch QJL de 1 bit del residual
    ResidualNorm  float64            // γ = ||x - x̂_mse||
    Dim           int                // dimensión original
    Bits          int                // bits efectivos por coordenada
}
```

> **Nota**: `Dequantize` reconstruye solo la parte MSE. El residual es un sketch irreversible de 1 bit. Para estimar inner products use `EstimateInnerProduct`, que aplica la corrección QJL para obtener un estimador **unbiased**.

#### PolarQuantizer

```go
func NewPolarQuantizer(config PolarConfig) (*PolarQuantizer, error)
```

```go
func (pq *PolarQuantizer) Quantize(vec []float64) (CompressedVector, error)
func (pq *PolarQuantizer) Dequantize(cv CompressedVector) ([]float64, error)
func (pq *PolarQuantizer) Bits() int       // efectivo: 3.875 con config default
func (pq *PolarQuantizer) Dim() int
func (pq *PolarQuantizer) RotationMatrix() *mat.Dense
```

#### PolarConfig

```go
type PolarConfig struct {
    Dim        int   // debe ser múltiplo de 2^Levels (típicamente múltiplo de 16)
    Levels     int   // niveles de descomposición polar (default: 4)
    BitsLevel1 int   // bits para ángulos nivel-1 (default: 4)
    BitsRest   int   // bits para ángulos niveles 2..L (default: 2)
    RadiusBits int   // bits para radios finales (default: 16, tipo FP16)
    Seed       int64 // semilla aleatoria
}

func DefaultPolarConfig(dim int) PolarConfig
func (c PolarConfig) BitsPerCoord() float64
```

#### PolarVector

```go
type PolarVector struct {
    AngleIndices [][]int   // índices de ángulos por nivel
    Radii        []float64 // radios del nivel final (d/2^L valores)
    Dim          int       // dimensión original
    BitsPerLevel []int     // bits por nivel [4, 2, 2, 2]
}
```

#### Funciones Avanzadas

```go
// Lloyd-Max solver para centroides óptimos
func LloydMax(pdf func(float64) float64, min, max float64, levels int, iterations int) (centroids []float64, boundaries []float64, err error)

// PDF de la distribución Beta(d) post-rotación
func BetaPDF(d int) func(float64) float64

// Codebooks precomputados para TurboQuant
func TurboCodebook(d int, b int) ([]float64, error)

// Codebooks por nivel para PolarQuant
func LevelCodebook(level, n, bits int) ([]float64, error)

// Exponente sin para distribución de ángulos polar
func PolarSinExponent(level int) int

// Transformada polar directa e inversa
func PolarTransform(vec []float64, levels int) (angles [][]float64, radii []float64, err error)
func InversePolarTransform(angleCentroids [][]float64, finalRadii []float64, levels int, dim int) ([]float64, error)

// Búsqueda de centroide más cercano
func NearestCentroid(x float64, centroids []float64) int
func QuantizeWithCodebook(values []float64, centroids []float64) []int
```

---

### `sketch/` — 1-Bit Sketching (QJL)

#### QJLSketcher

```go
func NewQJLSketcher(opts QJLOptions) (*QJLSketcher, error)
```

```go
type QJLOptions struct {
    Dim       int   // dimensión del vector de entrada (> 0)
    SketchDim int   // dimensión del sketch / salida (> 0, <= Dim)
    Seed      int64 // semilla aleatoria
    OutlierK  int   // proyecciones outlier en precisión completa (0 = desactivado)
    UseSRHT   bool  // usar SRHT en lugar de Gaussiana (requiere Dim potencia de 2)
}
```

```go
func (s *QJLSketcher) Sketch(vec []float64) (*BitVector, error)
func (s *QJLSketcher) Dim() int
func (s *QJLSketcher) SketchDim() int
```

#### BitVector

```go
type BitVector struct {
    Bits           []uint64   // bits empaquetados, longitud (Dim+63)/64
    Dim            int        // número de bits significativos
    OutlierIndices []int      // índices de outliers (nil si OutlierK=0)
    OutlierValues  []float64  // valores de outliers (nil si OutlierK=0)
}
```

Implementa `encoding.BinaryMarshaler` / `BinaryUnmarshaler`.

#### Funciones de Distancia

```go
// Distancia Hamming (panics en dimension mismatch — usar en hot paths)
func HammingDistance(a, b BitVector) int

// Distancia Hamming safe (retorna error en dimension mismatch)
func HammingDistanceSafe(a, b BitVector) (int, error)

// Estimar inner product en [-1, 1]: 1 - 2 * (hamming / sketchDim)
func EstimateInnerProduct(a, b BitVector) (float64, error)
```

#### Errores

| Error | Descripción |
|-------|-------------|
| `ErrDimensionMismatch` | Dimensiones de BitVector no coinciden |
| `ErrInvalidDimension` | Dimensión <= 0 o inválida |
| `ErrInvalidConfiguration` | OutlierK > SketchDim |

#### Proyectores (Avanzado)

```go
type Projector interface {
    Project(vec []float64) ([]float64, error)
    SourceDim() int
    TargetDim() int
}

// Proyección Gaussiana densa: targetDim × sourceDim con N(0, 1/sourceDim)
func NewGaussianProjection(sourceDim, targetDim int, seed int64) (Projector, error)

// Subsampled Randomized Hadamard Transform: sign flip → FWHT → subsample
// Requiere sourceDim potencia de 2
func NewSRHT(sourceDim, targetDim int, seed int64) (Projector, error)
```

---

### `rotate/` — Transformadas y Rotaciones

```go
// Fast Walsh-Hadamard Transform in-place. O(n log n). No normalizada.
// len(x) debe ser potencia de 2. Retorna ErrNotPowerOfTwo si no.
func FWHT(x []float64) error

// Verifica si n es potencia de 2 (incluyendo n=1)
func IsPowerOfTwo(n int) bool

// Genera matriz ortogonal dim×dim aleatoria vía QR de matriz Gaussiana.
// Determinística para un rng dado.
func RandomOrthogonal(dim int, rng *rand.Rand) (*mat.Dense, error)
```

#### Errores

| Error | Descripción |
|-------|-------------|
| `ErrNotPowerOfTwo` | Dimensión del vector no es potencia de 2 |
| `ErrInvalidDimension` | Dimensión <= 0 |
| `ErrNilRNG` | RNG requerido es nil |

---

### `internal/bits/` — Empaquetamiento de Bits

```go
// Empaqueta signos (+1/-1) en []uint64. Little-endian: bit 0 = LSB.
func Pack(signs []int8) ([]uint64, error)

// Desempaqueta []uint64 a signos (+1/-1).
func Unpack(packed []uint64, n int) ([]int8, error)

// PopCount vía math/bits.OnesCount64 (hardware POPCNT).
func PopCount(bits []uint64) int
```

---

## Papers Referenciados

- **TurboQuant** — Ameli et al., *TurboQuant: Breaking the Quantization Barrier with Joint-Optimization of Random Rotation and Uniform Quantization*, ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
  - **TurboQuant_mse** — Rotación aleatoria + cuantización Lloyd-Max con codebooks Beta(d).
  - **TurboQuant_prod** — Cuantización MSE + sketch QJL del residual. Estimación de inner product UNBIASED.
- **PolarQuant** — Li et al., *PolarQuant: Quantizing Full-Range Weights and Activations for Neural Networks*, AISTATS 2026. [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
- **QJL** — Zandieh et al., *QJL: 1-Bit Quantized Johnson-Lindenstrauss Embedding for Efficient Approximate Near Neighbor Search*. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- Código de referencia QJL: [github.com/amirzandieh/QJL](https://github.com/amirzandieh/QJL)

---

## Licencia

[MIT](LICENSE)
