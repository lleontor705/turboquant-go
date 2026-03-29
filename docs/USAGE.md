# Uso

Este documento muestra ejemplos mínimos de uso para cada algoritmo.

> Todos los imports usan el módulo: `github.com/lleontor705/turboquant-go`.

## PolarQuant (recomendado)

```go
package main

import (
    "math"

    "github.com/lleontor705/turboquant-go/quantize"
)

func main() {
    config := quantize.DefaultPolarConfig(768)
    pq, err := quantize.NewPolarQuantizer(config)
    if err != nil {
        panic(err)
    }

    vec := make([]float64, 768)
    for i := range vec {
        vec[i] = math.Sin(float64(i) * 0.01)
    }

    cv, err := pq.Quantize(vec)
    if err != nil {
        panic(err)
    }

    recovered, err := pq.Dequantize(cv)
    if err != nil {
        panic(err)
    }

    _ = recovered
}
```

## TurboQuant_mse

```go
package main

import (
    "math"

    "github.com/lleontor705/turboquant-go/quantize"
)

func main() {
    tq, err := quantize.NewTurboQuantizer(768, 4, 42)
    if err != nil {
        panic(err)
    }

    vec := make([]float64, 768)
    for i := range vec {
        vec[i] = math.Cos(float64(i) * 0.02)
    }

    cv, err := tq.Quantize(vec)
    if err != nil {
        panic(err)
    }

    recovered, err := tq.Dequantize(cv)
    if err != nil {
        panic(err)
    }

    _ = recovered
}
```

## TurboQuant_prod (inner product unbiased)

```go
package main

import (
    "math"
    "math/rand"

    "github.com/lleontor705/turboquant-go/quantize"
)

func main() {
    const dim = 768
    const bits = 3

    q, err := quantize.NewTurboProdQuantizer(dim, bits, dim, 42)
    if err != nil {
        panic(err)
    }

    rng := rand.New(rand.NewSource(123))
    x := randomUnitVector(rng, dim)
    query := randomUnitVector(rng, dim)

    compressed, err := q.Quantize(x)
    if err != nil {
        panic(err)
    }

    // Reconstrucción MSE (el residual es 1-bit)
    reconstructed, err := q.Dequantize(compressed)
    if err != nil {
        panic(err)
    }
    _ = reconstructed

    // IP unbiased
    ipEstimated, err := q.EstimateInnerProduct(query, compressed)
    if err != nil {
        panic(err)
    }

    _ = ipEstimated
}

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
```

## QJL Sketching

```go
package main

import (
    "math"

    "github.com/lleontor705/turboquant-go/sketch"
)

func main() {
    sketcher, err := sketch.NewQJLSketcher(sketch.QJLOptions{
        Dim:       768,
        SketchDim: 256,
        Seed:      42,
    })
    if err != nil {
        panic(err)
    }

    vecA := make([]float64, 768)
    vecB := make([]float64, 768)
    for i := range vecA {
        vecA[i] = math.Sin(float64(i) * 0.01)
        vecB[i] = math.Cos(float64(i) * 0.01)
    }

    a, err := sketcher.Sketch(vecA)
    if err != nil {
        panic(err)
    }
    b, err := sketcher.Sketch(vecB)
    if err != nil {
        panic(err)
    }

    dist := sketch.HammingDistance(*a, *b)
    _ = dist

    ip, err := sketch.EstimateInnerProduct(*a, *b)
    if err != nil {
        panic(err)
    }
    _ = ip
}
```

## Cuantización Escalar Uniforme

```go
package main

import "github.com/lleontor705/turboquant-go/quantize"

func main() {
    q, err := quantize.NewUniformQuantizer(0, 10, 8)
    if err != nil {
        panic(err)
    }

    vec := []float64{1.2, 3.4, 5.6, 7.8, 9.9}
    cv, err := q.Quantize(vec)
    if err != nil {
        panic(err)
    }

    recovered, err := q.Dequantize(cv)
    if err != nil {
        panic(err)
    }

    _ = recovered
}
```

## Rotaciones / FWHT

```go
package main

import (
    "github.com/lleontor705/turboquant-go/rotate"
)

func main() {
    x := []float64{1, 2, 3, 4}
    if err := rotate.FWHT(x); err != nil {
        panic(err)
    }

    // x contiene la transformada in-place
}
```
