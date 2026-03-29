<p align="center">
  <strong>turboquant-go</strong><br>
  <em>Pure Go vector compression — TurboQuant, PolarQuant, QJL Sketching</em>
</p>

<p align="center">
  <a href="https://github.com/lleontor705/turboquant-go/actions/workflows/ci.yml"><img src="https://github.com/lleontor705/turboquant-go/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/lleontor705/turboquant-go/releases/latest"><img src="https://img.shields.io/github/v/release/lleontor705/turboquant-go?label=release" alt="Release" /></a>
  <a href="https://pkg.go.dev/github.com/lleontor705/turboquant-go"><img src="https://pkg.go.dev/badge/github.com/lleontor705/turboquant-go.svg" alt="Go Reference" /></a>
  <a href="https://goreportcard.com/report/github.com/lleontor705/turboquant-go"><img src="https://goreportcard.com/badge/github.com/lleontor705/turboquant-go" alt="Go Report Card" /></a>
  <a href="https://github.com/lleontor705/turboquant-go/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" /></a>
</p>

<p align="center">
  <a href="docs/OVERVIEW.md">Overview</a> •
  <a href="docs/ARCHITECTURE.md">Architecture</a> •
  <a href="docs/USAGE.md">Usage</a> •
  <a href="docs/FAQ.md">FAQ</a>
</p>

---

Implements 5 vector compression algorithms from recent Google Research papers. Single dependency (gonum), no CGO, deterministic output, concurrency-safe.

## Install

### Library

```bash
go get github.com/lleontor705/turboquant-go@latest
```

### Example binaries

```bash
# macOS / Linux
brew install lleontor705/tap/turboquant-go

# Pre-built binaries
# Download from https://github.com/lleontor705/turboquant-go/releases
```

## Algorithms

| Algorithm | Paper | Bits/coord | Compression | Cosine sim | Encode (d=768) |
|-----------|-------|:----------:|:-----------:|:----------:|:--------------:|
| **PolarQuant** | AISTATS 2026 | 3.875 | 9.1x | 0.984 | 624 us |
| **TurboQuant_mse** | ICLR 2026 | 3 | 21.3x | — | 724 us |
| **TurboQuant_prod** | ICLR 2026 | 3 | 20x | unbiased IP | 2.6 ms |
| **QJL Sketch** | arXiv:2406 | 0.33 | 192x | — | 251 us |
| **Scalar** | baseline | 8 | 8x | — | 2.9 us |

## Quick Start

### PolarQuant (recommended for most use cases)

```go
import "github.com/lleontor705/turboquant-go/quantize"

pq, _ := quantize.NewPolarQuantizer(quantize.PolarConfig{
    Dim:    768,
    Levels: 15,
    Rings:  3,
})

compressed := pq.Quantize(vector)   // []float64 -> CompressedPolar
restored  := pq.Dequantize(compressed) // CompressedPolar -> []float64
```

### QJL Sketching (maximum compression)

```go
import "github.com/lleontor705/turboquant-go/sketch"

sketcher, _ := sketch.NewQJLSketcher(sketch.QJLOptions{
    InputDim:  768,
    SketchDim: 256,
})

bv := sketcher.Sketch(vector)                    // []float64 -> BitVector
dist := sketch.HammingDistance(bv1.Bits, bv2.Bits) // fast similarity
```

### TurboQuant_mse (best rate-distortion)

```go
import "github.com/lleontor705/turboquant-go/quantize"

tq := quantize.NewTurboQuantizer(3) // 3-bit

compressed := tq.Quantize(vector)
restored  := tq.Dequantize(compressed)
```

Full examples with ANN search: [`examples/`](examples/)

## Benchmarks

AMD Ryzen 7 5800X3D, Go 1.24, Windows/amd64:

| Operation | Time |
|-----------|------|
| Hamming Distance 256-bit | 7.5 ns |
| FWHT 256-dim | 1.0 us |
| FWHT 1024-dim | 3.8 us |
| FWHT 4096-dim | 20.8 us |
| PolarQuant Encode d=768 | 624 us |
| PolarQuant Decode d=768 | 1.1 ms |
| TurboQuant Encode d=768 | 724 us |
| QJL Sketch d=768 | 251 us |

Run benchmarks locally:

```bash
make bench
```

## Project Structure

```
quantize/      TurboQuantizer, PolarQuantizer, TurboProdQuantizer, UniformQuantizer
sketch/        QJLSketcher, BitVector, HammingDistance
rotate/        FWHT, RandomOrthogonal
internal/      Bit packing, serialization
examples/      ANN search demos (ann_search, polar_search, turbo_search, prod_search)
docs/          Overview, Architecture, Usage, FAQ
```

## Development

```bash
make test        # Run all tests
make test-race   # Tests with race detector
make bench       # Run benchmarks
make vet         # go vet
make lint        # golangci-lint (install: go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest)
make fmt         # Format code
make fixtures    # Regenerate test fixtures
```

## Contributing

1. Fork the repo
2. Create a feature branch from `develop`: `git checkout -b feat/my-feature develop`
3. Make your changes and add tests
4. Run `make test-race && make vet`
5. Open a PR to `develop`

## Documentation

| Doc | Description |
|-----|-------------|
| [Overview](docs/OVERVIEW.md) | Algorithm comparison and when to use which |
| [Architecture](docs/ARCHITECTURE.md) | Package structure, data flow, wire formats |
| [Usage](docs/USAGE.md) | Code examples for all algorithms |
| [FAQ](docs/FAQ.md) | Common questions |

## References

- PolarQuant: [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- TurboQuant: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- QJL Sketching: [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)

## License

MIT
