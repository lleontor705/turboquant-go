# Code Review: turboquant-go vs TurboQuant Paper

Review against: [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## Paper Alignment

| Paper Concept | Implementation | Status |
|---|---|---|
| Random rotation (incoherence) | `rotate/orthogonal.go` via QR decomposition | Correct |
| Beta distribution codebooks | `quantize/codebooks.go` + `lloydmax.go` | Correct |
| PolarQuant (recursive polar decomposition) | `quantize/polar.go` + `polar_transform.go` | Correct |
| QJL 1-bit sketching | `sketch/qjl.go` + `sketch/jl.go` | Correct |
| TurboQuant_prod (MSE + QJL residual) | `quantize/turbo_prod.go` | Correct |
| LDLQ (data-dependent quantization) | Not implemented | Out of scope |

The implementation correctly covers the "data-free" TurboQuant algorithms. LDLQ is a data-dependent approach documented in the papers but out of scope for this library.

## Issues Found and Resolved

### HIGH Severity (Fixed)

1. **`HammingDistance()` panicked on dimension mismatch** (`sketch/hamming.go`)
   - Changed to return `(int, error)` — the safe API is now the default
   - Added `hammingDistanceUnchecked()` for internal hot paths
   - `HammingDistanceSafe()` kept as a deprecated alias

2. **`BetaPDF()` panicked on `d < 2`** (`quantize/codebooks.go`)
   - Changed signature to return `(func(float64) float64, error)`
   - All callers updated to handle the error

### MEDIUM Severity (Fixed)

3. **Matrix-vector multiply used `.At()` per-element** (`turbo.go`, `polar.go`)
   - Replaced with `gonum/mat.VecDense.MulVec()` for BLAS-backed multiply
   - Extracted `rotateForward()` and `rotateInverse()` helpers in `normalize.go`

4. **Duplicate normalization code** across 3 quantizers
   - Extracted `normalizeUnit()` helper in `quantize/normalize.go`

5. **`NearestCentroid()` used linear scan** (`lloydmax.go`)
   - Replaced with `sort.SearchFloat64s` binary search: O(k) -> O(log k)

6. **Fragile seed derivation** in `TurboProdQuantizer` (`turbo_prod.go`)
   - Replaced `seed ^ 0x5A5A...` with FNV-1a hash derivation

### LOW Severity (Fixed)

7. **Dead `max` variable** in `quantizeRadii16` (`polar.go`)
   - Removed unused computation

### Acknowledged (Not Fixed)

8. **`bytes.Buffer.Write` error handling** — structurally unreachable error branches in MarshalBinary/encode functions. These are defensive checks that `bytes.Buffer` never triggers. Kept for robustness.

9. **Wire format has no CRC/checksum** — acceptable for a library; callers handle I/O integrity.

10. **`Bits()` semantics vary across quantizers** — documented in interface contract; not changed to maintain backward compatibility.

## Coverage Improvements

| Package | Before | After | Delta |
|---------|-------:|------:|------:|
| `internal/bits` | 100.0% | 100.0% | -- |
| `internal/serial` | 91.2% | 100.0% | +8.8% |
| `rotate` | 96.7% | 100.0% | +3.3% |
| `quantize` | 88.1% | 93.1% | +5.0% |
| `sketch` | 81.9% | 96.4% | +14.5% |

The remaining uncovered code (~7% in quantize, ~4% in sketch) is exclusively:
- `bytes.Buffer.Write` error branches (structurally unreachable)
- Internal error propagation after already-validated inputs
- These are defensive checks, not missing test coverage

## New Tests Added

- `quantize/coverage_test.go` — 40+ tests covering error paths, edge cases, helpers
- `quantize/compat_test.go` — cross-quantizer compatibility verification
- `sketch/coverage_test.go` — SRHT accessors, outlier paths, serialization
- Extended: `internal/serial/serial_test.go`, `rotate/orthogonal_test.go`

## New Benchmarks Added

- `quantize/comparative_bench_test.go` — side-by-side comparison of all quantizers
- `sketch/coverage_bench_test.go` — SRHT vs Gaussian, IP estimation, serialization
- `internal/bits/bench_test.go` — Pack/Unpack/PopCount
- `rotate/bench_test.go` — RandomOrthogonal, FWHT at all dimensions
