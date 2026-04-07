# TurboQuant-Go Benchmarks

## Environment

- **CPU:** 13th Gen Intel Core i7-1360P
- **OS:** Windows 11 Enterprise (amd64)
- **Go:** 1.24.0
- **Date:** 2026-03-29

## Quantization Encode Performance (dim=768)

| Algorithm | ns/op | B/op | allocs/op |
|-----------|------:|-----:|----------:|
| Scalar 8-bit | 2,970 | 768 | 1 |
| Scalar 4-bit | 3,489 | 384 | 1 |
| **Turbo 3-bit** | **114,324** | **19,200** | **8** |
| **Polar default** | **129,905** | **40,600** | **29** |
| TurboProd 3-bit | 821,004 | 57,864 | 25 |

## Quantization Decode Performance (dim=768)

| Algorithm | ns/op | B/op | allocs/op |
|-----------|------:|-----:|----------:|
| Scalar 8-bit | 1,193 | 6,144 | 1 |
| **Turbo 3-bit** | **102,997** | **18,888** | **9** |
| **TurboProd 3-bit** | **103,016** | **19,160** | **13** |
| Polar default | 110,237 | 30,592 | 23 |

## Compression Ratios (dim=768, original=6144 bytes)

| Algorithm | Compressed Bytes | Ratio | Bits/Coord |
|-----------|----------------:|---------:|-----------:|
| Turbo 1-bit | 105 | 58.5x | 1.0 |
| Turbo 2-bit | 201 | 30.6x | 2.0 |
| Turbo 3-bit | 297 | 20.7x | 3.0 |
| Scalar 4-bit | 384 | 16.0x | 4.0 |
| Polar default | 393 | 15.6x | 3.875 |
| Scalar 8-bit | 768 | 8.0x | 8.0 |

## Dimension Scaling (Turbo 3-bit)

| Dimension | Encode ns/op | Decode ns/op |
|----------:|-------------:|-------------:|
| 128 | 6,101 | - |
| 256 | 15,813 | - |
| 512 | 56,323 | - |
| 768 | 110,842 | 82,824 |
| 1024 | 221,330 | - |

Scaling is approximately O(d^2) due to the d x d rotation matrix multiply.

## TurboProd Inner Product Estimation

| Operation | dim=768 ns/op |
|-----------|-------------:|
| Quantize | 821,004 |
| Dequantize (MSE only) | 85,881 |
| EstimateInnerProduct | 631,654 |

## Transform Performance

| Transform | dim=256 ns/op | dim=1024 ns/op | dim=4096 ns/op |
|-----------|-------------:|---------------:|---------------:|
| FWHT | 898 | 4,285 | 18,631 |
| Polar Transform | 3,765 | - | - |
| Inverse Polar Transform | 3,574 | - | - |

## Projection Performance (sketchDim = dim/2)

| Method | dim=64 ns/op | dim=256 ns/op | dim=768 ns/op |
|--------|------------:|-------------:|-------------:|
| SRHT | 348 | 1,343 | - |
| Gaussian | 2,219 | 31,767 | 266,071 |

SRHT is O(n log n) vs Gaussian O(n^2).

## Bit Operations

| Operation | ns/op | Notes |
|-----------|------:|-------|
| Pack (256 signs) | 167 | 0 allocs |
| Unpack (256 signs) | 219 | 1 alloc |
| PopCount (16 words) | 4.2 | hardware POPCNT |
| Hamming Distance (256-bit) | 3.7 | hardware POPCNT |
| Hamming Distance (4096-bit) | 37.3 | hardware POPCNT |

## Codebook Lookup (cached)

| Config | ns/op |
|--------|------:|
| d=768 b=1 | 6.9 |
| d=768 b=2 | 8.2 |
| d=768 b=3 | 9.6 |
| d=768 b=4 | 11.0 |

## Matrix Generation

| Operation | dim=64 ns/op | dim=256 ns/op | dim=768 ns/op |
|-----------|------------:|-------------:|-------------:|
| RandomOrthogonal | 107,842 | 5,313,305 | 77,975,800 |
| Lloyd-Max (4 levels) | 1,071,161 | - | - |
| Lloyd-Max (16 levels) | 4,250,868 | - | - |

Note: RandomOrthogonal and Lloyd-Max are constructor-time costs. They happen once during quantizer creation and are amortized across all subsequent Quantize/Dequantize calls.
