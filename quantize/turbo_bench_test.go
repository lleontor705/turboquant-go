package quantize

import (
	"fmt"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// TurboQuant_mse benchmarks
// ---------------------------------------------------------------------------

// BenchmarkTurboQuantize measures encoding throughput for d=768, b=3.
func BenchmarkTurboQuantize(b *testing.B) {
	const dim = 768
	const bits = 3

	tq, err := NewTurboQuantizer(dim, bits, 42)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	const nVecs = 1000
	vecs := generateUnitVectors(rng, nVecs, dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := tq.Quantize(vecs[i%nVecs])
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTurboDequantize measures decoding throughput for d=768, b=3.
func BenchmarkTurboDequantize(b *testing.B) {
	const dim = 768
	const bits = 3

	tq, err := NewTurboQuantizer(dim, bits, 42)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	const nVecs = 1000
	vecs := generateUnitVectors(rng, nVecs, dim)

	// Pre-quantize all vectors.
	cvs := make([]CompressedVector, nVecs)
	for i, v := range vecs {
		cvs[i], err = tq.Quantize(v)
		if err != nil {
			b.Fatal(err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := tq.Dequantize(cvs[i%nVecs])
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTurboQuantize_Dims measures encoding performance across different
// vector dimensions (128, 256, 512, 768, 1024).
func BenchmarkTurboQuantize_Dims(b *testing.B) {
	dims := []int{128, 256, 512, 768, 1024}

	for _, dim := range dims {
		b.Run(fmt.Sprintf("d=%d", dim), func(b *testing.B) {
			const bits = 3
			tq, err := NewTurboQuantizer(dim, bits, 42)
			if err != nil {
				b.Fatal(err)
			}

			rng := rand.New(rand.NewSource(42))
			const nVecs = 500
			vecs := generateUnitVectors(rng, nVecs, dim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := tq.Quantize(vecs[i%nVecs])
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkTurboQuantize_Bits measures encoding performance across different
// bit widths (1, 2, 3, 4).
func BenchmarkTurboQuantize_Bits(b *testing.B) {
	bitWidths := []int{1, 2, 3, 4}

	for _, bits := range bitWidths {
		b.Run(fmt.Sprintf("b=%d", bits), func(b *testing.B) {
			const dim = 768
			tq, err := NewTurboQuantizer(dim, bits, 42)
			if err != nil {
				b.Fatal(err)
			}

			rng := rand.New(rand.NewSource(42))
			const nVecs = 500
			vecs := generateUnitVectors(rng, nVecs, dim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := tq.Quantize(vecs[i%nVecs])
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkTurboCodebook measures codebook lookup performance
// (NearestCentroid) for different codebook sizes.
func BenchmarkTurboCodebook(b *testing.B) {
	dims := []int{128, 768}
	bitWidths := []int{1, 2, 3, 4}

	for _, dim := range dims {
		for _, bits := range bitWidths {
			b.Run(fmt.Sprintf("d=%d_b=%d", dim, bits), func(b *testing.B) {
				cb, err := TurboCodebook(dim, bits)
				if err != nil {
					b.Fatal(err)
				}

				// Generate random values in [-1, 1].
				rng := rand.New(rand.NewSource(42))
				const nVals = 10000
				vals := make([]float64, nVals)
				for i := range vals {
					vals[i] = 2.0*rng.Float64() - 1.0
				}

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					NearestCentroid(vals[i%nVals], cb)
				}
			})
		}
	}
}

// BenchmarkTurboQuantize_Dequantize_RoundTrip measures the full encode→decode
// pipeline, which is the typical use case for approximate nearest neighbor search.
func BenchmarkTurboQuantize_Dequantize_RoundTrip(b *testing.B) {
	const dim = 768
	const bits = 3

	tq, err := NewTurboQuantizer(dim, bits, 42)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	const nVecs = 500
	vecs := generateUnitVectors(rng, nVecs, dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cv, err := tq.Quantize(vecs[i%nVecs])
		if err != nil {
			b.Fatal(err)
		}
		_, err = tq.Dequantize(cv)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTurboQuantize_Parallel measures encoding throughput under concurrent
// load (GOMAXPROCS goroutines).
func BenchmarkTurboQuantize_Parallel(b *testing.B) {
	const dim = 768
	const bits = 3

	tq, err := NewTurboQuantizer(dim, bits, 42)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	const nVecs = 500
	vecs := generateUnitVectors(rng, nVecs, dim)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			_, err := tq.Quantize(vecs[i%nVecs])
			if err != nil {
				b.Fatal(err)
			}
			i++
		}
	})
}
