package quantize

import (
	"fmt"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// PolarQuant benchmarks
// ---------------------------------------------------------------------------

// BenchmarkPolarQuantize measures encoding throughput for PolarQuant at d=768
// with default configuration (~3.875 bits/coord).
func BenchmarkPolarQuantize(b *testing.B) {
	const dim = 768

	cfg := DefaultPolarConfig(dim)
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	const nVecs = 1000
	vecs := generateUnitVectors(rng, nVecs, dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pq.Quantize(vecs[i%nVecs])
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPolarDequantize measures decoding throughput for PolarQuant at d=768
// with default configuration (~3.875 bits/coord).
func BenchmarkPolarDequantize(b *testing.B) {
	const dim = 768

	cfg := DefaultPolarConfig(dim)
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	const nVecs = 1000
	vecs := generateUnitVectors(rng, nVecs, dim)

	// Pre-quantize all vectors.
	cvs := make([]CompressedVector, nVecs)
	for i, v := range vecs {
		cvs[i], err = pq.Quantize(v)
		if err != nil {
			b.Fatal(err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pq.Dequantize(cvs[i%nVecs])
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPolarQuantize_Dims measures PolarQuant encoding performance across
// different vector dimensions.
func BenchmarkPolarQuantize_Dims(b *testing.B) {
	dims := []int{64, 128, 256, 512, 768}

	for _, dim := range dims {
		b.Run(fmt.Sprintf("d=%d", dim), func(b *testing.B) {
			cfg := DefaultPolarConfig(dim)
			pq, err := NewPolarQuantizer(cfg)
			if err != nil {
				b.Fatal(err)
			}

			rng := rand.New(rand.NewSource(42))
			const nVecs = 500
			vecs := generateUnitVectors(rng, nVecs, dim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := pq.Quantize(vecs[i%nVecs])
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkPolarTransform isolates the polar transform cost (without rotation
// or quantization) to measure the recursive atan2 decomposition alone.
func BenchmarkPolarTransform(b *testing.B) {
	dims := []int{128, 256, 512, 768}

	for _, dim := range dims {
		b.Run(fmt.Sprintf("d=%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			const nVecs = 500
			vecs := generateUnitVectors(rng, nVecs, dim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, err := PolarTransform(vecs[i%nVecs], 4)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
