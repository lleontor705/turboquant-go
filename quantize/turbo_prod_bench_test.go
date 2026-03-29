package quantize

import (
	"fmt"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// TurboQuant_prod benchmarks
// ---------------------------------------------------------------------------

// BenchmarkTurboProd_Quantize measures encoding throughput for d=768, bits=3.
func BenchmarkTurboProd_Quantize(b *testing.B) {
	const dim = 768
	const bits = 3 // 2-bit MSE + 1-bit QJL

	tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	const nVecs = 500
	vecs := generateUnitVectors(rng, nVecs, dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := tpq.Quantize(vecs[i%nVecs])
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTurboProd_Dequantize measures decoding throughput for d=768, bits=3.
func BenchmarkTurboProd_Dequantize(b *testing.B) {
	const dim = 768
	const bits = 3

	tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	const nVecs = 500
	vecs := generateUnitVectors(rng, nVecs, dim)

	// Pre-quantize all vectors.
	cvs := make([]CompressedVector, nVecs)
	for i, v := range vecs {
		cvs[i], err = tpq.Quantize(v)
		if err != nil {
			b.Fatal(err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := tpq.Dequantize(cvs[i%nVecs])
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTurboProd_EstimateIP measures inner product estimation throughput
// for d=768, bits=3. This is the critical operation for ANN search ranking.
func BenchmarkTurboProd_EstimateIP(b *testing.B) {
	const dim = 768
	const bits = 3

	tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	const nVecs = 500
	vecs := generateUnitVectors(rng, nVecs, dim)

	// Pre-quantize all vectors.
	cvs := make([]CompressedVector, nVecs)
	for i, v := range vecs {
		cvs[i], err = tpq.Quantize(v)
		if err != nil {
			b.Fatal(err)
		}
	}

	// Generate query vectors.
	const nQueries = 50
	queries := generateUnitVectors(rng, nQueries, dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := tpq.EstimateInnerProduct(queries[i%nQueries], cvs[i%nVecs])
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTurboProd_Dims measures encoding performance across different
// vector dimensions (128, 256, 512, 768).
func BenchmarkTurboProd_Dims(b *testing.B) {
	dims := []int{128, 256, 512, 768}

	for _, dim := range dims {
		b.Run(fmt.Sprintf("d=%d", dim), func(b *testing.B) {
			const bits = 3
			tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
			if err != nil {
				b.Fatal(err)
			}

			rng := rand.New(rand.NewSource(42))
			const nVecs = 200
			vecs := generateUnitVectors(rng, nVecs, dim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := tpq.Quantize(vecs[i%nVecs])
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkTurboProd_Bits measures encoding performance across different
// bit widths (2, 3, 4).
func BenchmarkTurboProd_Bits(b *testing.B) {
	bitWidths := []int{2, 3, 4}

	for _, bits := range bitWidths {
		b.Run(fmt.Sprintf("b=%d", bits), func(b *testing.B) {
			const dim = 768
			tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
			if err != nil {
				b.Fatal(err)
			}

			rng := rand.New(rand.NewSource(42))
			const nVecs = 200
			vecs := generateUnitVectors(rng, nVecs, dim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := tpq.Quantize(vecs[i%nVecs])
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
