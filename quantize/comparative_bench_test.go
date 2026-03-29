package quantize

import (
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// Missing benchmarks for coverage
// ---------------------------------------------------------------------------

func BenchmarkNearestCentroid(b *testing.B) {
	centroids := []float64{-0.8, -0.4, -0.1, 0.1, 0.4, 0.8}
	rng := rand.New(rand.NewSource(42))
	values := make([]float64, 1000)
	for i := range values {
		values[i] = rng.Float64()*2 - 1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NearestCentroid(values[i%len(values)], centroids)
	}
}

func BenchmarkPackIndices(b *testing.B) {
	indices := make([]int, 768)
	for i := range indices {
		indices[i] = i % 16
	}

	b.Run("4bit", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			packIndices(indices, 4)
		}
	})

	b.Run("2bit", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			packIndices(indices, 2)
		}
	})
}

func BenchmarkUnpackIndices(b *testing.B) {
	indices := make([]int, 768)
	for i := range indices {
		indices[i] = i % 16
	}
	packed4 := packIndices(indices, 4)
	packed2 := packIndices(indices, 2)

	b.Run("4bit", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			unpackIndices(packed4, 768, 4)
		}
	})

	b.Run("2bit", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			unpackIndices(packed2, 768, 2)
		}
	})
}

func BenchmarkLloydMax(b *testing.B) {
	pdf := func(x float64) float64 { return 1.0 }

	b.Run("4_levels", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			LloydMax(pdf, 0, 1, 4, 100) //nolint:errcheck
		}
	})

	b.Run("16_levels", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			LloydMax(pdf, 0, 1, 16, 100) //nolint:errcheck
		}
	})
}

func BenchmarkPolarTransform_Dims(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dims := []int{64, 256, 768}

	for _, dim := range dims {
		vec := make([]float64, dim)
		for i := range vec {
			vec[i] = rng.NormFloat64()
		}

		b.Run(benchDim(dim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				PolarTransform(vec, 4) //nolint:errcheck
			}
		})
	}
}

func BenchmarkInversePolarTransform(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dims := []int{64, 256, 768}

	for _, dim := range dims {
		vec := make([]float64, dim)
		for i := range vec {
			vec[i] = rng.NormFloat64()
		}
		angles, radii, _ := PolarTransform(vec, 4)
		angleCentroids := make([][]float64, len(angles))
		for i := range angles {
			angleCentroids[i] = angles[i]
		}

		b.Run(benchDim(dim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				InversePolarTransform(angleCentroids, radii, 4, dim) //nolint:errcheck
			}
		})
	}
}

func BenchmarkNormalizeUnit(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	vec := make([]float64, 768)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		normalizeUnit(vec)
	}
}

// ---------------------------------------------------------------------------
// Comparative benchmarks: all quantizers side-by-side
// ---------------------------------------------------------------------------

func BenchmarkAllQuantizers_Encode(b *testing.B) {
	const dim = 768
	const seed int64 = 42

	rng := rand.New(rand.NewSource(seed))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	b.Run("Turbo_3bit", func(b *testing.B) {
		tq, _ := NewTurboQuantizer(dim, 3, seed)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tq.Quantize(vec) //nolint:errcheck
		}
	})

	b.Run("TurboProd_3bit", func(b *testing.B) {
		tpq, _ := NewTurboProdQuantizer(dim, 3, dim, seed)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tpq.Quantize(vec) //nolint:errcheck
		}
	})

	b.Run("Polar_default", func(b *testing.B) {
		cfg := DefaultPolarConfig(dim)
		cfg.Seed = seed
		pq, _ := NewPolarQuantizer(cfg)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pq.Quantize(vec) //nolint:errcheck
		}
	})

	b.Run("Scalar_8bit", func(b *testing.B) {
		sq, _ := NewUniformQuantizer(-1, 1, 8)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sq.Quantize(vec) //nolint:errcheck
		}
	})

	b.Run("Scalar_4bit", func(b *testing.B) {
		sq, _ := NewUniformQuantizer(-1, 1, 4)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sq.Quantize(vec) //nolint:errcheck
		}
	})
}

func BenchmarkAllQuantizers_Decode(b *testing.B) {
	const dim = 768
	const seed int64 = 42

	rng := rand.New(rand.NewSource(seed))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	b.Run("Turbo_3bit", func(b *testing.B) {
		tq, _ := NewTurboQuantizer(dim, 3, seed)
		cv, _ := tq.Quantize(vec)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tq.Dequantize(cv) //nolint:errcheck
		}
	})

	b.Run("TurboProd_3bit", func(b *testing.B) {
		tpq, _ := NewTurboProdQuantizer(dim, 3, dim, seed)
		cv, _ := tpq.Quantize(vec)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tpq.Dequantize(cv) //nolint:errcheck
		}
	})

	b.Run("Polar_default", func(b *testing.B) {
		cfg := DefaultPolarConfig(dim)
		cfg.Seed = seed
		pq, _ := NewPolarQuantizer(cfg)
		cv, _ := pq.Quantize(vec)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pq.Dequantize(cv) //nolint:errcheck
		}
	})

	b.Run("Scalar_8bit", func(b *testing.B) {
		sq, _ := NewUniformQuantizer(-1, 1, 8)
		cv, _ := sq.Quantize(vec)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sq.Dequantize(cv) //nolint:errcheck
		}
	})
}

func BenchmarkAllQuantizers_RoundTrip(b *testing.B) {
	const dim = 768
	const seed int64 = 42

	rng := rand.New(rand.NewSource(seed))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	b.Run("Turbo_3bit", func(b *testing.B) {
		tq, _ := NewTurboQuantizer(dim, 3, seed)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cv, _ := tq.Quantize(vec)
			tq.Dequantize(cv) //nolint:errcheck
		}
	})

	b.Run("Polar_default", func(b *testing.B) {
		cfg := DefaultPolarConfig(dim)
		cfg.Seed = seed
		pq, _ := NewPolarQuantizer(cfg)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cv, _ := pq.Quantize(vec)
			pq.Dequantize(cv) //nolint:errcheck
		}
	})
}

func BenchmarkCompressionRatio(b *testing.B) {
	const dim = 768
	const seed int64 = 42

	rng := rand.New(rand.NewSource(seed))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	originalBytes := dim * 8 // float64

	type quantizerSetup struct {
		name    string
		quantFn func() (CompressedVector, error)
	}

	tq3, _ := NewTurboQuantizer(dim, 3, seed)
	tq2, _ := NewTurboQuantizer(dim, 2, seed)
	tq1, _ := NewTurboQuantizer(dim, 1, seed)
	sq8, _ := NewUniformQuantizer(-1, 1, 8)
	sq4, _ := NewUniformQuantizer(-1, 1, 4)
	cfg := DefaultPolarConfig(dim)
	cfg.Seed = seed
	pq, _ := NewPolarQuantizer(cfg)

	setups := []quantizerSetup{
		{"Turbo_1bit", func() (CompressedVector, error) { return tq1.Quantize(vec) }},
		{"Turbo_2bit", func() (CompressedVector, error) { return tq2.Quantize(vec) }},
		{"Turbo_3bit", func() (CompressedVector, error) { return tq3.Quantize(vec) }},
		{"Polar_default", func() (CompressedVector, error) { return pq.Quantize(vec) }},
		{"Scalar_4bit", func() (CompressedVector, error) { return sq4.Quantize(vec) }},
		{"Scalar_8bit", func() (CompressedVector, error) { return sq8.Quantize(vec) }},
	}

	for _, s := range setups {
		b.Run(s.name, func(b *testing.B) {
			cv, _ := s.quantFn()
			ratio := float64(originalBytes) / float64(len(cv.Data))
			b.ReportMetric(ratio, "compression_ratio")
			b.ReportMetric(float64(len(cv.Data)), "bytes")
			for i := 0; i < b.N; i++ {
				s.quantFn() //nolint:errcheck
			}
		})
	}
}

// benchDim returns a benchmark sub-test name for a dimension.
func benchDim(dim int) string {
	switch {
	case dim >= 1024:
		return "1024"
	case dim >= 768:
		return "768"
	case dim >= 256:
		return "256"
	case dim >= 64:
		return "64"
	default:
		return "small"
	}
}
