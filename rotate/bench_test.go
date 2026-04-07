package rotate

import (
	"math/rand"
	"testing"
)

func BenchmarkRandomOrthogonal(b *testing.B) {
	dims := []int{64, 256, 768}
	for _, dim := range dims {
		b.Run(benchDim(dim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				rng := rand.New(rand.NewSource(42))
				RandomOrthogonal(dim, rng) //nolint:errcheck
			}
		})
	}
}

func BenchmarkFWHT_AllDims(b *testing.B) {
	dims := []int{64, 256, 1024, 4096}
	for _, dim := range dims {
		rng := rand.New(rand.NewSource(42))
		vec := make([]float64, dim)
		for i := range vec {
			vec[i] = rng.NormFloat64()
		}

		b.Run(benchDim(dim), func(b *testing.B) {
			buf := make([]float64, dim)
			for i := 0; i < b.N; i++ {
				copy(buf, vec)
				FWHT(buf) //nolint:errcheck
			}
		})
	}
}

func benchDim(dim int) string {
	switch dim {
	case 64:
		return "64"
	case 256:
		return "256"
	case 768:
		return "768"
	case 1024:
		return "1024"
	case 4096:
		return "4096"
	default:
		return "other"
	}
}
