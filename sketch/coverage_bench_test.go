package sketch

import (
	"math/rand"
	"testing"
)

func BenchmarkSRHT_Project(b *testing.B) {
	dims := []int{64, 256, 1024}
	for _, dim := range dims {
		proj, _ := NewSRHT(dim, dim/2, 42)
		rng := rand.New(rand.NewSource(42))
		vec := make([]float64, dim)
		for i := range vec {
			vec[i] = rng.NormFloat64()
		}

		b.Run(benchDimStr(dim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				proj.Project(vec) //nolint:errcheck
			}
		})
	}
}

func BenchmarkGaussianProjection(b *testing.B) {
	dims := []int{64, 256, 768}
	for _, dim := range dims {
		proj, _ := NewGaussianProjection(dim, dim/2, 42)
		rng := rand.New(rand.NewSource(42))
		vec := make([]float64, dim)
		for i := range vec {
			vec[i] = rng.NormFloat64()
		}

		b.Run(benchDimStr(dim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				proj.Project(vec) //nolint:errcheck
			}
		})
	}
}

func BenchmarkEstimateInnerProduct(b *testing.B) {
	const dim = 256
	words := (dim + 63) / 64
	rng := rand.New(rand.NewSource(42))
	bitsA := make([]uint64, words)
	bitsB := make([]uint64, words)
	for i := range bitsA {
		bitsA[i] = rng.Uint64()
		bitsB[i] = rng.Uint64()
	}
	a := BitVector{Bits: bitsA, Dim: dim}
	bv := BitVector{Bits: bitsB, Dim: dim}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EstimateInnerProduct(a, bv) //nolint:errcheck
	}
}

func BenchmarkBitVector_MarshalBinary(b *testing.B) {
	const dim = 256
	words := (dim + 63) / 64
	rng := rand.New(rand.NewSource(42))
	bits := make([]uint64, words)
	for i := range bits {
		bits[i] = rng.Uint64()
	}
	bv := BitVector{Bits: bits, Dim: dim}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bv.MarshalBinary() //nolint:errcheck
	}
}

func BenchmarkBitVector_UnmarshalBinary(b *testing.B) {
	const dim = 256
	words := (dim + 63) / 64
	rng := rand.New(rand.NewSource(42))
	bits := make([]uint64, words)
	for i := range bits {
		bits[i] = rng.Uint64()
	}
	bv := BitVector{Bits: bits, Dim: dim}
	data, _ := bv.MarshalBinary()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var dst BitVector
		dst.UnmarshalBinary(data) //nolint:errcheck
	}
}

func benchDimStr(dim int) string {
	switch dim {
	case 64:
		return "64"
	case 256:
		return "256"
	case 768:
		return "768"
	case 1024:
		return "1024"
	default:
		return "other"
	}
}
