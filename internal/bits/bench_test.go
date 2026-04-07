package bits

import (
	"math/rand"
	"testing"
)

func BenchmarkPack(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	signs := make([]int8, 256)
	for i := range signs {
		if rng.Float64() < 0.5 {
			signs[i] = 1
		} else {
			signs[i] = -1
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Pack(signs) //nolint:errcheck
	}
}

func BenchmarkUnpack(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	signs := make([]int8, 256)
	for i := range signs {
		if rng.Float64() < 0.5 {
			signs[i] = 1
		} else {
			signs[i] = -1
		}
	}
	packed, _ := Pack(signs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Unpack(packed, 256) //nolint:errcheck
	}
}

func BenchmarkPopCount(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	data := make([]uint64, 16)
	for i := range data {
		data[i] = rng.Uint64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PopCount(data)
	}
}
