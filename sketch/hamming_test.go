package sketch

import (
	"math"
	"math/rand"
	"testing"
)

// --- HammingDistance ---

func TestHammingDistance_Identical(t *testing.T) {
	bv := BitVector{Bits: []uint64{0xAAAAAAAAAAAAAAAA, 0x5555555555555555}, Dim: 128}
	got := HammingDistance(bv, bv)
	if got != 0 {
		t.Errorf("HammingDistance(identical) = %d, want 0", got)
	}
}

func TestHammingDistance_Complementary(t *testing.T) {
	a := BitVector{Bits: []uint64{0x0000000000000000}, Dim: 64}
	b := BitVector{Bits: []uint64{0xFFFFFFFFFFFFFFFF}, Dim: 64}
	got := HammingDistance(a, b)
	if got != 64 {
		t.Errorf("HammingDistance(complementary) = %d, want 64", got)
	}
}

func TestHammingDistance_Symmetry(t *testing.T) {
	a := BitVector{Bits: []uint64{0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0}, Dim: 128}
	b := BitVector{Bits: []uint64{0xFF00FF00FF00FF00, 0x00FF00FF00FF00FF}, Dim: 128}
	ab := HammingDistance(a, b)
	ba := HammingDistance(b, a)
	if ab != ba {
		t.Errorf("HammingDistance not symmetric: a,b=%d, b,a=%d", ab, ba)
	}
}

func TestHammingDistance_Mismatch(t *testing.T) {
	a := BitVector{Bits: []uint64{0}, Dim: 64}
	b := BitVector{Bits: []uint64{0, 0}, Dim: 128}

	defer func() {
		if r := recover(); r == nil {
			t.Error("HammingDistance expected panic on dimension mismatch")
		}
	}()
	HammingDistance(a, b)
}

// --- HammingDistanceSafe ---

func TestHammingDistanceSafe_Mismatch(t *testing.T) {
	a := BitVector{Bits: []uint64{0}, Dim: 64}
	b := BitVector{Bits: []uint64{0, 0}, Dim: 128}
	_, err := HammingDistanceSafe(a, b)
	if err == nil {
		t.Error("HammingDistanceSafe expected error on dimension mismatch")
	}
}

func TestHammingDistanceSafe_Identical(t *testing.T) {
	bv := BitVector{Bits: []uint64{0xABCDEF0123456789}, Dim: 64}
	got, err := HammingDistanceSafe(bv, bv)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 0 {
		t.Errorf("HammingDistanceSafe(identical) = %d, want 0", got)
	}
}

// --- EstimateInnerProduct ---

func TestEstimateInnerProduct_Accuracy(t *testing.T) {
	// Simulate QJL sketching: random Gaussian projection → sign quantization.
	// The estimator 1 − 2·hamming/d has expected value (2/π)·arcsin(ρ).
	// We verify the statistical variance is small: |estimate − E[estimate]| ≤ 0.1
	// for ≥95% of trials with true cosine ≥ 0.9.
	const srcDim = 768
	const sketchDim = 256
	const trials = 1000
	const threshold = 0.1
	const minCosine = 0.9

	rng := rand.New(rand.NewSource(42))
	words := (sketchDim + 63) / 64

	proj := makeProjectionMatrix(sketchDim, srcDim, rng)

	failCount := 0
	validTrials := 0
	for i := 0; i < trials; i++ {
		vecA := make([]float64, srcDim)
		vecB := make([]float64, srcDim)
		for j := range vecA {
			vecA[j] = rng.NormFloat64()
			vecB[j] = rng.NormFloat64()
		}
		normalize(vecA)
		normalize(vecB)

		alpha := 0.7 + 0.3*rng.Float64()
		for j := range vecB {
			vecB[j] = alpha*vecA[j] + (1-alpha)*vecB[j]
		}
		normalize(vecB)

		trueCos := dot(vecA, vecB)
		if trueCos < minCosine {
			continue
		}
		validTrials++

		projA := applyProjection(proj, vecA)
		projB := applyProjection(proj, vecB)
		bvA := signVectorToBitVector(projA, words, sketchDim)
		bvB := signVectorToBitVector(projB, words, sketchDim)

		est, err := EstimateInnerProduct(bvA, bvB)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Expected value of the estimator: (2/π)·arcsin(ρ)
		expected := (2.0 / math.Pi) * math.Asin(trueCos)
		errAbs := math.Abs(est - expected)
		if errAbs > threshold {
			failCount++
		}
	}

	if validTrials == 0 {
		t.Skip("no trials with cosine ≥ 0.9")
	}
	failRate := float64(failCount) / float64(validTrials)
	if failRate > 0.05 {
		t.Errorf("too many estimation failures: %d/%d (%.1f%%), want ≤5%%", failCount, validTrials, failRate*100)
	}
}

func TestEstimateInnerProduct_Orthogonal(t *testing.T) {
	// Two orthogonal vectors projected through a random matrix → sign quantized.
	// The estimate should be within ±0.15 of 0.
	const srcDim = 768
	const sketchDim = 256
	const tolerance = 0.15
	const words = (sketchDim + 63) / 64

	rng := rand.New(rand.NewSource(123))
	proj := makeProjectionMatrix(sketchDim, srcDim, rng)

	// Construct two guaranteed-orthogonal vectors via Gram-Schmidt.
	vecA := make([]float64, srcDim)
	vecB := make([]float64, srcDim)
	for i := range vecA {
		vecA[i] = rng.NormFloat64()
		vecB[i] = rng.NormFloat64()
	}
	normalize(vecA)
	// Remove component of vecA from vecB
	ip := dot(vecA, vecB)
	for i := range vecB {
		vecB[i] -= ip * vecA[i]
	}
	normalize(vecB)

	projA := applyProjection(proj, vecA)
	projB := applyProjection(proj, vecB)
	bvA := signVectorToBitVector(projA, words, sketchDim)
	bvB := signVectorToBitVector(projB, words, sketchDim)

	est, err := EstimateInnerProduct(bvA, bvB)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if math.Abs(est) > tolerance {
		t.Errorf("EstimateInnerProduct(orthogonal) = %.4f, want within ±%.2f", est, tolerance)
	}
}

func TestEstimateInnerProduct_Mismatch(t *testing.T) {
	a := BitVector{Bits: []uint64{0}, Dim: 64}
	b := BitVector{Bits: []uint64{0, 0}, Dim: 128}
	_, err := EstimateInnerProduct(a, b)
	if err == nil {
		t.Error("EstimateInnerProduct expected error on dimension mismatch")
	}
}

// --- Benchmarks ---

func BenchmarkHammingDistance(b *testing.B) {
	benchmarks := []struct {
		name string
		dim  int
	}{
		{"256", 256},
		{"1024", 1024},
		{"4096", 4096},
	}
	for _, bm := range benchmarks {
		words := (bm.dim + 63) / 64
		bitsA := make([]uint64, words)
		bitsB := make([]uint64, words)
		for i := range bitsA {
			bitsA[i] = uint64(i) * 0x9E3779B97F4A7C15
			bitsB[i] = ^bitsA[i]
		}
		a := BitVector{Bits: bitsA, Dim: bm.dim}
		bv := BitVector{Bits: bitsB, Dim: bm.dim}

		b.Run(bm.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				HammingDistance(a, bv)
			}
		})
	}
}

// --- helpers ---

// makeProjectionMatrix generates a sketchDim × srcDim Gaussian random matrix.
func makeProjectionMatrix(sketchDim, srcDim int, rng *rand.Rand) [][]float64 {
	mat := make([][]float64, sketchDim)
	for i := range mat {
		row := make([]float64, srcDim)
		for j := range row {
			row[j] = rng.NormFloat64()
		}
		mat[i] = row
	}
	return mat
}

// applyProjection multiplies a sketchDim × srcDim matrix by a srcDim vector.
func applyProjection(mat [][]float64, vec []float64) []float64 {
	out := make([]float64, len(mat))
	for i, row := range mat {
		s := 0.0
		for j, v := range row {
			s += v * vec[j]
		}
		out[i] = s
	}
	return out
}

func normalize(v []float64) {
	n := math.Sqrt(dot(v, v))
	if n == 0 {
		return
	}
	for i := range v {
		v[i] /= n
	}
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// signVectorToBitVector converts a float64 vector to a BitVector by sign quantization.
// Positive values → bit 1, zero/negative → bit 0. Little-endian bit packing.
func signVectorToBitVector(v []float64, words, dim int) BitVector {
	bits := make([]uint64, words)
	for i, val := range v {
		if val > 0 {
			bits[i/64] |= 1 << uint(i%64)
		}
	}
	return BitVector{Bits: bits, Dim: dim}
}
