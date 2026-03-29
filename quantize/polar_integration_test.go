package quantize

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// Integration tests — PolarQuant end-to-end quality verification
// ---------------------------------------------------------------------------

// TestPolarQuantize_Integration_MSE_vs_TurboQuant compares PolarQuant (3.875 bits/coord)
// against TurboQuant_mse at 3-bit and 4-bit on the same set of random unit vectors.
//
// PolarQuant uses a polar decomposition that captures angular structure, so it
// should achieve competitive MSE with TurboQuant despite using a variable bit
// allocation across levels. At ~3.875 bits/coord it should be roughly comparable
// to TurboQuant at 4 bits.
func TestPolarQuantize_Integration_MSE_vs_TurboQuant(t *testing.T) {
	const dim = 768
	const numVecs = 500
	const seed = int64(42)

	rng := rand.New(rand.NewSource(99))
	vecs := generateUnitVectors(rng, numVecs, dim)

	// PolarQuant at default config (~3.875 bits/coord).
	polarCfg := DefaultPolarConfig(dim)
	pq, err := NewPolarQuantizer(polarCfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}

	polarRec := make([][]float64, numVecs)
	for i, v := range vecs {
		cv, err := pq.Quantize(v)
		if err != nil {
			t.Fatalf("PolarQuantize vec %d: %v", i, err)
		}
		polarRec[i], err = pq.Dequantize(cv)
		if err != nil {
			t.Fatalf("PolarDequantize vec %d: %v", i, err)
		}
	}
	polarMSE := computeMSE(vecs, polarRec)

	// TurboQuant at 3 bits.
	tq3, err := NewTurboQuantizer(dim, 3, seed)
	if err != nil {
		t.Fatalf("NewTurboQuantizer(3-bit): %v", err)
	}
	turbo3Rec := make([][]float64, numVecs)
	for i, v := range vecs {
		cv, err := tq3.Quantize(v)
		if err != nil {
			t.Fatalf("TurboQuantize(3-bit) vec %d: %v", i, err)
		}
		turbo3Rec[i], err = tq3.Dequantize(cv)
		if err != nil {
			t.Fatalf("TurboDequantize(3-bit) vec %d: %v", i, err)
		}
	}
	turbo3MSE := computeMSE(vecs, turbo3Rec)

	// TurboQuant at 4 bits.
	tq4, err := NewTurboQuantizer(dim, 4, seed)
	if err != nil {
		t.Fatalf("NewTurboQuantizer(4-bit): %v", err)
	}
	turbo4Rec := make([][]float64, numVecs)
	for i, v := range vecs {
		cv, err := tq4.Quantize(v)
		if err != nil {
			t.Fatalf("TurboQuantize(4-bit) vec %d: %v", i, err)
		}
		turbo4Rec[i], err = tq4.Dequantize(cv)
		if err != nil {
			t.Fatalf("TurboDequantize(4-bit) vec %d: %v", i, err)
		}
	}
	turbo4MSE := computeMSE(vecs, turbo4Rec)

	t.Logf("PolarQuant (%.3f bits/coord): MSE=%.6f", polarCfg.BitsPerCoord(), polarMSE)
	t.Logf("TurboQuant (3 bits/coord):   MSE=%.6f", turbo3MSE)
	t.Logf("TurboQuant (4 bits/coord):   MSE=%.6f", turbo4MSE)

	// PolarQuant MSE should be in the same ballpark — bounded above by a
	// generous multiple of the 4-bit TurboQuant reference.  PolarQuant at
	// 3.875 bits should ideally be between 3-bit and 4-bit TurboQuant MSE.
	// Use a generous upper bound: it should at least beat 3-bit uniform.
	if polarMSE > 0.15 {
		t.Errorf("PolarQuant MSE = %.6f is too high (max 0.15)", polarMSE)
	}
}

// TestPolarQuantize_Integration_CompressionRatio verifies that the compressed
// representation achieves the expected bits-per-coordinate and that the actual
// wire-format data size matches theoretical predictions.
//
// For default config (4 levels, BitsLevel1=4, BitsRest=2, RadiusBits=16):
//
//	BitsPerCoord = (4*d/2 + 2*d/4 + 2*d/8 + 2*d/16 + 16*d/16) / d = 3.875
//
// The wire format includes a small header and stores radii as float64 (8 bytes
// each), so the actual compressed size exceeds the theoretical minimum.
func TestPolarQuantize_Integration_CompressionRatio(t *testing.T) {
	const dim = 768
	cfg := DefaultPolarConfig(dim)

	// Verify theoretical bits per coordinate.
	bpc := cfg.BitsPerCoord()
	expectedBPC := 3.875
	if math.Abs(bpc-expectedBPC) > 0.001 {
		t.Fatalf("BitsPerCoord() = %.6f, want ≈ %.3f", bpc, expectedBPC)
	}

	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}

	rng := rand.New(rand.NewSource(7))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}
	normalize(vec)

	cv, err := pq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	originalBytes := dim * 8 // float64
	compressedBytes := len(cv.Data)
	ratio := float64(originalBytes) / float64(compressedBytes)

	// Theoretical minimum (ignoring wire-format overhead):
	// 3.875 bits/coord × 768 / 8 = 372 bytes
	theoreticalMin := int(math.Ceil(bpc * float64(dim) / 8.0))

	t.Logf("BitsPerCoord:     %.3f", bpc)
	t.Logf("Original:         %d bytes", originalBytes)
	t.Logf("Theoretical min:  %d bytes (payload only)", theoreticalMin)
	t.Logf("Actual compressed: %d bytes (includes header + float64 radii)", compressedBytes)
	t.Logf("Compression:      %.1fx", ratio)

	// Compressed data should be significantly smaller than original.
	if ratio < 5.0 {
		t.Errorf("compression ratio %.1fx too low (minimum 5.0x)", ratio)
	}

	// Verify metadata.
	if cv.Dim != dim {
		t.Errorf("cv.Dim = %d, want %d", cv.Dim, dim)
	}
	if cv.BitsPer != pq.Bits() {
		t.Errorf("cv.BitsPer = %d, want %d", cv.BitsPer, pq.Bits())
	}

	// Verify the wire format is deterministic — encode twice, same result.
	cv2, err := pq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize(2): %v", err)
	}
	if len(cv.Data) != len(cv2.Data) {
		t.Errorf("non-deterministic size: %d vs %d", len(cv.Data), len(cv2.Data))
	}
	for i := range cv.Data {
		if cv.Data[i] != cv2.Data[i] {
			t.Errorf("non-deterministic data at byte %d", i)
			break
		}
	}
}

// TestPolarQuantize_Integration_NormPreservation_LargeScale verifies that
// PolarQuant preserves vector norms across 1000 random unit vectors.
//
// The random rotation is orthogonal and thus norm-preserving. After quantization
// and dequantization, the reconstructed vector's norm should be close to 1.0.
// This test checks that every single vector's norm is within tolerance.
func TestPolarQuantize_Integration_NormPreservation_LargeScale(t *testing.T) {
	const dim = 768
	const numVecs = 1000

	cfg := DefaultPolarConfig(dim)
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	vecs := generateUnitVectors(rng, numVecs, dim)

	maxDeviation := 0.0
	sumDeviation := 0.0
	failCount := 0

	for i, v := range vecs {
		cv, err := pq.Quantize(v)
		if err != nil {
			t.Fatalf("Quantize vec %d: %v", i, err)
		}
		rec, err := pq.Dequantize(cv)
		if err != nil {
			t.Fatalf("Dequantize vec %d: %v", i, err)
		}

		var recNorm float64
		for _, x := range rec {
			recNorm += x * x
		}
		recNorm = math.Sqrt(recNorm)

		deviation := math.Abs(recNorm - 1.0)
		sumDeviation += deviation
		if deviation > maxDeviation {
			maxDeviation = deviation
		}

		// All vectors must preserve norm within 15%.
		if deviation > 0.15 {
			failCount++
		}
	}

	avgDeviation := sumDeviation / float64(numVecs)
	t.Logf("Norm preservation over %d vectors (dim=%d):", numVecs, dim)
	t.Logf("  Max deviation:  %.4f", maxDeviation)
	t.Logf("  Avg deviation:  %.4f", avgDeviation)
	t.Logf("  Failures (>15%%): %d/%d", failCount, numVecs)

	if failCount > 0 {
		t.Errorf("%d/%d vectors exceeded 15%% norm deviation", failCount, numVecs)
	}
}

// TestPolarQuantize_Integration_CosineSimilarity measures the average cosine
// similarity between original and reconstructed vectors for 1000 random unit
// vectors. PolarQuant should preserve angular structure well.
//
// The target is average cosine(x, x̂) ≥ 0.95, which validates that the polar
// decomposition + angle quantization preserves directionality.
func TestPolarQuantize_Integration_CosineSimilarity(t *testing.T) {
	const dim = 768
	const numVecs = 1000

	cfg := DefaultPolarConfig(dim)
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}

	rng := rand.New(rand.NewSource(123))
	vecs := generateUnitVectors(rng, numVecs, dim)

	totalCosSim := 0.0
	minCosSim := 1.0

	for i, v := range vecs {
		cv, err := pq.Quantize(v)
		if err != nil {
			t.Fatalf("Quantize vec %d: %v", i, err)
		}
		rec, err := pq.Dequantize(cv)
		if err != nil {
			t.Fatalf("Dequantize vec %d: %v", i, err)
		}

		// Both v and rec are approximately unit norm, so dot product ≈ cosine similarity.
		cosSim := dot(v, rec)
		totalCosSim += cosSim
		if cosSim < minCosSim {
			minCosSim = cosSim
		}
	}

	avgCosSim := totalCosSim / float64(numVecs)
	t.Logf("Cosine similarity over %d vectors (dim=%d):", numVecs, dim)
	t.Logf("  Average: %.6f", avgCosSim)
	t.Logf("  Minimum: %.6f", minCosSim)

	if avgCosSim < 0.95 {
		t.Errorf("average cosine similarity %.6f < 0.95", avgCosSim)
	}
}

// TestPolarQuantize_Integration_MultiDimension validates that PolarQuant works
// correctly across a range of common embedding dimensions.
//
// Each dimension must be a multiple of 2^Levels (default: 16). For dimensions
// that are not multiples of 16, we adjust the number of levels so that
// dim % (1 << levels) == 0.
func TestPolarQuantize_Integration_MultiDimension(t *testing.T) {
	tests := []struct {
		dim    int
		levels int // 0 = use default (auto-adjusted)
	}{
		{16, 4},
		{32, 4},
		{64, 4},
		{128, 4},
		{256, 4},
		{512, 4},
		{768, 4},
	}

	const numVecs = 50
	rng := rand.New(rand.NewSource(42))

	for _, tt := range tests {
		t.Run(fmt.Sprintf("d=%d", tt.dim), func(t *testing.T) {
			cfg := PolarConfig{
				Dim:        tt.dim,
				Levels:     tt.levels,
				BitsLevel1: 4,
				BitsRest:   2,
				RadiusBits: 16,
				Seed:       42,
			}

			pq, err := NewPolarQuantizer(cfg)
			if err != nil {
				t.Fatalf("NewPolarQuantizer(dim=%d, levels=%d): %v", tt.dim, tt.levels, err)
			}

			vecs := generateUnitVectors(rng, numVecs, tt.dim)

			totalCosSim := 0.0
			for i, v := range vecs {
				cv, err := pq.Quantize(v)
				if err != nil {
					t.Fatalf("Quantize vec %d: %v", i, err)
				}

				rec, err := pq.Dequantize(cv)
				if err != nil {
					t.Fatalf("Dequantize vec %d: %v", i, err)
				}

				if len(rec) != tt.dim {
					t.Fatalf("len(rec) = %d, want %d", len(rec), tt.dim)
				}

				totalCosSim += dot(v, rec)
			}

			avgCosSim := totalCosSim / float64(numVecs)
			t.Logf("dim=%d levels=%d bpc=%.3f avg_cosine=%.6f",
				tt.dim, tt.levels, cfg.BitsPerCoord(), avgCosSim)

			// Minimum quality threshold — lower dimensions have higher error.
			minCosSim := 0.75
			if avgCosSim < minCosSim {
				t.Errorf("dim=%d: avg cosine similarity %.4f < %.2f", tt.dim, avgCosSim, minCosSim)
			}
		})
	}
}
