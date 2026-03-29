package quantize

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// Integration tests — end-to-end quality verification
// ---------------------------------------------------------------------------

// TestTurboQuantize_MSE_TheoreticalBounds verifies that measured MSE stays
// within a tolerance factor of the theoretical MSE for TurboQuant_mse.
//
// After a random rotation of a unit vector in d dimensions, each coordinate
// follows a Beta distribution. The theoretical MSE per coordinate for optimal
// b-bit quantization of this distribution provides a lower bound; we check
// that actual MSE is within 2x of these theoretical values.
//
// Theoretical MSE targets (approximate):
//   - b=3: ~0.03  → tolerance: 0.06
//   - b=4: ~0.009 → tolerance: 0.018
func TestTurboQuantize_MSE_TheoreticalBounds(t *testing.T) {
	const dim = 768
	const numVecs = 1000
	const seed = int64(42)

	tests := []struct {
		bits       int
		maxMSE     float64
		theoryDesc string
	}{
		{3, 0.06, "0.03×2.0"},
		{4, 0.018, "0.009×2.0"},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("b=%d", tt.bits), func(t *testing.T) {
			tq, err := NewTurboQuantizer(dim, tt.bits, seed)
			if err != nil {
				t.Fatalf("NewTurboQuantizer(%d, %d): %v", dim, tt.bits, err)
			}

			rng := rand.New(rand.NewSource(99))
			vecs := generateUnitVectors(rng, numVecs, dim)

			reconstructed := make([][]float64, numVecs)
			for i, v := range vecs {
				cv, err := tq.Quantize(v)
				if err != nil {
					t.Fatalf("Quantize vec %d: %v", i, err)
				}
				rec, err := tq.Dequantize(cv)
				if err != nil {
					t.Fatalf("Dequantize vec %d: %v", i, err)
				}
				reconstructed[i] = rec
			}

			mse := computeMSE(vecs, reconstructed)
			t.Logf("bits=%d dim=%d numVecs=%d MSE=%.6f (bound=%s=%.3f)",
				tt.bits, dim, numVecs, mse, tt.theoryDesc, tt.maxMSE)

			if mse > tt.maxMSE {
				t.Errorf("MSE = %.6f exceeds theoretical bound %.6f (%s)",
					mse, tt.maxMSE, tt.theoryDesc)
			}
		})
	}
}

// TestTurboQuantize_CompressionRatio verifies that the compressed representation
// uses the expected number of bits: dim × bits_per_coordinate plus the wire header.
//
// For d=768, b=3: total bits = 768×3 = 2304 → ⌈2304/8⌉ = 288 bytes payload.
// Wire format adds 9 bytes (version + norm), so payload = 288 + 9 = 297 bytes.
// Original float64: 768×8 = 6144 bytes.
// Compression ratio: 6144 / 297 ≈ 20.7x.
func TestTurboQuantize_CompressionRatio(t *testing.T) {
	const dim = 768

	tests := []struct {
		bits            int
		expectedPayload int // expected compressed data size in bytes
	}{
		{1, 9 + (768*1+7)/8}, // 96 bytes + header
		{2, 9 + (768*2+7)/8}, // 192 bytes + header
		{3, 9 + (768*3+7)/8}, // 288 bytes + header
		{4, 9 + (768*4+7)/8}, // 384 bytes + header
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("b=%d", tt.bits), func(t *testing.T) {
			tq, err := NewTurboQuantizer(dim, tt.bits, 42)
			if err != nil {
				t.Fatal(err)
			}

			rng := rand.New(rand.NewSource(7))
			vec := make([]float64, dim)
			for i := range vec {
				vec[i] = rng.NormFloat64()
			}

			cv, err := tq.Quantize(vec)
			if err != nil {
				t.Fatal(err)
			}

			originalBytes := dim * 8 // float64 = 8 bytes per element
			compressedBytes := len(cv.Data)
			ratio := float64(originalBytes) / float64(compressedBytes)

			t.Logf("bits=%d: payload=%d bytes (expected %d), original=%d bytes, ratio=%.1fx",
				tt.bits, compressedBytes, tt.expectedPayload, originalBytes, ratio)

			if compressedBytes != tt.expectedPayload {
				t.Errorf("compressed data size = %d, expected %d",
					compressedBytes, tt.expectedPayload)
			}

			// Verify CompressedVector metadata.
			if cv.Dim != dim {
				t.Errorf("cv.Dim = %d, want %d", cv.Dim, dim)
			}
			if cv.BitsPer != tt.bits {
				t.Errorf("cv.BitsPer = %d, want %d", cv.BitsPer, tt.bits)
			}

			// Report compression ratio.
			t.Logf("Compression: %d bytes → %d bytes (%.1fx)",
				originalBytes, compressedBytes, ratio)
		})
	}
}

// TestTurboQuantize_NormPreservation verifies that after encode+decode,
// the reconstructed vector's norm is close to the original unit norm.
//
// The random rotation is orthogonal, so it preserves norms. After quantization
// and dequantization, there is some distortion, but the norm should remain
// close to 1.0 for unit-norm inputs. This validates that the
// rotation→quantize→inverse-rotation pipeline preserves geometric structure.
func TestTurboQuantize_NormPreservation(t *testing.T) {
	const dim = 768
	const numVecs = 200
	const seed = int64(42)

	for _, bits := range []int{2, 3, 4} {
		t.Run(fmt.Sprintf("b=%d", bits), func(t *testing.T) {
			tq, err := NewTurboQuantizer(dim, bits, seed)
			if err != nil {
				t.Fatal(err)
			}

			rng := rand.New(rand.NewSource(123))
			vecs := generateUnitVectors(rng, numVecs, dim)

			maxDeviation := 0.0
			for i, v := range vecs {
				cv, err := tq.Quantize(v)
				if err != nil {
					t.Fatalf("Quantize vec %d: %v", i, err)
				}
				rec, err := tq.Dequantize(cv)
				if err != nil {
					t.Fatalf("Dequantize vec %d: %v", i, err)
				}

				var recNorm float64
				for _, x := range rec {
					recNorm += x * x
				}
				recNorm = math.Sqrt(recNorm)

				deviation := math.Abs(recNorm - 1.0)
				if deviation > maxDeviation {
					maxDeviation = deviation
				}
			}

			// With more bits, we expect better norm preservation.
			// Tolerance: 15% for 2-bit, 10% for 3-bit, 8% for 4-bit.
			tolerance := map[int]float64{2: 0.15, 3: 0.10, 4: 0.08}[bits]
			t.Logf("bits=%d: max norm deviation = %.4f (tolerance=%.2f)",
				bits, maxDeviation, tolerance)

			if maxDeviation > tolerance {
				t.Errorf("max norm deviation = %.4f, want <= %.4f",
					maxDeviation, tolerance)
			}
		})
	}
}

// TestTurboQuantize_vs_ScalarQuantizer compares TurboQuant_mse against
// UniformQuantizer on the same unit-norm vectors at 4-bit quantization.
//
// TurboQuant_mse should achieve lower MSE because:
//  1. The random rotation decorrelates coordinates, concentrating the distribution
//     into a narrow range where Lloyd-Max centroids are optimally placed.
//  2. UniformQuantizer spreads its levels uniformly over the full range [-1, 1],
//     wasting levels on regions of low probability.
//
// We also test with 8-bit uniform as a reference for higher-quality quantization.
func TestTurboQuantize_vs_ScalarQuantizer(t *testing.T) {
	const dim = 128
	const numVecs = 200

	rng := rand.New(rand.NewSource(42))
	vecs := generateUnitVectors(rng, numVecs, dim)

	// TurboQuant at 4-bit.
	tq, err := NewTurboQuantizer(dim, 4, 42)
	if err != nil {
		t.Fatal(err)
	}

	turboRec := make([][]float64, numVecs)
	for i, v := range vecs {
		cv, err := tq.Quantize(v)
		if err != nil {
			t.Fatalf("TurboQuantize vec %d: %v", i, err)
		}
		turboRec[i], err = tq.Dequantize(cv)
		if err != nil {
			t.Fatalf("TurboDequantize vec %d: %v", i, err)
		}
	}
	turboMSE := computeMSE(vecs, turboRec)

	// UniformQuantizer at 4-bit over [-1, 1].
	uq4, err := NewUniformQuantizer(-1.0, 1.0, 4)
	if err != nil {
		t.Fatal(err)
	}

	uniformRec4 := make([][]float64, numVecs)
	for i, v := range vecs {
		cv, err := uq4.Quantize(v)
		if err != nil {
			t.Fatalf("UniformQuantize vec %d: %v", i, err)
		}
		uniformRec4[i], err = uq4.Dequantize(cv)
		if err != nil {
			t.Fatalf("UniformDequantize vec %d: %v", i, err)
		}
	}
	uniformMSE4 := computeMSE(vecs, uniformRec4)

	t.Logf("4-bit comparison (dim=%d, numVecs=%d):", dim, numVecs)
	t.Logf("  TurboQuant_mse MSE:   %.6f", turboMSE)
	t.Logf("  UniformQuantizer MSE: %.6f", uniformMSE4)
	t.Logf("  Improvement ratio:    %.2fx", uniformMSE4/turboMSE)

	// TurboQuant should have lower MSE than uniform at the same bit-width.
	if turboMSE >= uniformMSE4 {
		t.Errorf("TurboQuant MSE (%.6f) should be lower than Uniform MSE (%.6f)",
			turboMSE, uniformMSE4)
	}

	// Also compare against 8-bit uniform as a quality reference.
	uq8, err := NewUniformQuantizer(-1.0, 1.0, 8)
	if err != nil {
		t.Fatal(err)
	}

	uniformRec8 := make([][]float64, numVecs)
	for i, v := range vecs {
		cv, err := uq8.Quantize(v)
		if err != nil {
			t.Fatalf("Uniform8 Quantize vec %d: %v", i, err)
		}
		uniformRec8[i], err = uq8.Dequantize(cv)
		if err != nil {
			t.Fatalf("Uniform8 Dequantize vec %d: %v", i, err)
		}
	}
	uniformMSE8 := computeMSE(vecs, uniformRec8)

	t.Logf("  8-bit Uniform MSE:    %.6f", uniformMSE8)
	t.Logf("  Turbo 4-bit vs Uniform 8-bit ratio: %.2fx", turboMSE/uniformMSE8)
}

// TestTurboQuantize_MultipleDimensions validates quality across common
// embedding dimensions used in practice.
func TestTurboQuantize_MultipleDimensions(t *testing.T) {
	const bits = 4
	const numVecs = 50
	const seed = int64(42)

	dims := []int{128, 256, 512, 768, 1024}
	// MSE tends to decrease with higher dimension (more concentrated Beta dist).
	maxMSEs := map[int]float64{
		128:  0.02,
		256:  0.018,
		512:  0.016,
		768:  0.015,
		1024: 0.014,
	}

	for _, dim := range dims {
		t.Run(fmt.Sprintf("d=%d", dim), func(t *testing.T) {
			tq, err := NewTurboQuantizer(dim, bits, seed)
			if err != nil {
				t.Fatalf("NewTurboQuantizer(%d, %d): %v", dim, bits, err)
			}

			rng := rand.New(rand.NewSource(int64(dim)))
			vecs := generateUnitVectors(rng, numVecs, dim)

			reconstructed := make([][]float64, numVecs)
			for i, v := range vecs {
				cv, err := tq.Quantize(v)
				if err != nil {
					t.Fatalf("Quantize vec %d: %v", i, err)
				}
				reconstructed[i], err = tq.Dequantize(cv)
				if err != nil {
					t.Fatalf("Dequantize vec %d: %v", i, err)
				}
			}

			mse := computeMSE(vecs, reconstructed)
			maxMSE := maxMSEs[dim]
			t.Logf("dim=%d bits=%d numVecs=%d MSE=%.6f (max=%.3f)",
				dim, bits, numVecs, mse, maxMSE)

			if mse > maxMSE {
				t.Errorf("dim=%d MSE=%.6f exceeds max %.6f", dim, mse, maxMSE)
			}
		})
	}
}

// TestTurboQuantize_AllBitWidths_Integration runs a comprehensive round-trip
// check at all supported bit widths with larger vector counts.
func TestTurboQuantize_AllBitWidths_Integration(t *testing.T) {
	const dim = 256
	const numVecs = 500
	const seed = int64(42)

	tests := []struct {
		bits   int
		maxMSE float64
	}{
		{1, 0.45},
		{2, 0.15},
		{3, 0.05},
		{4, 0.015},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("b=%d", tt.bits), func(t *testing.T) {
			tq, err := NewTurboQuantizer(dim, tt.bits, seed)
			if err != nil {
				t.Fatal(err)
			}

			rng := rand.New(rand.NewSource(int64(tt.bits * 100)))
			vecs := generateUnitVectors(rng, numVecs, dim)

			reconstructed := make([][]float64, numVecs)
			for i, v := range vecs {
				cv, err := tq.Quantize(v)
				if err != nil {
					t.Fatalf("Quantize %d: %v", i, err)
				}
				reconstructed[i], err = tq.Dequantize(cv)
				if err != nil {
					t.Fatalf("Dequantize %d: %v", i, err)
				}
			}

			mse := computeMSE(vecs, reconstructed)
			t.Logf("bits=%d dim=%d numVecs=%d MSE=%.6f (max=%.3f)",
				tt.bits, dim, numVecs, mse, tt.maxMSE)

			if mse > tt.maxMSE {
				t.Errorf("MSE=%.6f exceeds max %.6f", mse, tt.maxMSE)
			}
		})
	}
}
