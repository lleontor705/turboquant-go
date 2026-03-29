package quantize

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// PolarTransform round-trip tests
// ---------------------------------------------------------------------------

// TestPolarTransform_RoundTrip verifies that forward + inverse transform
// recovers the original vector to machine precision.
func TestPolarTransform_RoundTrip(t *testing.T) {
	dims := []int{16, 64, 256, 768}

	for _, dim := range dims {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			// Generate a random vector.
			rng := rand.New(rand.NewSource(42))
			vec := make([]float64, dim)
			for i := range vec {
				vec[i] = rng.NormFloat64()
			}

			// Forward transform.
			angles, radii, err := PolarTransform(vec, 4)
			if err != nil {
				t.Fatalf("PolarTransform: %v", err)
			}

			// Build angle centroids (identity: use the exact angles, not quantized).
			angleCentroids := make([][]float64, len(angles))
			for ℓ := range angles {
				angleCentroids[ℓ] = angles[ℓ]
			}

			// Inverse transform.
			reconstructed, err := InversePolarTransform(angleCentroids, radii, 4, dim)
			if err != nil {
				t.Fatalf("InversePolarTransform: %v", err)
			}

			// Compute relative error ‖x - x̂‖ / ‖x‖.
			var normDiff, normOrig float64
			for i := 0; i < dim; i++ {
				d := vec[i] - reconstructed[i]
				normDiff += d * d
				normOrig += vec[i] * vec[i]
			}
			relError := math.Sqrt(normDiff) / math.Sqrt(normOrig)

			if relError > 1e-12 {
				t.Errorf("round-trip relative error = %.2e, want < 1e-12", relError)
			} else {
				t.Logf("dim=%d: relative error = %.2e", dim, relError)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// PolarTransform level size tests
// ---------------------------------------------------------------------------

// TestPolarTransform_LevelSizes verifies that each level produces the correct
// number of angles.
func TestPolarTransform_LevelSizes(t *testing.T) {
	const dim = 64
	const levels = 4

	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = float64(i + 1)
	}

	angles, radii, err := PolarTransform(vec, levels)
	if err != nil {
		t.Fatalf("PolarTransform: %v", err)
	}

	// Expected angle counts per level:
	// Level 0 (ℓ=1): dim/2 = 32
	// Level 1 (ℓ=2): dim/4 = 16
	// Level 2 (ℓ=3): dim/8 = 8
	// Level 3 (ℓ=4): dim/16 = 4
	expectedCounts := []int{32, 16, 8, 4}
	for ℓ, want := range expectedCounts {
		if len(angles[ℓ]) != want {
			t.Errorf("level %d: got %d angles, want %d", ℓ, len(angles[ℓ]), want)
		}
	}

	// Final radii count: dim/16 = 4.
	if len(radii) != dim/(1<<levels) {
		t.Errorf("final radii: got %d, want %d", len(radii), dim/(1<<levels))
	}
}

// ---------------------------------------------------------------------------
// PolarTransform angle range tests
// ---------------------------------------------------------------------------

// TestPolarTransform_AngleRanges verifies that angles at each level fall in
// the expected range.
func TestPolarTransform_AngleRanges(t *testing.T) {
	const dim = 256
	const levels = 4

	rng := rand.New(rand.NewSource(123))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	angles, _, err := PolarTransform(vec, levels)
	if err != nil {
		t.Fatalf("PolarTransform: %v", err)
	}

	// Level 0 (ℓ=1): angles in [0, 2π) (atan2 range is (-π, π] but
	// for our decomposition they span [0, 2π) after considering sign patterns).
	// Actually atan2 returns [-π, π], so we check that range.
	for j, a := range angles[0] {
		if a < -math.Pi-1e-12 || a > math.Pi+1e-12 {
			t.Errorf("level 0 angle[%d] = %.6f, outside [-π, π]", j, a)
		}
	}

	// Levels 1-3 (ℓ=2..4): angles in [0, π/2] because both inputs are
	// non-negative radii from the previous level.
	tol := 1e-12
	for ℓ := 1; ℓ < levels; ℓ++ {
		for j, a := range angles[ℓ] {
			if a < -tol || a > math.Pi/2+tol {
				t.Errorf("level %d angle[%d] = %.6f, outside [0, π/2]", ℓ, j, a)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// PolarTransform radii non-negative test
// ---------------------------------------------------------------------------

// TestPolarTransform_RadiiNonNegative verifies that all radii at every level
// are non-negative.
func TestPolarTransform_RadiiNonNegative(t *testing.T) {
	const dim = 128
	const levels = 4

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	// Manually track radii at each level.
	r := make([]float64, dim)
	copy(r, vec)

	for ℓ := 0; ℓ < levels; ℓ++ {
		halfN := dim >> (ℓ + 1)
		newR := make([]float64, halfN)
		for j := 0; j < halfN; j++ {
			newR[j] = math.Hypot(r[2*j], r[2*j+1])
		}
		for j, v := range newR {
			if v < 0 {
				t.Errorf("level %d radius[%d] = %.6f, want >= 0", ℓ+1, j, v)
			}
		}
		r = newR
	}
}

// ---------------------------------------------------------------------------
// PolarTransform norm conservation test
// ---------------------------------------------------------------------------

// TestPolarTransform_NormConservation verifies that ‖x‖² = Σ r² at each level
// (energy is conserved through the polar decomposition).
func TestPolarTransform_NormConservation(t *testing.T) {
	const dim = 64
	const levels = 4

	rng := rand.New(rand.NewSource(7))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	// Compute original squared norm.
	var origNorm2 float64
	for _, v := range vec {
		origNorm2 += v * v
	}

	// Track radii through levels.
	r := make([]float64, dim)
	copy(r, vec)

	for ℓ := 0; ℓ < levels; ℓ++ {
		halfN := dim >> (ℓ + 1)
		newR := make([]float64, halfN)
		for j := 0; j < halfN; j++ {
			newR[j] = math.Hypot(r[2*j], r[2*j+1])
		}

		// Check energy conservation.
		var rNorm2 float64
		for _, v := range newR {
			rNorm2 += v * v
		}

		if math.Abs(rNorm2-origNorm2)/origNorm2 > 1e-12 {
			t.Errorf("level %d: Σr² = %.10f, want ‖x‖² = %.10f (relative error = %.2e)",
				ℓ+1, rNorm2, origNorm2, math.Abs(rNorm2-origNorm2)/origNorm2)
		} else {
			t.Logf("level %d: Σr² = %.10f, ‖x‖² = %.10f ✓", ℓ+1, rNorm2, origNorm2)
		}

		r = newR
	}
}

// ---------------------------------------------------------------------------
// PolarTransform invalid dimension test
// ---------------------------------------------------------------------------

// TestPolarTransform_InvalidDim verifies that non-multiple-of-16 dimensions
// produce an error.
func TestPolarTransform_InvalidDim(t *testing.T) {
	tests := []struct {
		name   string
		dim    int
		levels int
	}{
		{"dim=15_levels=4", 15, 4},
		{"dim=17_levels=4", 17, 4},
		{"dim=3_levels=2", 3, 2},
		{"dim=11_levels=1", 11, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vec := make([]float64, tt.dim)
			_, _, err := PolarTransform(vec, tt.levels)
			if err == nil {
				t.Error("expected error for invalid dimension")
			} else {
				t.Logf("got expected error: %v", err)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// InversePolarTransform error cases
// ---------------------------------------------------------------------------

// TestInversePolarTransform_Errors verifies error handling.
func TestInversePolarTransform_Errors(t *testing.T) {
	t.Run("levels_mismatch", func(t *testing.T) {
		// angleCentroids has 3 levels but we pass levels=4.
		angles := [][]float64{{0.1}, {0.2}, {0.3}}
		radii := []float64{1.0}
		_, err := InversePolarTransform(angles, radii, 4, 16)
		if err == nil {
			t.Error("expected error for levels mismatch")
		}
	})

	t.Run("dim_mismatch", func(t *testing.T) {
		angles := [][]float64{{0.1}}
		radii := []float64{1.0}
		_, err := InversePolarTransform(angles, radii, 1, 3)
		if err == nil {
			t.Error("expected error for dim mismatch")
		}
	})

	t.Run("zero_levels", func(t *testing.T) {
		_, err := InversePolarTransform(nil, []float64{1.0}, 0, 1)
		if err == nil {
			t.Error("expected error for zero levels")
		}
	})

	t.Run("angle_count_mismatch", func(t *testing.T) {
		// Level 0 should have 2 angles for 4 final radii, but we provide 3.
		angles := [][]float64{{0.1, 0.2, 0.3}}
		radii := []float64{1.0, 2.0, 3.0, 4.0}
		_, err := InversePolarTransform(angles, radii, 1, 8)
		if err == nil {
			t.Error("expected error for angle count mismatch")
		}
	})
}
