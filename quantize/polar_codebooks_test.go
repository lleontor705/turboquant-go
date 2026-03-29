package quantize

import (
	"fmt"
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// sinPowerPDF integration tests
// ---------------------------------------------------------------------------

// TestSinPowerPDF_Integration verifies that ∫₀^{π/2} sin^n(2ψ) dψ is correct
// for various values of n.
//
// Known values:
//
//	n=0: ∫ sin⁰(2ψ) dψ = π/2
//	n=1: ∫ sin(2ψ) dψ = 1
//	n=3: ∫ sin³(2ψ) dψ = 2/3
//	n=7: ∫ sin⁷(2ψ) dψ = 16/35
func TestSinPowerPDF_Integration(t *testing.T) {
	tests := []struct {
		n        int
		expected float64
	}{
		{1, 1.0},
		{3, 2.0 / 3.0},
		{7, 16.0 / 35.0},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("n=%d", tt.n), func(t *testing.T) {
			pdf := sinPowerPDF(tt.n)

			// Trapezoidal integration over [0, π/2].
			const numPts = 10000
			dx := (math.Pi / 2) / float64(numPts)
			sum := 0.0
			for i := 0; i <= numPts; i++ {
				x := float64(i) * dx
				w := pdf(x)
				if i == 0 || i == numPts {
					w *= 0.5
				}
				sum += w
			}
			sum *= dx

			relError := math.Abs(sum-tt.expected) / tt.expected
			if relError > 0.01 {
				t.Errorf("∫sin^%d(2ψ) dψ = %.8f, want ≈%.8f (relative error = %.4f)",
					tt.n, sum, tt.expected, relError)
			} else {
				t.Logf("n=%d: integral = %.8f, expected = %.8f (rel error = %.6f)",
					tt.n, sum, tt.expected, relError)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// LevelCodebook centroid count tests
// ---------------------------------------------------------------------------

// TestLevelCodebook_CentroidCount verifies the correct number of centroids
// for each level.
func TestLevelCodebook_CentroidCount(t *testing.T) {
	tests := []struct {
		level int
		n     int
		bits  int
		want  int
	}{
		{1, 0, 4, 16}, // Level 1: uniform, 4-bit → 16 centroids
		{2, 1, 2, 4},  // Level 2: sin(2ψ), 2-bit → 4 centroids
		{3, 3, 2, 4},  // Level 3: sin³(2ψ), 2-bit → 4 centroids
		{4, 7, 2, 4},  // Level 4: sin⁷(2ψ), 2-bit → 4 centroids
		{1, 0, 2, 4},  // Level 1: uniform, 2-bit → 4 centroids
		{1, 0, 3, 8},  // Level 1: uniform, 3-bit → 8 centroids
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("level=%d_n=%d_bits=%d", tt.level, tt.n, tt.bits), func(t *testing.T) {
			cb, err := LevelCodebook(tt.level, tt.n, tt.bits)
			if err != nil {
				t.Fatalf("LevelCodebook: %v", err)
			}
			if len(cb) != tt.want {
				t.Errorf("got %d centroids, want %d", len(cb), tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// LevelCodebook centroid range tests
// ---------------------------------------------------------------------------

// TestLevelCodebook_CentroidRange verifies centroids lie within the appropriate
// range for each level.
func TestLevelCodebook_CentroidRange(t *testing.T) {
	t.Run("level1_range", func(t *testing.T) {
		// Level 1: centroids in [0, 2π).
		cb, err := LevelCodebook(1, 0, 4)
		if err != nil {
			t.Fatalf("LevelCodebook: %v", err)
		}
		for i, c := range cb {
			if c < 0 || c >= 2*math.Pi {
				t.Errorf("centroid[%d] = %.6f, outside [0, 2π)", i, c)
			}
		}
		t.Logf("Level 1 centroids: %v", cb)
	})

	t.Run("level2_range", func(t *testing.T) {
		// Level 2+: centroids in [0, π/2].
		cb, err := LevelCodebook(2, 1, 2)
		if err != nil {
			t.Fatalf("LevelCodebook: %v", err)
		}
		tol := 1e-9
		for i, c := range cb {
			if c < -tol || c > math.Pi/2+tol {
				t.Errorf("centroid[%d] = %.6f, outside [0, π/2]", i, c)
			}
		}
		t.Logf("Level 2 centroids: %v", cb)
	})

	t.Run("level3_range", func(t *testing.T) {
		cb, err := LevelCodebook(3, 3, 2)
		if err != nil {
			t.Fatalf("LevelCodebook: %v", err)
		}
		tol := 1e-9
		for i, c := range cb {
			if c < -tol || c > math.Pi/2+tol {
				t.Errorf("centroid[%d] = %.6f, outside [0, π/2]", i, c)
			}
		}
		t.Logf("Level 3 centroids: %v", cb)
	})

	t.Run("level4_range", func(t *testing.T) {
		cb, err := LevelCodebook(4, 7, 2)
		if err != nil {
			t.Fatalf("LevelCodebook: %v", err)
		}
		tol := 1e-9
		for i, c := range cb {
			if c < -tol || c > math.Pi/2+tol {
				t.Errorf("centroid[%d] = %.6f, outside [0, π/2]", i, c)
			}
		}
		t.Logf("Level 4 centroids: %v", cb)
	})
}

// ---------------------------------------------------------------------------
// LevelCodebook sorted tests
// ---------------------------------------------------------------------------

// TestLevelCodebook_Sorted verifies that centroids are in ascending order.
func TestLevelCodebook_Sorted(t *testing.T) {
	tests := []struct {
		level int
		n     int
		bits  int
	}{
		{1, 0, 4},
		{2, 1, 2},
		{3, 3, 2},
		{4, 7, 2},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("level=%d", tt.level), func(t *testing.T) {
			cb, err := LevelCodebook(tt.level, tt.n, tt.bits)
			if err != nil {
				t.Fatalf("LevelCodebook: %v", err)
			}
			for i := 1; i < len(cb); i++ {
				if cb[i] <= cb[i-1] {
					t.Errorf("centroids not sorted: [%d]=%.10f >= [%d]=%.10f",
						i-1, cb[i-1], i, cb[i])
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// PolarSinExponent tests
// ---------------------------------------------------------------------------

// TestPolarSinExponent verifies the sin exponent calculation.
func TestPolarSinExponent(t *testing.T) {
	tests := []struct {
		level int
		want  int
	}{
		{1, 0},
		{2, 1},
		{3, 3},
		{4, 7},
		{5, 15},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("level=%d", tt.level), func(t *testing.T) {
			got := PolarSinExponent(tt.level)
			if got != tt.want {
				t.Errorf("PolarSinExponent(%d) = %d, want %d", tt.level, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// LevelCodebook caching test
// ---------------------------------------------------------------------------

// TestLevelCodebook_Caching verifies that repeated calls return the same
// centroids.
func TestLevelCodebook_Caching(t *testing.T) {
	cb1, err := LevelCodebook(2, 1, 2)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	cb2, err := LevelCodebook(2, 1, 2)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}

	for i := range cb1 {
		if cb1[i] != cb2[i] {
			t.Errorf("cached centroids differ at index %d: %.10e vs %.10e", i, cb1[i], cb2[i])
		}
	}
}

// ---------------------------------------------------------------------------
// LevelCodebook error cases
// ---------------------------------------------------------------------------

// TestLevelCodebook_Errors verifies error handling.
func TestLevelCodebook_Errors(t *testing.T) {
	t.Run("bits_zero", func(t *testing.T) {
		_, err := LevelCodebook(1, 0, 0)
		if err == nil {
			t.Error("expected error for bits=0")
		}
	})

	t.Run("level_zero", func(t *testing.T) {
		_, err := LevelCodebook(0, 0, 2)
		if err == nil {
			t.Error("expected error for level=0")
		}
	})

	t.Run("level2_n_zero", func(t *testing.T) {
		_, err := LevelCodebook(2, 0, 2)
		if err == nil {
			t.Error("expected error for level>=2 with n=0")
		}
	})
}
