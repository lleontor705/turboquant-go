package quantize

import (
	"fmt"
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// BetaPDF tests
// ---------------------------------------------------------------------------

// TestBetaPDF_Normalization verifies that the Beta PDF integrates to 1.0
// over its support [-1, 1].
func TestBetaPDF_Normalization(t *testing.T) {
	dims := []int{64, 256, 768}
	for _, d := range dims {
		t.Run(fmt.Sprintf("d=%d", d), func(t *testing.T) {
			pdf, err := BetaPDF(d)
			if err != nil {
				t.Fatalf("BetaPDF(%d): %v", d, err)
			}

			// Trapezoidal integration over [-1, 1].
			const n = 10000
			dx := 2.0 / float64(n)
			sum := 0.0
			for i := 0; i <= n; i++ {
				x := -1.0 + float64(i)*dx
				w := pdf(x)
				if i == 0 || i == n {
					w *= 0.5
				}
				sum += w
			}
			sum *= dx

			if math.Abs(sum-1.0) > 0.01 {
				t.Errorf("∫BetaPDF(d=%d) = %.6f, want ≈1.0 (error = %.6f)",
					d, sum, math.Abs(sum-1.0))
			} else {
				t.Logf("d=%d: integral = %.8f", d, sum)
			}
		})
	}
}

// TestBetaPDF_Symmetry verifies that BetaPDF(d)(x) == BetaPDF(d)(-x).
func TestBetaPDF_Symmetry(t *testing.T) {
	dims := []int{64, 256, 768, 1024}
	testX := []float64{0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}

	for _, d := range dims {
		d := d
		pdf, err := BetaPDF(d)
		if err != nil {
			t.Fatalf("BetaPDF(%d): %v", d, err)
		}
		for _, x := range testX {
			fx := pdf(x)
			fnegX := pdf(-x)
			if math.Abs(fx-fnegX) > 1e-12 {
				t.Errorf("BetaPDF(d=%d)(%.2f) = %.10e != BetaPDF(d=%d)(%.2f) = %.10e",
					d, x, fx, d, -x, fnegX)
			}
		}
	}
}

// TestBetaPDF_SmallDim verifies BetaPDF works for small dimensions.
func TestBetaPDF_SmallDim(t *testing.T) {
	t.Run("d=3_uniform", func(t *testing.T) {
		// For d=3, exponent = 0, so f(x) = constant = 0.5 for x ∈ (-1, 1).
		pdf, err := BetaPDF(3)
		if err != nil {
			t.Fatalf("BetaPDF(3): %v", err)
		}
		mid := pdf(0.0)
		if math.Abs(mid-0.5) > 1e-10 {
			t.Errorf("BetaPDF(3)(0) = %.10f, want 0.5", mid)
		}
	})

	t.Run("d=1_error", func(t *testing.T) {
		_, err := BetaPDF(1)
		if err == nil {
			t.Error("expected error for BetaPDF(1)")
		}
	})
}

// ---------------------------------------------------------------------------
// TurboCodebook tests
// ---------------------------------------------------------------------------

// TestTurboCodebook_ValidBits verifies that b=1,2,3,4 work and b=5 returns error.
func TestTurboCodebook_ValidBits(t *testing.T) {
	const d = 64

	for _, b := range []int{1, 2, 3, 4} {
		t.Run(fmt.Sprintf("b=%d", b), func(t *testing.T) {
			cb, err := TurboCodebook(d, b)
			if err != nil {
				t.Fatalf("TurboCodebook(d=%d, b=%d): %v", d, b, err)
			}
			expectedLen := 1 << b
			if len(cb) != expectedLen {
				t.Errorf("len(codebook) = %d, want %d", len(cb), expectedLen)
			}
		})
	}

	t.Run("b=5_error", func(t *testing.T) {
		_, err := TurboCodebook(d, 5)
		if err == nil {
			t.Error("expected error for b=5")
		}
	})

	t.Run("b=0_error", func(t *testing.T) {
		_, err := TurboCodebook(d, 0)
		if err == nil {
			t.Error("expected error for b=0")
		}
	})
}

// TestTurboCodebook_Dimensions verifies that common dimensions work and d=0 returns error.
func TestTurboCodebook_Dimensions(t *testing.T) {
	for _, d := range []int{64, 768, 1024} {
		t.Run(fmt.Sprintf("d=%d", d), func(t *testing.T) {
			cb, err := TurboCodebook(d, 2)
			if err != nil {
				t.Fatalf("TurboCodebook(d=%d, b=2): %v", d, err)
			}
			if len(cb) != 4 {
				t.Errorf("len(codebook) = %d, want 4", len(cb))
			}
		})
	}

	t.Run("d=0_error", func(t *testing.T) {
		_, err := TurboCodebook(0, 2)
		if err == nil {
			t.Error("expected error for d=0")
		}
	})

	t.Run("d=1_error", func(t *testing.T) {
		_, err := TurboCodebook(1, 2)
		if err == nil {
			t.Error("expected error for d=1")
		}
	})
}

// TestTurboCodebook_Values verifies codebook invariants: centroids in [-1, 1],
// sorted, and correct count.
func TestTurboCodebook_Values(t *testing.T) {
	dims := []int{64, 256, 768, 1024}
	bits := []int{1, 2, 3, 4}

	for _, d := range dims {
		for _, b := range bits {
			name := fmt.Sprintf("d=%d_b=%d", d, b)
			t.Run(name, func(t *testing.T) {
				cb, err := TurboCodebook(d, b)
				if err != nil {
					t.Fatalf("TurboCodebook: %v", err)
				}

				expectedLen := 1 << b
				if len(cb) != expectedLen {
					t.Fatalf("len = %d, want %d", len(cb), expectedLen)
				}

				// All centroids must be in [-1, 1].
				for i, c := range cb {
					if c < -1.0-1e-9 || c > 1.0+1e-9 {
						t.Errorf("centroids[%d] = %.10f, outside [-1, 1]", i, c)
					}
				}

				// Centroids must be sorted in ascending order.
				for i := 1; i < len(cb); i++ {
					if cb[i] <= cb[i-1] {
						t.Errorf("centroids not sorted: [%d]=%.10f >= [%d]=%.10f",
							i-1, cb[i-1], i, cb[i])
					}
				}

				t.Logf("centroids: %v", cb)
			})
		}
	}
}

// TestTurboCodebook_Caching verifies that calling TurboCodebook twice returns
// the same slice (cached).
func TestTurboCodebook_Caching(t *testing.T) {
	cb1, err := TurboCodebook(128, 3)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	cb2, err := TurboCodebook(128, 3)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}

	// Should return the exact same slice (cached).
	if len(cb1) != len(cb2) {
		t.Fatalf("lengths differ: %d vs %d", len(cb1), len(cb2))
	}
	for i := range cb1 {
		if cb1[i] != cb2[i] {
			t.Errorf("cb1[%d] = %.10e != cb2[%d] = %.10e", i, cb1[i], i, cb2[i])
		}
	}
}

// TestTurboCodebook_Symmetry verifies that codebook centroids are symmetric
// around zero (which they should be for a symmetric distribution).
func TestTurboCodebook_Symmetry(t *testing.T) {
	cb, err := TurboCodebook(256, 3)
	if err != nil {
		t.Fatalf("TurboCodebook: %v", err)
	}

	n := len(cb)
	for i := 0; i < n; i++ {
		j := n - 1 - i
		if math.Abs(cb[i]+cb[j]) > 1e-3 {
			t.Errorf("symmetry violated: cb[%d]=%.8f, cb[%d]=%.8f (sum=%.2e)",
				i, cb[i], j, cb[j], cb[i]+cb[j])
		}
	}
}
