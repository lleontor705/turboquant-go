package rotate

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// frobeniusNormDiffIdentity computes ‖Q^T Q − I‖_F, measuring how far Q is
// from being a perfectly orthogonal matrix.
func frobeniusNormDiffIdentity(Q *mat.Dense) float64 {
	_, c := Q.Dims()
	// Compute Q^T * Q
	QtQ := mat.NewDense(c, c, nil)
	QtQ.Mul(Q.T(), Q)

	// Subtract identity and compute Frobenius norm of the difference.
	norm := 0.0
	for i := 0; i < c; i++ {
		for j := 0; j < c; j++ {
			d := QtQ.At(i, j)
			if i == j {
				d -= 1.0
			}
			norm += d * d
		}
	}
	return norm
}

// ---------------------------------------------------------------------------
// TestOrthogonalMatrixUnitarity
// ---------------------------------------------------------------------------

// TestOrthogonalMatrixUnitarity verifies that ‖Q^T Q − I‖_F < 1e-10 for
// dimensions 1, 2, 8, 64, and 256.
func TestOrthogonalMatrixUnitarity(t *testing.T) {
	dims := []int{1, 2, 8, 64, 256}

	for _, dim := range dims {
		t.Run("", func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(dim)))
			Q, err := RandomOrthogonal(dim, rng)
			if err != nil {
				t.Fatalf("RandomOrthogonal(%d, rng) returned error: %v", dim, err)
			}

			rows, cols := Q.Dims()
			if rows != dim || cols != dim {
				t.Fatalf("expected %d×%d matrix, got %d×%d", dim, dim, rows, cols)
			}

			// Check orthogonality: ‖Q^T Q − I‖_F < 1e-10
			frobSq := frobeniusNormDiffIdentity(Q)
			if frobSq >= 1e-10 {
				t.Errorf("dim=%d: ‖Q^T Q − I‖_F² = %g (want < 1e-10)", dim, frobSq)
			}

			// Special case: dim=1 should contain ±1.0
			if dim == 1 {
				v := Q.At(0, 0)
				if v != 1.0 && v != -1.0 {
					t.Errorf("dim=1: expected ±1.0, got %g", v)
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestOrthogonalDeterminism
// ---------------------------------------------------------------------------

// TestOrthogonalDeterminism verifies that the same seed produces identical
// matrices (element-by-element comparison).
func TestOrthogonalDeterminism(t *testing.T) {
	dims := []int{1, 2, 8, 64, 256}
	seed := int64(12345)

	for _, dim := range dims {
		t.Run("", func(t *testing.T) {
			rng1 := rand.New(rand.NewSource(seed))
			Q1, err := RandomOrthogonal(dim, rng1)
			if err != nil {
				t.Fatalf("first call error: %v", err)
			}

			rng2 := rand.New(rand.NewSource(seed))
			Q2, err := RandomOrthogonal(dim, rng2)
			if err != nil {
				t.Fatalf("second call error: %v", err)
			}

			// Compare element-by-element: identical seed must produce
			// bit-for-bit identical floating point values.
			for i := 0; i < dim; i++ {
				for j := 0; j < dim; j++ {
					if Q1.At(i, j) != Q2.At(i, j) {
						t.Errorf("dim=%d: matrices differ at (%d,%d): %g vs %g",
							dim, i, j, Q1.At(i, j), Q2.At(i, j))
						return
					}
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestOrthogonal_InvalidDim
// ---------------------------------------------------------------------------

// TestOrthogonal_InvalidDim verifies that dim=0 and dim<0 return an error.
func TestOrthogonal_InvalidDim(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	tests := []struct {
		name string
		dim  int
	}{
		{"dim=0", 0},
		{"dim=-1", -1},
		{"dim=-100", -100},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Q, err := RandomOrthogonal(tt.dim, rng)
			if Q != nil {
				t.Errorf("expected nil matrix, got %v", Q)
			}
			if err == nil {
				t.Error("expected error, got nil")
			}
			if err != ErrInvalidDimension {
				t.Errorf("expected ErrInvalidDimension, got %v", err)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestOrthogonal_NilRng
// ---------------------------------------------------------------------------

// TestOrthogonal_NilRng verifies that a nil rng returns an error.
func TestOrthogonal_NilRng(t *testing.T) {
	Q, err := RandomOrthogonal(8, nil)
	if Q != nil {
		t.Errorf("expected nil matrix, got %v", Q)
	}
	if err == nil {
		t.Error("expected error for nil rng, got nil")
	}
	if err != ErrNilRNG {
		t.Errorf("expected ErrNilRNG, got %v", err)
	}
}

// TestOrthogonal_Dim1_BothSigns verifies that RandomOrthogonal(1, rng) can
// produce both +1.0 and -1.0 by trying multiple seeds. This ensures the
// negative branch (rng.Float64() < 0.5) is exercised.
func TestOrthogonal_Dim1_BothSigns(t *testing.T) {
	gotPositive := false
	gotNegative := false
	for seed := int64(0); seed < 100; seed++ {
		rng := rand.New(rand.NewSource(seed))
		Q, err := RandomOrthogonal(1, rng)
		if err != nil {
			t.Fatalf("seed=%d: unexpected error: %v", seed, err)
		}
		v := Q.At(0, 0)
		if v == 1.0 {
			gotPositive = true
		} else if v == -1.0 {
			gotNegative = true
		} else {
			t.Fatalf("seed=%d: expected ±1.0, got %g", seed, v)
		}
		if gotPositive && gotNegative {
			return
		}
	}
	t.Fatal("failed to observe both +1.0 and -1.0 across 100 seeds")
}
