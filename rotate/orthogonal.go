// Package rotate provides random orthogonal matrix generation and the Fast
// Walsh-Hadamard Transform (FWHT). These are the foundation for random
// rotations in the PolarQuant pipeline and SRHT-based sketching.
//
// All public types and methods are safe for concurrent use without external
// synchronization.
package rotate

import (
	"errors"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Sentinel errors for the rotate package.
// All errors are defined as var values so callers can use errors.Is for
// matching, including through wrapped errors created with fmt.Errorf("%w", err).
var (
	// ErrInvalidDimension indicates that a dimension parameter is zero,
	// negative, or otherwise invalid for construction.
	ErrInvalidDimension = errors.New("rotate: invalid dimension")

	// ErrNilRNG indicates that a required *rand.Rand argument was nil.
	ErrNilRNG = errors.New("rotate: nil random source")
)

// RandomOrthogonal generates a dim×dim random orthogonal matrix via QR
// decomposition of a random Gaussian matrix. The matrix Q from the QR
// decomposition satisfies ‖Q^T Q − I‖_F < 1e-10.
//
// The generation is deterministic for a given seed: constructing a
// *rand.Rand from the same seed will always produce the same matrix.
//
// Returns ErrInvalidDimension if dim ≤ 0.
// Returns ErrNilRNG if rng is nil.
func RandomOrthogonal(dim int, rng *rand.Rand) (*mat.Dense, error) {
	if dim <= 0 {
		return nil, ErrInvalidDimension
	}
	if rng == nil {
		return nil, ErrNilRNG
	}

	// Special case: dim=1 produces a 1×1 matrix containing either +1.0 or -1.0.
	if dim == 1 {
		v := 1.0
		if rng.Float64() < 0.5 {
			v = -1.0
		}
		return mat.NewDense(1, 1, []float64{v}), nil
	}

	// Step 1: Generate a dim×dim matrix with entries from N(0,1).
	data := make([]float64, dim*dim)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	A := mat.NewDense(dim, dim, data)

	// Step 2: Compute QR decomposition.
	var qr mat.QR
	qr.Factorize(A)

	// Step 3: Extract Q (the orthogonal factor).
	var Q mat.Dense
	qr.QTo(&Q)

	return &Q, nil
}
