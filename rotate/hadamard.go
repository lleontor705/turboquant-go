// Package rotate provides random orthogonal matrix generation and the Fast
// Walsh-Hadamard Transform (FWHT). These are foundational operations for
// random rotations in the PolarQuant pipeline and SRHT-based sketching.
//
// All public types and functions are safe for concurrent use without external
// synchronization.
package rotate

import "errors"

// Sentinel errors for the rotate package.
// All errors are defined as var values so callers can use errors.Is for
// matching, including through wrapped errors created with fmt.Errorf("%w", err).
var (
	// ErrNotPowerOfTwo indicates that the input vector length is not a power
	// of 2, which is required by the Fast Walsh-Hadamard Transform.
	ErrNotPowerOfTwo = errors.New("rotate: dimension must be a power of 2")
)

// IsPowerOfTwo returns true if n is a power of 2 (including n=1).
// Returns false for n <= 0.
func IsPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

// FWHT performs an in-place Fast Walsh-Hadamard Transform on x.
//
// The length of x must be a power of 2. Returns ErrNotPowerOfTwo otherwise,
// and the input slice is NOT modified.
//
// The transform is unnormalized: the result is H*x where H is the Hadamard
// matrix. The caller is responsible for applying 1/sqrt(n) normalization if
// needed.
//
// Complexity: O(n log n).
func FWHT(x []float64) error {
	n := len(x)
	if !IsPowerOfTwo(n) {
		return ErrNotPowerOfTwo
	}

	// Iterative butterfly pattern:
	//   for step = 1; step < n; step *= 2:
	//     for i = 0; i < n; i += 2*step:
	//       for j = 0; j < step; j++:
	//         a = x[i+j]
	//         b = x[i+j+step]
	//         x[i+j]     = a + b
	//         x[i+j+step] = a - b
	for step := 1; step < n; step <<= 1 {
		for i := 0; i < n; i += step << 1 {
			for j := 0; j < step; j++ {
				a := x[i+j]
				b := x[i+j+step]
				x[i+j] = a + b
				x[i+j+step] = a - b
			}
		}
	}

	return nil
}
