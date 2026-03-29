package sketch

import (
	"math/bits"
)

// HammingDistance computes the Hamming distance (number of differing bits)
// between two BitVectors. It panics on dimension mismatch — use this on the
// hot path where inputs are guaranteed to have matching dimensions.
func HammingDistance(a, b BitVector) int {
	if a.Dim != b.Dim {
		panic("sketch: dimension mismatch")
	}
	return hamming(a.Bits, b.Bits)
}

// HammingDistanceSafe computes the Hamming distance between two BitVectors,
// returning an error on dimension mismatch.
func HammingDistanceSafe(a, b BitVector) (int, error) {
	if a.Dim != b.Dim {
		return 0, ErrDimensionMismatch
	}
	return hamming(a.Bits, b.Bits), nil
}

// hamming computes the popcount of the XOR of two packed uint64 slices.
func hamming(a, b []uint64) int {
	dist := 0
	for i := range a {
		dist += bits.OnesCount64(a[i] ^ b[i])
	}
	return dist
}

// EstimateInnerProduct estimates the inner product (cosine similarity) between
// two original vectors from the Hamming distance of their QJL sketches.
//
// The estimation follows the relationship derived from the JL-preserving
// property of 1-bit quantization:
//
//	estimated_IP = 1 - 2 * (hamming / sketchDim)
//
// Returns the estimated similarity in [-1, 1].
// Returns an error if the sketch dimensions differ.
func EstimateInnerProduct(a, b BitVector) (float64, error) {
	if a.Dim != b.Dim {
		return 0, ErrDimensionMismatch
	}
	hamming := hamming(a.Bits, b.Bits)
	return 1.0 - 2.0*float64(hamming)/float64(a.Dim), nil
}
