// Package sketch provides 1-bit sketching (QJL) for approximate nearest
// neighbor search via Johnson-Lindenstrauss random projection and sign
// quantization.
//
// The core type is BitVector, a packed bit representation of sign-quantized
// vectors. All operations on BitVector values use little-endian bit ordering:
// element i maps to bit (i % 64) of uint64[i/64], where bit 0 is the LSB.
//
// All public types and methods are safe for concurrent use without external
// synchronization.
package sketch

import "errors"

// BitVector is a packed bit representation of sign-quantized vectors.
//
// Bits are packed in little-endian order: element i of the original sign
// vector maps to bit (i % 64) of Bits[i/64], where bit 0 is the least
// significant bit. A positive sign (+1) encodes as bit 1, and a negative
// sign (-1) encodes as bit 0.
//
// For dimensions that are not a multiple of 64, the unused high bits of the
// final uint64 word are set to 0.
//
// BitVector implements encoding.BinaryMarshaler and
// encoding.BinaryUnmarshaler for cross-platform serialization using a
// little-endian wire format with a version byte prefix.
type BitVector struct {
	// Bits holds the packed sign bits in []uint64 words.
	// Length is (Dim + 63) / 64.
	Bits []uint64
	// Dim is the original vector dimension (number of meaningful bits).
	Dim int
	// OutlierIndices holds the indices of top-K outlier projections stored
	// in full precision. nil when outlier handling is disabled (OutlierK=0).
	OutlierIndices []int
	// OutlierValues holds the full-precision projection values at outlier
	// positions, sorted by absolute value descending. nil when outlier
	// handling is disabled (OutlierK=0).
	OutlierValues []float64
}

// Sentinel errors for the sketch package.
// All errors are defined as var values so callers can use errors.Is for
// matching, including through wrapped errors created with fmt.Errorf("%w", err).
var (
	// ErrDimensionMismatch indicates that two BitVector operands have
	// different Dim values, or an input vector length does not match the
	// expected dimension.
	ErrDimensionMismatch = errors.New("sketch: dimension mismatch")

	// ErrInvalidDimension indicates that a dimension parameter is zero,
	// negative, or otherwise invalid for construction.
	ErrInvalidDimension = errors.New("sketch: invalid dimension")

	// ErrInvalidConfiguration indicates that the sketcher configuration
	// parameters are invalid (e.g. OutlierK exceeds sketch dimension).
	ErrInvalidConfiguration = errors.New("sketch: invalid configuration")
)
