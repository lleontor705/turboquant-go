// Package bits provides bit packing utilities for converting sign vectors
// (+1/-1) to packed []uint64 and back.
//
// The bit packing convention is little-endian: element i of the sign vector
// maps to bit (i % 64) of uint64[i/64], where bit 0 is the least significant
// bit. Positive sign (+1) encodes as bit 1, negative sign (-1) encodes as
// bit 0.
//
// For dimensions that are not a multiple of 64, the unused high bits of the
// final uint64 word are set to 0.
//
// Example: Pack([+1, -1, +1, +1, -1, -1, +1, -1])
//
//	Element 0: +1 → bit 0 = 1
//	Element 1: -1 → bit 1 = 0
//	Element 2: +1 → bit 2 = 1
//	Element 3: +1 → bit 3 = 1
//	Element 4: -1 → bit 4 = 0
//	Element 5: -1 → bit 5 = 0
//	Element 6: +1 → bit 6 = 1
//	Element 7: -1 → bit 7 = 0
//
// Result: []uint64{0x4D} (binary: 01001101, reading from MSB to LSB)
package bits

import (
	"errors"

	mathbits "math/bits"
)

// Sentinel errors for the bits package.
// All errors are defined as var values so callers can use errors.Is for
// matching, including through wrapped errors created with fmt.Errorf("%w", err).
var (
	// ErrInvalidSignValue indicates that a sign value is not +1 or -1.
	ErrInvalidSignValue = errors.New("bits: sign value must be +1 or -1")

	// ErrInsufficientData indicates that the packed []uint64 does not have
	// enough words for the requested dimension.
	ErrInsufficientData = errors.New("bits: insufficient packed data for dimension")

	// ErrInvalidDimension indicates that a dimension parameter is negative.
	ErrInvalidDimension = errors.New("bits: invalid dimension")
)

// Pack converts a slice of sign values (+1 or -1) into packed uint64 bits.
//
// Convention: little-endian, bit 0 = LSB = first element.
// +1 maps to bit=1, -1 maps to bit=0.
// For non-multiple-of-64 dimensions, high bits are zero-padded.
//
// Returns an empty (non-nil) []uint64 for empty input.
// Returns ErrInvalidSignValue if any element is not +1 or -1.
func Pack(signs []int8) ([]uint64, error) {
	if len(signs) == 0 {
		return []uint64{}, nil
	}

	numWords := (len(signs) + 63) / 64
	packed := make([]uint64, numWords)

	for i, s := range signs {
		switch s {
		case 1:
			packed[i/64] |= 1 << (uint(i) % 64)
		case -1:
			// bit stays 0 (zero value)
		default:
			return nil, ErrInvalidSignValue
		}
	}

	return packed, nil
}

// Unpack converts packed uint64 bits back to sign values (+1 or -1).
//
// n is the original number of elements (may be < 64*len(packed)).
// Only the first n bits are unpacked; trailing bits in the last word are
// ignored.
//
// Returns an empty (non-nil) []int8 for n == 0.
// Returns ErrInsufficientData if len(packed) < ceil(n/64).
// Returns ErrInvalidDimension if n < 0.
func Unpack(packed []uint64, n int) ([]int8, error) {
	if n < 0 {
		return nil, ErrInvalidDimension
	}
	if n == 0 {
		return []int8{}, nil
	}

	required := (n + 63) / 64
	if len(packed) < required {
		return nil, ErrInsufficientData
	}

	signs := make([]int8, n)
	for i := 0; i < n; i++ {
		bit := (packed[i/64] >> (uint(i) % 64)) & 1
		if bit == 1 {
			signs[i] = 1
		} else {
			signs[i] = -1
		}
	}

	return signs, nil
}

// PopCount returns the number of set bits in a []uint64.
// Uses math/bits.OnesCount64 for hardware-accelerated popcount where
// available.
func PopCount(bits []uint64) int {
	count := 0
	for _, w := range bits {
		count += mathbits.OnesCount64(w)
	}
	return count
}
