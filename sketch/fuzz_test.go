package sketch

import (
	"encoding/binary"
	"math"
	"testing"
)

// FuzzQJLSketch fuzzes the QJLSketcher.Sketch method. It checks:
//  1. No panic on any input (including NaN, ±Inf, zero vectors).
//  2. The returned BitVector has Dim equal to the configured SketchDim.
//  3. The returned BitVector has a valid Bits slice of the expected length.
//
// The fuzzer provides raw bytes that are interpreted as float64 values
// (8 bytes each, little-endian).
func FuzzQJLSketch(f *testing.F) {
	const dim = 16
	const sketchDim = 8

	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:       dim,
		SketchDim: sketchDim,
		Seed:      42,
	})
	if err != nil {
		f.Fatalf("NewQJLSketcher: %v", err)
	}

	// Helper: encode float64s as []byte for seed corpus.
	encodeFloats := func(vals ...float64) []byte {
		buf := make([]byte, len(vals)*8)
		for i, v := range vals {
			binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
		}
		return buf
	}

	// Seed corpus: various patterns (must be exactly dim=16 floats = 128 bytes).
	pad := func(vals ...float64) []byte {
		if len(vals) > dim {
			vals = vals[:dim]
		}
		for len(vals) < dim {
			vals = append(vals, 0.0)
		}
		return encodeFloats(vals...)
	}

	seeds := [][]byte{
		// All zeros
		pad(),
		// Normal values
		pad(1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16),
		// All positive
		pad(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
		// All negative
		pad(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16),
		// Huge values
		pad(1e300, -1e300, 1e-300, -1e-300),
		// ±Inf
		pad(math.Inf(1), math.Inf(-1)),
		// NaN
		pad(math.NaN()),
		// Tiny values near zero
		pad(1e-300, -1e-300, 1e-310, -1e-310),
	}

	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		// Must have exactly dim*8 bytes for a valid vector.
		if len(data) != dim*8 {
			return
		}

		vec := make([]float64, dim)
		for i := range vec {
			bits := binary.LittleEndian.Uint64(data[i*8:])
			vec[i] = math.Float64frombits(bits)
		}

		bv, err := sketcher.Sketch(vec)
		if err != nil {
			// Dimension mismatch would return an error — skip silently.
			t.Skip()
		}

		// Invariant: BitVector.Dim must equal sketchDim.
		if bv.Dim != sketchDim {
			t.Errorf("BitVector.Dim = %d, want %d", bv.Dim, sketchDim)
		}

		// Invariant: Bits length must be ceil(sketchDim/64).
		expectedWords := (sketchDim + 63) / 64
		if len(bv.Bits) != expectedWords {
			t.Errorf("len(Bits) = %d, want %d", len(bv.Bits), expectedWords)
		}

		// Invariant: trailing bits beyond Dim must be zero.
		if bv.Dim%64 != 0 && len(bv.Bits) > 0 {
			lastWord := bv.Bits[len(bv.Bits)-1]
			// Bits above bv.Dim should be 0. Use variable shift to avoid
			// compile-time constant overflow.
			trailingBits := uint(bv.Dim % 64)
			mask := ^uint64(0) << trailingBits
			if lastWord&mask != 0 {
				t.Errorf("trailing bits non-zero: last word = 0x%X, mask = 0x%X", lastWord, mask)
			}
		}
	})
}
