package quantize

import (
	"encoding/binary"
	"math"
	"testing"
)

// FuzzScalarQuantize fuzzes the Quantize→Dequantize round-trip for the
// UniformQuantizer. It checks two invariants:
//  1. No panic on any input (including NaN, ±Inf, extreme values).
//  2. For finite in-range values, the round-trip error is within MaxError.
//
// The fuzzer provides raw bytes that are interpreted as float64 values
// (8 bytes each, little-endian) plus a bit-width selector.
func FuzzScalarQuantize(f *testing.F) {
	// Helper: encode float64s as []byte for seed corpus.
	encodeFloats := func(vals ...float64) []byte {
		buf := make([]byte, len(vals)*8)
		for i, v := range vals {
			binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
		}
		return buf
	}

	// Seed corpus: cover a range of interesting patterns.
	seeds := []struct {
		bits int
		data []byte
	}{
		// Normal values
		{8, encodeFloats(0, 1, -1, 0.5, -0.5)},
		{4, encodeFloats(0, 1, -1, 0.5, -0.5)},
		// All zeros
		{8, encodeFloats(0, 0, 0, 0, 0)},
		{4, encodeFloats(0)},
		// NaN values (should return ErrNaNInput, not panic)
		{8, encodeFloats(math.NaN())},
		{4, encodeFloats(1, math.NaN(), -1)},
		// ±Inf (should be clamped)
		{8, encodeFloats(math.Inf(1), math.Inf(-1))},
		{4, encodeFloats(math.Inf(1))},
		// Huge values
		{8, encodeFloats(1e300, -1e300, 1e-300, -1e-300)},
		{4, encodeFloats(1e300)},
		// Edge of range
		{8, encodeFloats(-10, -10, 10, 10)},
		{4, encodeFloats(-1, -1, 1, 1)},
		// Single element
		{8, encodeFloats(3.14)},
		{4, encodeFloats(-0.5)},
		// Mixed special values
		{8, encodeFloats(0, math.Inf(1), -5, math.NaN(), 5)},
	}

	for _, s := range seeds {
		f.Add(s.bits, s.data)
	}

	f.Fuzz(func(t *testing.T, bits int, data []byte) {
		// Only valid bit widths are supported.
		if bits != 4 && bits != 8 {
			return
		}

		var minVal, maxVal float64
		if bits == 8 {
			minVal, maxVal = -10.0, 10.0
		} else {
			minVal, maxVal = -1.0, 1.0
		}

		q, err := NewUniformQuantizer(minVal, maxVal, bits)
		if err != nil {
			t.Fatalf("NewUniformQuantizer: %v", err)
		}

		// Decode []byte → []float64 (8 bytes per float64).
		if len(data)%8 != 0 {
			return // skip non-aligned data
		}
		if len(data) == 0 {
			return // skip empty
		}
		vec := make([]float64, len(data)/8)
		for i := range vec {
			bits := binary.LittleEndian.Uint64(data[i*8:])
			vec[i] = math.Float64frombits(bits)
		}

		cv, err := q.Quantize(vec)
		if err != nil {
			// NaN input is expected to return ErrNaNInput — that's fine.
			if err == ErrNaNInput {
				return
			}
			t.Fatalf("Quantize unexpected error: %v", err)
		}

		rt, err := q.Dequantize(cv)
		if err != nil {
			t.Fatalf("Dequantize error: %v", err)
		}

		if len(rt) != len(vec) {
			t.Fatalf("round-trip length: got %d, want %d", len(rt), len(vec))
		}

		// Verify round-trip error for finite in-range values.
		maxErr := q.MaxError()
		for i, orig := range vec {
			// Skip non-finite originals (clamping makes error bound meaningless).
			if math.IsNaN(orig) || math.IsInf(orig, 0) {
				continue
			}
			// Skip out-of-range values (they get clamped, so round-trip error is larger).
			if orig < minVal || orig > maxVal {
				continue
			}
			diff := math.Abs(rt[i] - orig)
			if diff > maxErr+1e-12 {
				t.Errorf("round-trip error at [%d]: |%f - %f| = %f > MaxError %f",
					i, rt[i], orig, diff, maxErr)
			}
		}
	})
}
