package quantize

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/lleontor705/turboquant-go/internal/serial"
)

// TestCompressedVector_SerializationRoundTrip verifies that a CompressedVector
// survives a MarshalBinary → UnmarshalBinary round-trip with all fields intact,
// and that the dequantized output matches the original within error bounds.
func TestCompressedVector_SerializationRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		min  float64
		max  float64
		bits int
		vec  []float64
	}{
		{
			name: "8-bit simple",
			min:  0.0,
			max:  10.0,
			bits: 8,
			vec:  []float64{0.0, 5.0, 10.0, 2.5},
		},
		{
			name: "4-bit simple",
			min:  -1.0,
			max:  1.0,
			bits: 4,
			vec:  []float64{0.0, 1.0, -1.0, 0.5},
		},
		{
			name: "4-bit odd dimension",
			min:  0.0,
			max:  1.0,
			bits: 4,
			vec:  []float64{0.0, 1.0, 0.5},
		},
		{
			name: "8-bit negative range",
			min:  -100.0,
			max:  -1.0,
			bits: 8,
			vec:  []float64{-50.0, -1.0, -100.0, -25.0, -75.0},
		},
		{
			name: "8-bit large vector",
			min:  -10.0,
			max:  10.0,
			bits: 8,
			vec:  generateRandomVec(256, -10.0, 10.0, 99),
		},
		{
			name: "4-bit large vector",
			min:  0.0,
			max:  1.0,
			bits: 4,
			vec:  generateRandomVec(127, 0.0, 1.0, 42),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q, err := NewUniformQuantizer(tt.min, tt.max, tt.bits)
			if err != nil {
				t.Fatalf("NewUniformQuantizer: %v", err)
			}

			// Quantize the original vector.
			cv, err := q.Quantize(tt.vec)
			if err != nil {
				t.Fatalf("Quantize: %v", err)
			}

			// Marshal to binary.
			data, err := cv.MarshalBinary()
			if err != nil {
				t.Fatalf("MarshalBinary: %v", err)
			}

			// Unmarshal into a fresh CompressedVector.
			var cv2 CompressedVector
			if err := cv2.UnmarshalBinary(data); err != nil {
				t.Fatalf("UnmarshalBinary: %v", err)
			}

			// Verify all metadata fields match.
			if cv2.Dim != cv.Dim {
				t.Errorf("Dim mismatch: got %d, want %d", cv2.Dim, cv.Dim)
			}
			if cv2.Min != cv.Min {
				t.Errorf("Min mismatch: got %v, want %v", cv2.Min, cv.Min)
			}
			if cv2.Max != cv.Max {
				t.Errorf("Max mismatch: got %v, want %v", cv2.Max, cv.Max)
			}
			if cv2.BitsPer != cv.BitsPer {
				t.Errorf("BitsPer mismatch: got %d, want %d", cv2.BitsPer, cv.BitsPer)
			}
			if len(cv2.Data) != len(cv.Data) {
				t.Fatalf("Data length mismatch: got %d, want %d", len(cv2.Data), len(cv.Data))
			}
			for i := range cv.Data {
				if cv2.Data[i] != cv.Data[i] {
					t.Errorf("Data[%d] mismatch: got 0x%02x, want 0x%02x", i, cv2.Data[i], cv2.Data[i])
				}
			}

			// Dequantize the deserialized vector and verify against original.
			recovered, err := q.Dequantize(cv2)
			if err != nil {
				t.Fatalf("Dequantize after round-trip: %v", err)
			}

			for i, orig := range tt.vec {
				absErr := math.Abs(recovered[i] - orig)
				if absErr > q.MaxError()+1e-12 {
					t.Errorf("element %d: abs error %.6e exceeds MaxError %.6e",
						i, absErr, q.MaxError())
				}
			}
		})
	}
}

// TestCompressedVector_SerializationCorrupted verifies that UnmarshalBinary
// returns errors for corrupted or truncated data.
func TestCompressedVector_SerializationCorrupted(t *testing.T) {
	// Build a valid serialized blob first.
	q, err := NewUniformQuantizer(0.0, 10.0, 8)
	if err != nil {
		t.Fatalf("NewUniformQuantizer: %v", err)
	}
	cv, err := q.Quantize([]float64{1.0, 2.0, 3.0})
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	validData, err := cv.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	tests := []struct {
		name string
		data []byte
	}{
		{"empty data", []byte{}},
		{"only version byte", validData[:1]},
		{"truncated after version", validData[:2]},
		{"truncated after dim", validData[:5]},
		{"truncated after min", validData[:13]},
		{"truncated after max", validData[:21]},
		{"truncated after bitsPer", validData[:25]},
		{"truncated data length prefix", validData[:28]},
		{"truncated data payload", validData[:len(validData)-1]},
		{"wrong version", append([]byte{0xFF}, validData[1:]...)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cv2 CompressedVector
			err := cv2.UnmarshalBinary(tt.data)
			if err == nil {
				t.Error("expected error for corrupted data, got nil")
			}
		})
	}

	// Verify that the wrong-version case specifically returns
	// serial.ErrUnsupportedVersion wrapped in our error.
	t.Run("wrong version is ErrUnsupportedVersion", func(t *testing.T) {
		bad := append([]byte{0xFF}, validData[1:]...)
		var cv2 CompressedVector
		err := cv2.UnmarshalBinary(bad)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !errors.Is(err, serial.ErrUnsupportedVersion) {
			t.Errorf("error = %v, want wrapping serial.ErrUnsupportedVersion", err)
		}
	})
}

// TestCompressedVector_MarshalBinaryEmptyData verifies serialization of a
// CompressedVector with empty Data (edge case).
func TestCompressedVector_MarshalBinaryEmptyData(t *testing.T) {
	cv := CompressedVector{
		Data:    []byte{},
		Dim:     0,
		Min:     -1.0,
		Max:     1.0,
		BitsPer: 8,
	}

	data, err := cv.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	var cv2 CompressedVector
	if err := cv2.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary: %v", err)
	}

	if cv2.Dim != cv.Dim {
		t.Errorf("Dim: got %d, want %d", cv2.Dim, cv.Dim)
	}
	if cv2.Min != cv.Min {
		t.Errorf("Min: got %v, want %v", cv2.Min, cv.Min)
	}
	if cv2.Max != cv.Max {
		t.Errorf("Max: got %v, want %v", cv2.Max, cv.Max)
	}
	if cv2.BitsPer != cv.BitsPer {
		t.Errorf("BitsPer: got %d, want %d", cv2.BitsPer, cv.BitsPer)
	}
	if len(cv2.Data) != 0 {
		t.Errorf("Data length: got %d, want 0", len(cv2.Data))
	}
}

// TestCompressedVector_MarshalBinaryIdempotent verifies that marshaling twice
// produces identical bytes.
func TestCompressedVector_MarshalBinaryIdempotent(t *testing.T) {
	q, _ := NewUniformQuantizer(-5.0, 5.0, 8)
	cv, _ := q.Quantize([]float64{-3.0, 0.0, 4.5, -1.2, 2.8})

	data1, err := cv.MarshalBinary()
	if err != nil {
		t.Fatalf("first MarshalBinary: %v", err)
	}
	data2, err := cv.MarshalBinary()
	if err != nil {
		t.Fatalf("second MarshalBinary: %v", err)
	}

	if len(data1) != len(data2) {
		t.Fatalf("length mismatch: %d vs %d", len(data1), len(data2))
	}
	for i := range data1 {
		if data1[i] != data2[i] {
			t.Errorf("byte %d differs: 0x%02x vs 0x%02x", i, data1[i], data2[i])
		}
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func generateRandomVec(dim int, min, max float64, seed int64) []float64 {
	rng := rand.New(rand.NewSource(seed))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.Float64()*(max-min) + min
	}
	return vec
}
