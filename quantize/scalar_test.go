package quantize

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

func TestNewUniformQuantizer(t *testing.T) {
	tests := []struct {
		name    string
		min     float64
		max     float64
		bits    int
		wantErr error
	}{
		{"8-bit valid", 0.0, 10.0, 8, nil},
		{"4-bit valid", -5.0, 5.0, 4, nil},
		{"4-bit positive range", 0.0, 1.0, 4, nil},
		{"8-bit negative range", -100.0, -1.0, 8, nil},
		{"inverted range", 10.0, 0.0, 8, ErrInvalidConfig},
		{"equal range", 5.0, 5.0, 8, ErrInvalidConfig},
		{"unsupported bits 3", 0.0, 10.0, 3, ErrInvalidConfig},
		{"unsupported bits 0", 0.0, 10.0, 0, ErrInvalidConfig},
		{"unsupported bits 16", 0.0, 10.0, 16, ErrInvalidConfig},
		{"unsupported bits 1", 0.0, 10.0, 1, ErrInvalidConfig},
		{"unsupported bits 2", 0.0, 10.0, 2, ErrInvalidConfig},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q, err := NewUniformQuantizer(tt.min, tt.max, tt.bits)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Errorf("NewUniformQuantizer() error = %v, want %v", err, tt.wantErr)
				}
				if q != nil {
					t.Error("NewUniformQuantizer() returned non-nil quantizer on error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if q == nil {
				t.Fatal("returned nil quantizer without error")
			}
			if q.Bits() != tt.bits {
				t.Errorf("Bits() = %d, want %d", q.Bits(), tt.bits)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Round-trip fidelity tests
// ---------------------------------------------------------------------------

func TestScalarQuantizeRoundTrip_8bit(t *testing.T) {
	const (
		minVal   = -10.0
		maxVal   = 10.0
		bits     = 8
		nVectors = 10000
		dim      = 128
	)

	q, err := NewUniformQuantizer(minVal, maxVal, bits)
	if err != nil {
		t.Fatalf("NewUniformQuantizer: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	totalAbsErr := 0.0
	totalElements := 0

	for v := 0; v < nVectors; v++ {
		vec := make([]float64, dim)
		for i := range vec {
			vec[i] = rng.Float64()*(maxVal-minVal) + minVal // uniform in [min, max]
		}

		cv, err := q.Quantize(vec)
		if err != nil {
			t.Fatalf("Quantize: %v", err)
		}
		if cv.Dim != dim {
			t.Fatalf("cv.Dim = %d, want %d", cv.Dim, dim)
		}
		if cv.BitsPer != bits {
			t.Fatalf("cv.BitsPer = %d, want %d", cv.BitsPer, bits)
		}

		recovered, err := q.Dequantize(cv)
		if err != nil {
			t.Fatalf("Dequantize: %v", err)
		}

		for i := range vec {
			absErr := math.Abs(recovered[i] - vec[i])
			totalAbsErr += absErr
			totalElements++

			// Every element must satisfy the MaxError bound.
			if absErr > q.MaxError()+1e-12 {
				t.Errorf("vec[%d][%d]: abs error %.6e exceeds MaxError %.6e",
					v, i, absErr, q.MaxError())
			}
		}
	}

	// Mean relative error = mean absolute error / quantization range.
	// For 8-bit this is approximately 0.5/(2^8-1) ≈ 0.2%, well under 1%.
	range_ := maxVal - minVal
	meanRelErr := (totalAbsErr / float64(totalElements)) / range_
	if meanRelErr >= 0.01 {
		t.Errorf("mean relative error = %.6f, want < 0.01", meanRelErr)
	}
	t.Logf("8-bit mean relative error over %d elements: %.6f (%.4f%% of range)",
		totalElements, meanRelErr, meanRelErr*100)
}

func TestScalarQuantizeRoundTrip_4bit(t *testing.T) {
	q, err := NewUniformQuantizer(0.0, 1.0, 4)
	if err != nil {
		t.Fatalf("NewUniformQuantizer: %v", err)
	}

	// Boundary values must be recovered exactly.
	vec := []float64{0.0, 1.0, 0.5}
	cv, err := q.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	if cv.Dim != 3 {
		t.Fatalf("cv.Dim = %d, want 3", cv.Dim)
	}
	// 3 dims with 4-bit → ceil(3/2) = 2 bytes
	if len(cv.Data) != 2 {
		t.Fatalf("len(cv.Data) = %d, want 2", len(cv.Data))
	}

	recovered, err := q.Dequantize(cv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	// 0.0 → code 0 → 0.0 (exact)
	if recovered[0] != 0.0 {
		t.Errorf("boundary 0.0: got %v, want 0.0", recovered[0])
	}
	// 1.0 → code 15 → 0+15*(1/15) = 1.0 (exact)
	if recovered[1] != 1.0 {
		t.Errorf("boundary 1.0: got %v, want 1.0", recovered[1])
	}
	// 0.5 must be within MaxError
	absErr := math.Abs(recovered[2] - 0.5)
	if absErr > q.MaxError()+1e-12 {
		t.Errorf("0.5: abs error %v exceeds MaxError %v", absErr, q.MaxError())
	}
	t.Logf("4-bit boundary: 0.0→%v  1.0→%v  0.5→%v (err %.6e)",
		recovered[0], recovered[1], recovered[2], absErr)
}

// TestScalarQuantizeRoundTrip_4bit_OddDim verifies that 4-bit packing handles
// odd-dimension vectors correctly (low nibble of last byte is zero-padded).
func TestScalarQuantizeRoundTrip_4bit_OddDim(t *testing.T) {
	q, err := NewUniformQuantizer(-1.0, 1.0, 4)
	if err != nil {
		t.Fatalf("NewUniformQuantizer: %v", err)
	}

	vec := []float64{0.2, -0.5, 0.8} // 3 dims (odd)
	cv, err := q.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	if len(cv.Data) != 2 {
		t.Fatalf("expected 2 packed bytes for 3 dims, got %d", len(cv.Data))
	}

	recovered, err := q.Dequantize(cv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	for i := range vec {
		absErr := math.Abs(recovered[i] - vec[i])
		if absErr > q.MaxError()+1e-12 {
			t.Errorf("element %d: abs error %v exceeds MaxError %v", i, absErr, q.MaxError())
		}
	}
}

// ---------------------------------------------------------------------------
// Error-path tests
// ---------------------------------------------------------------------------

func TestScalarQuantizeDimMismatch(t *testing.T) {
	t.Run("8-bit", func(t *testing.T) {
		q, _ := NewUniformQuantizer(0.0, 10.0, 8)
		cv := CompressedVector{
			Data:    []byte{0, 1, 2}, // 3 bytes
			Dim:     5,               // claims 5 dims
			BitsPer: 8,
		}
		_, err := q.Dequantize(cv)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("error = %v, want ErrDimensionMismatch", err)
		}
	})

	t.Run("4-bit", func(t *testing.T) {
		q, _ := NewUniformQuantizer(0.0, 1.0, 4)
		cv := CompressedVector{
			Data:    []byte{0x12}, // 1 byte = 2 dims for 4-bit
			Dim:     4,            // claims 4 dims (needs 2 bytes)
			BitsPer: 4,
		}
		_, err := q.Dequantize(cv)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("error = %v, want ErrDimensionMismatch", err)
		}
	})
}

func TestNaNInput(t *testing.T) {
	q, err := NewUniformQuantizer(0.0, 10.0, 8)
	if err != nil {
		t.Fatalf("NewUniformQuantizer: %v", err)
	}

	t.Run("single NaN", func(t *testing.T) {
		vec := []float64{1.0, math.NaN(), 3.0}
		_, err := q.Quantize(vec)
		if !errors.Is(err, ErrNaNInput) {
			t.Errorf("error = %v, want ErrNaNInput", err)
		}
	})

	t.Run("all NaN", func(t *testing.T) {
		vec := []float64{math.NaN(), math.NaN()}
		_, err := q.Quantize(vec)
		if !errors.Is(err, ErrNaNInput) {
			t.Errorf("error = %v, want ErrNaNInput", err)
		}
	})

	t.Run("NaN at start", func(t *testing.T) {
		vec := []float64{math.NaN(), 1.0, 2.0}
		_, err := q.Quantize(vec)
		if !errors.Is(err, ErrNaNInput) {
			t.Errorf("error = %v, want ErrNaNInput", err)
		}
	})
}

func TestInfInput(t *testing.T) {
	q, err := NewUniformQuantizer(0.0, 10.0, 8)
	if err != nil {
		t.Fatalf("NewUniformQuantizer: %v", err)
	}

	vec := []float64{math.Inf(1), math.Inf(-1), 5.0}
	cv, err := q.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize(Inf) unexpected error: %v", err)
	}

	recovered, err := q.Dequantize(cv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	// +Inf → clamped to max
	if recovered[0] != 10.0 {
		t.Errorf("+Inf clamped: got %v, want 10.0", recovered[0])
	}
	// -Inf → clamped to min
	if recovered[1] != 0.0 {
		t.Errorf("-Inf clamped: got %v, want 0.0", recovered[1])
	}
	// 5.0 should be within error bound
	if math.Abs(recovered[2]-5.0) > q.MaxError()+1e-12 {
		t.Errorf("5.0: got %v, exceeds MaxError", recovered[2])
	}
}

func TestOutOfRangeClamp(t *testing.T) {
	q, err := NewUniformQuantizer(0.0, 10.0, 8)
	if err != nil {
		t.Fatalf("NewUniformQuantizer: %v", err)
	}

	vec := []float64{-5.0, 5.0, 15.0, 0.0, 10.0}
	cv, err := q.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	recovered, err := q.Dequantize(cv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	// -5.0 clamped to min
	if recovered[0] != 0.0 {
		t.Errorf("clamped -5.0: got %v, want 0.0", recovered[0])
	}
	// 5.0 within error bound
	if math.Abs(recovered[1]-5.0) > q.MaxError()+1e-12 {
		t.Errorf("5.0: got %v, exceeds MaxError", recovered[1])
	}
	// 15.0 clamped to max
	if recovered[2] != 10.0 {
		t.Errorf("clamped 15.0: got %v, want 10.0", recovered[2])
	}
	// 0.0 exact
	if recovered[3] != 0.0 {
		t.Errorf("0.0: got %v, want 0.0", recovered[3])
	}
	// 10.0 exact
	if recovered[4] != 10.0 {
		t.Errorf("10.0: got %v, want 10.0", recovered[4])
	}
}

// ---------------------------------------------------------------------------
// MaxError bound test
// ---------------------------------------------------------------------------

func TestMaxErrorBound(t *testing.T) {
	tests := []struct {
		min       float64
		max       float64
		bits      int
		wantError float64
	}{
		{0.0, 100.0, 8, 100.0 / 510.0}, // (max-min) / (2*(2^8-1))
		{0.0, 1.0, 4, 1.0 / 30.0},      // (1-0) / (2*(2^4-1))
		{-10.0, 10.0, 8, 20.0 / 510.0}, // (20) / (2*255)
		{-5.0, 5.0, 4, 10.0 / 30.0},    // (10) / (2*15)
	}

	for _, tt := range tests {
		name := "bits"
		t.Run(name, func(t *testing.T) {
			q, err := NewUniformQuantizer(tt.min, tt.max, tt.bits)
			if err != nil {
				t.Fatalf("NewUniformQuantizer: %v", err)
			}
			got := q.MaxError()
			if math.Abs(got-tt.wantError) > 1e-15 {
				t.Errorf("MaxError() = %.10e, want %.10e", got, tt.wantError)
			}
		})
	}
}

// TestMaxErrorBound_Exhaustive verifies that no value in the quantization range
// exceeds the reported MaxError after a round trip.
func TestMaxErrorBound_Exhaustive(t *testing.T) {
	q, err := NewUniformQuantizer(0.0, 10.0, 8)
	if err != nil {
		t.Fatalf("NewUniformQuantizer: %v", err)
	}

	maxErr := q.MaxError()
	// Test midpoints between every pair of adjacent code levels.
	levels := 255
	scale := 10.0 / 255.0
	for c := 0; c < levels; c++ {
		lower := float64(c) * scale
		upper := float64(c+1) * scale
		mid := (lower + upper) / 2.0

		vec := []float64{mid}
		cv, err := q.Quantize(vec)
		if err != nil {
			t.Fatalf("Quantize: %v", err)
		}
		recovered, err := q.Dequantize(cv)
		if err != nil {
			t.Fatalf("Dequantize: %v", err)
		}
		absErr := math.Abs(recovered[0] - mid)
		if absErr > maxErr+1e-12 {
			t.Errorf("midpoint %.4f (code %d): abs error %.6e > MaxError %.6e",
				mid, c, absErr, maxErr)
		}
	}
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func BenchmarkScalarQuantize(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	const dim = 768
	const nVectors = 10000

	// Pre-generate test vectors.
	vecs := make([][]float64, nVectors)
	for i := range vecs {
		vecs[i] = make([]float64, dim)
		for j := range vecs[i] {
			vecs[i][j] = rng.NormFloat64()
		}
	}

	b.Run("Quantize_8bit", func(b *testing.B) {
		q, _ := NewUniformQuantizer(-10.0, 10.0, 8)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q.Quantize(vecs[i%nVectors])
		}
	})

	b.Run("Quantize_4bit", func(b *testing.B) {
		q, _ := NewUniformQuantizer(-10.0, 10.0, 4)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q.Quantize(vecs[i%nVectors])
		}
	})

	b.Run("Dequantize_8bit", func(b *testing.B) {
		q, _ := NewUniformQuantizer(-10.0, 10.0, 8)
		// Pre-quantize all vectors.
		cvs := make([]CompressedVector, nVectors)
		for i := range vecs {
			cvs[i], _ = q.Quantize(vecs[i])
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q.Dequantize(cvs[i%nVectors])
		}
	})

	b.Run("Dequantize_4bit", func(b *testing.B) {
		q, _ := NewUniformQuantizer(-10.0, 10.0, 4)
		cvs := make([]CompressedVector, nVectors)
		for i := range vecs {
			cvs[i], _ = q.Quantize(vecs[i])
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q.Dequantize(cvs[i%nVectors])
		}
	})
}
