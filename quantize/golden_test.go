package quantize

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// ScalarQCase represents a single test case from the scalar quantize fixture.
type ScalarQCase struct {
	Name           string    `json:"name"`
	Min            float64   `json:"min"`
	Max            float64   `json:"max"`
	Bits           int       `json:"bits"`
	Input          []float64 `json:"input"`
	ExpectedApprox []float64 `json:"expected_approx"`
}

// ScalarQFixture is the top-level fixture structure.
type ScalarQFixture struct {
	Version int           `json:"version"`
	Cases   []ScalarQCase `json:"cases"`
}

func loadScalarQFixture(t *testing.T) *ScalarQFixture {
	t.Helper()
	data, err := os.ReadFile("../testdata/scalar_quantize.json")
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}
	var fixture ScalarQFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("failed to parse fixture: %v", err)
	}
	return &fixture
}

func TestGoldenScalarQuantize(t *testing.T) {
	fixture := loadScalarQFixture(t)

	if fixture.Version != 1 {
		t.Fatalf("unsupported fixture version: %d", fixture.Version)
	}

	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			q, err := NewUniformQuantizer(tc.Min, tc.Max, tc.Bits)
			if err != nil {
				t.Fatalf("NewUniformQuantizer returned error: %v", err)
			}

			// Quantize
			cv, err := q.Quantize(tc.Input)
			if err != nil {
				t.Fatalf("Quantize returned error: %v", err)
			}

			// Verify dimension
			if cv.Dim != len(tc.Input) {
				t.Errorf("CompressedVector.Dim = %d, want %d", cv.Dim, len(tc.Input))
			}
			if cv.BitsPer != tc.Bits {
				t.Errorf("CompressedVector.BitsPer = %d, want %d", cv.BitsPer, tc.Bits)
			}
			if cv.Min != tc.Min {
				t.Errorf("CompressedVector.Min = %f, want %f", cv.Min, tc.Min)
			}
			if cv.Max != tc.Max {
				t.Errorf("CompressedVector.Max = %f, want %f", cv.Max, tc.Max)
			}

			// Dequantize
			deq, err := q.Dequantize(cv)
			if err != nil {
				t.Fatalf("Dequantize returned error: %v", err)
			}

			if len(deq) != len(tc.ExpectedApprox) {
				t.Fatalf("dequantized length %d != expected %d", len(deq), len(tc.ExpectedApprox))
			}

			// Compare with tolerance based on max error
			maxErr := q.MaxError()
			tolerance := maxErr * 1.01 // Allow 1% margin over theoretical max error

			for i, got := range deq {
				want := tc.ExpectedApprox[i]
				diff := math.Abs(got - want)
				if diff > tolerance {
					t.Errorf("element %d: got %.10f, want approx %.10f (diff %.2e, tolerance %.2e)",
						i, got, want, diff, tolerance)
				}
			}
		})
	}
}

func TestGoldenScalarQuantizeRoundTrip(t *testing.T) {
	// Test that quantize→dequantize round-trip error is bounded by MaxError
	inputs := []struct {
		min  float64
		max  float64
		bits int
		vec  []float64
	}{
		{0, 10, 8, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
		{-5, 5, 4, []float64{-5, -3, -1, 0, 1, 3, 5}},
		{0, 1, 8, []float64{0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}},
	}

	for _, tt := range inputs {
		name := fmt.Sprintf("%dbit_%.0f_%.0f", tt.bits, tt.min, tt.max)
		t.Run(name, func(t *testing.T) {
			q, err := NewUniformQuantizer(tt.min, tt.max, tt.bits)
			if err != nil {
				t.Fatal(err)
			}

			cv, err := q.Quantize(tt.vec)
			if err != nil {
				t.Fatal(err)
			}

			deq, err := q.Dequantize(cv)
			if err != nil {
				t.Fatal(err)
			}

			maxErr := q.MaxError()
			for i, orig := range tt.vec {
				diff := math.Abs(deq[i] - orig)
				if diff > maxErr {
					t.Errorf("element %d: round-trip error %.6e exceeds max %.6e",
						i, diff, maxErr)
				}
			}
		})
	}
}

func TestGoldenScalarQuantizeErrors(t *testing.T) {
	// Invalid bit widths
	_, err := NewUniformQuantizer(0, 10, 3)
	if err == nil {
		t.Error("3-bit quantizer should be rejected")
	}
	if err != ErrInvalidConfig {
		t.Errorf("expected ErrInvalidConfig, got: %v", err)
	}

	_, err = NewUniformQuantizer(0, 10, 16)
	if err == nil {
		t.Error("16-bit quantizer should be rejected")
	}

	// Inverted range
	_, err = NewUniformQuantizer(10, 0, 8)
	if err == nil {
		t.Error("inverted range should be rejected")
	}

	// Degenerate range
	_, err = NewUniformQuantizer(5, 5, 8)
	if err == nil {
		t.Error("degenerate range should be rejected")
	}

	// NaN input
	q, _ := NewUniformQuantizer(0, 10, 8)
	_, err = q.Quantize([]float64{5, math.NaN()})
	if err == nil {
		t.Error("NaN input should be rejected")
	}
	if err != ErrNaNInput {
		t.Errorf("expected ErrNaNInput, got: %v", err)
	}
}
