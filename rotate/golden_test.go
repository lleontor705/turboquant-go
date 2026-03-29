package rotate

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// FWHTECase represents a single test case from the FWHT fixture.
type FWHTECase struct {
	Name     string    `json:"name"`
	Input    []float64 `json:"input"`
	Expected []float64 `json:"expected"`
}

// FWHTFixture is the top-level fixture structure.
type FWHTFixture struct {
	Version int         `json:"version"`
	Cases   []FWHTECase `json:"cases"`
}

func loadFWHTFixture(t *testing.T) *FWHTFixture {
	t.Helper()
	data, err := os.ReadFile("../testdata/fwht.json")
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}
	var fixture FWHTFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("failed to parse fixture: %v", err)
	}
	return &fixture
}

const fwhtTolerance = 1e-9

func TestGoldenFWHT(t *testing.T) {
	fixture := loadFWHTFixture(t)

	if fixture.Version != 1 {
		t.Fatalf("unsupported fixture version: %d", fixture.Version)
	}

	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			// Copy input since FWHT is in-place
			x := make([]float64, len(tc.Input))
			copy(x, tc.Input)

			err := FWHT(x)
			if err != nil {
				t.Fatalf("FWHT returned error: %v", err)
			}

			if len(x) != len(tc.Expected) {
				t.Fatalf("result length %d != expected %d", len(x), len(tc.Expected))
			}

			for i, got := range x {
				want := tc.Expected[i]
				if diff := math.Abs(got - want); diff > fwhtTolerance {
					t.Errorf("element %d: got %.10f, want %.10f (diff %.2e)",
						i, got, want, diff)
				}
			}
		})
	}
}

func TestGoldenFWHTEdgeCases(t *testing.T) {
	// Non-power-of-2 should return error
	err := FWHT(make([]float64, 3))
	if err == nil {
		t.Error("FWHT with non-power-of-2 length should return error")
	}
	if err != ErrNotPowerOfTwo {
		t.Errorf("expected ErrNotPowerOfTwo, got: %v", err)
	}

	// Empty slice is not a power of 2
	err = FWHT([]float64{})
	if err == nil {
		t.Error("FWHT with empty slice should return error")
	}

	// Verify idempotency: H(H(x)) = n*x (unnormalized)
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	copy_x := make([]float64, len(x))
	copy(copy_x, x)

	FWHT(x) // First transform
	FWHT(x) // Second transform

	n := float64(len(x))
	for i, got := range x {
		want := copy_x[i] * n
		if math.Abs(got-want) > fwhtTolerance {
			t.Errorf("idempotency check element %d: got %.10f, want %.10f",
				i, got, want)
		}
	}
}

func TestGoldenIsPowerOfTwo(t *testing.T) {
	tests := []struct {
		n    int
		want bool
	}{
		{1, true},
		{2, true},
		{4, true},
		{8, true},
		{1024, true},
		{0, false},
		{3, false},
		{5, false},
		{6, false},
		{7, false},
		{-1, false},
		{-4, false},
	}
	for _, tt := range tests {
		got := IsPowerOfTwo(tt.n)
		if got != tt.want {
			t.Errorf("IsPowerOfTwo(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}
