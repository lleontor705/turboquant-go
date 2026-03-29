package sketch

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/lleontor705/turboquant-go/internal/bits"
)

// QJLSketchCase represents a single test case from the QJL sketch fixture.
type QJLSketchCase struct {
	Name             string    `json:"name"`
	Dim              int       `json:"dim"`
	SketchDim        int       `json:"sketch_dim"`
	Seed             int64     `json:"seed"`
	OutlierK         int       `json:"outlier_k"`
	Input            []float64 `json:"input"`
	ExpectedBitCount int       `json:"expected_bit_count"`
	ExpectedDim      int       `json:"expected_dim"`
}

// QJLSketchFixture is the top-level fixture structure.
type QJLSketchFixture struct {
	Version int             `json:"version"`
	Cases   []QJLSketchCase `json:"cases"`
}

func loadQJLSketchFixture(t *testing.T) *QJLSketchFixture {
	t.Helper()
	data, err := os.ReadFile("../testdata/qjl_sketch.json")
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}
	var fixture QJLSketchFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("failed to parse fixture: %v", err)
	}
	return &fixture
}

func TestGoldenQJLSketch(t *testing.T) {
	fixture := loadQJLSketchFixture(t)

	if fixture.Version != 1 {
		t.Fatalf("unsupported fixture version: %d", fixture.Version)
	}

	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			sketcher, err := NewQJLSketcher(QJLOptions{
				Dim:       tc.Dim,
				SketchDim: tc.SketchDim,
				Seed:      tc.Seed,
				OutlierK:  tc.OutlierK,
			})
			if err != nil {
				t.Fatalf("NewQJLSketcher returned error: %v", err)
			}

			bv, err := sketcher.Sketch(tc.Input)
			if err != nil {
				t.Fatalf("Sketch returned error: %v", err)
			}

			// Verify output dimension
			if bv.Dim != tc.ExpectedDim {
				t.Errorf("BitVector.Dim = %d, want %d", bv.Dim, tc.ExpectedDim)
			}

			// Verify bit count (number of positive projections)
			gotBitCount := bits.PopCount(bv.Bits)
			if gotBitCount != tc.ExpectedBitCount {
				t.Errorf("PopCount(Bits) = %d, want %d", gotBitCount, tc.ExpectedBitCount)
			}

			// Verify packed bits length
			expectedWords := (tc.SketchDim + 63) / 64
			if len(bv.Bits) != expectedWords {
				t.Errorf("len(Bits) = %d, want %d", len(bv.Bits), expectedWords)
			}

			// Verify outlier handling
			if tc.OutlierK > 0 {
				if len(bv.OutlierIndices) != tc.OutlierK {
					t.Errorf("len(OutlierIndices) = %d, want %d",
						len(bv.OutlierIndices), tc.OutlierK)
				}
				if len(bv.OutlierValues) != tc.OutlierK {
					t.Errorf("len(OutlierValues) = %d, want %d",
						len(bv.OutlierValues), tc.OutlierK)
				}
			} else {
				if bv.OutlierIndices != nil {
					t.Errorf("OutlierIndices should be nil when OutlierK=0, got %v",
						bv.OutlierIndices)
				}
				if bv.OutlierValues != nil {
					t.Errorf("OutlierValues should be nil when OutlierK=0, got %v",
						bv.OutlierValues)
				}
			}

			// Verify determinism: same seed produces same result
			sketcher2, _ := NewQJLSketcher(QJLOptions{
				Dim:       tc.Dim,
				SketchDim: tc.SketchDim,
				Seed:      tc.Seed,
				OutlierK:  tc.OutlierK,
			})
			bv2, err := sketcher2.Sketch(tc.Input)
			if err != nil {
				t.Fatalf("Second Sketch returned error: %v", err)
			}
			if bv.Dim != bv2.Dim {
				t.Errorf("determinism check: Dim mismatch %d vs %d", bv.Dim, bv2.Dim)
			}
			for i := range bv.Bits {
				if bv.Bits[i] != bv2.Bits[i] {
					t.Errorf("determinism check: Bits word %d mismatch: 0x%X vs 0x%X",
						i, bv.Bits[i], bv2.Bits[i])
				}
			}
		})
	}
}

func TestGoldenQJLSketchSerialization(t *testing.T) {
	// Test that BitVector round-trips through MarshalBinary/UnmarshalBinary
	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:       8,
		SketchDim: 4,
		Seed:      42,
		OutlierK:  0,
	})
	if err != nil {
		t.Fatal(err)
	}

	vec := []float64{1, -2, 3, -4, 0.5, -0.5, 2, -1}
	bv, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatal(err)
	}

	// Marshal
	data, err := bv.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary error: %v", err)
	}

	// Unmarshal into new BitVector
	var bv2 BitVector
	if err := bv2.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary error: %v", err)
	}

	// Compare
	if bv.Dim != bv2.Dim {
		t.Errorf("Dim mismatch: %d vs %d", bv.Dim, bv2.Dim)
	}
	if len(bv.Bits) != len(bv2.Bits) {
		t.Fatalf("Bits length mismatch: %d vs %d", len(bv.Bits), len(bv2.Bits))
	}
	for i := range bv.Bits {
		if bv.Bits[i] != bv2.Bits[i] {
			t.Errorf("Bits[%d] mismatch: 0x%X vs 0x%X", i, bv.Bits[i], bv2.Bits[i])
		}
	}
}

func TestGoldenQJLSketchErrors(t *testing.T) {
	// Invalid dimensions
	_, err := NewQJLSketcher(QJLOptions{Dim: 0, SketchDim: 4, Seed: 42})
	if err == nil {
		t.Error("Dim=0 should be rejected")
	}

	_, err = NewQJLSketcher(QJLOptions{Dim: 8, SketchDim: 0, Seed: 42})
	if err == nil {
		t.Error("SketchDim=0 should be rejected")
	}

	_, err = NewQJLSketcher(QJLOptions{Dim: 4, SketchDim: 8, Seed: 42})
	if err == nil {
		t.Error("SketchDim > Dim should be rejected")
	}

	// Negative OutlierK
	_, err = NewQJLSketcher(QJLOptions{Dim: 8, SketchDim: 4, Seed: 42, OutlierK: -1})
	if err == nil {
		t.Error("negative OutlierK should be rejected")
	}

	// OutlierK > SketchDim
	_, err = NewQJLSketcher(QJLOptions{Dim: 8, SketchDim: 4, Seed: 42, OutlierK: 5})
	if err == nil {
		t.Error("OutlierK > SketchDim should be rejected")
	}

	// Wrong input dimension
	sketcher, _ := NewQJLSketcher(QJLOptions{Dim: 8, SketchDim: 4, Seed: 42})
	_, err = sketcher.Sketch([]float64{1, 2, 3}) // wrong dim
	if err == nil {
		t.Error("wrong input dimension should be rejected")
	}
}
