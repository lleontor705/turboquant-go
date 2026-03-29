package bits

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// BitPackCase represents a single test case from the bitpacking fixture.
type BitPackCase struct {
	Name           string   `json:"name"`
	Input          []int8   `json:"input"`
	ExpectedPacked []uint64 `json:"expected_packed"`
	ExpectedDim    int      `json:"expected_dim"`
}

// BitPackFixture is the top-level fixture structure.
type BitPackFixture struct {
	Version int           `json:"version"`
	Cases   []BitPackCase `json:"cases"`
}

func loadBitPackFixture(t *testing.T) *BitPackFixture {
	t.Helper()
	data, err := os.ReadFile("../../testdata/bitpacking.json")
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}
	var fixture BitPackFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("failed to parse fixture: %v", err)
	}
	return &fixture
}

func TestGoldenBitPacking(t *testing.T) {
	fixture := loadBitPackFixture(t)

	if fixture.Version != 1 {
		t.Fatalf("unsupported fixture version: %d", fixture.Version)
	}

	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			packed, err := Pack(tc.Input)
			if err != nil {
				t.Fatalf("Pack returned error: %v", err)
			}

			// Verify dimension
			if len(tc.Input) != tc.ExpectedDim {
				t.Errorf("input length %d != expected dim %d", len(tc.Input), tc.ExpectedDim)
			}

			// Verify packed output
			if len(packed) != len(tc.ExpectedPacked) {
				t.Fatalf("packed length %d != expected %d", len(packed), len(tc.ExpectedPacked))
			}
			for i, got := range packed {
				if got != tc.ExpectedPacked[i] {
					t.Errorf("word %d: got 0x%X, want 0x%X", i, got, tc.ExpectedPacked[i])
				}
			}

			// Verify round-trip: Unpack(Pack(x)) == x
			unpacked, err := Unpack(packed, tc.ExpectedDim)
			if err != nil {
				t.Fatalf("Unpack returned error: %v", err)
			}
			if len(unpacked) != len(tc.Input) {
				t.Fatalf("round-trip length %d != original %d", len(unpacked), len(tc.Input))
			}
			for i, got := range unpacked {
				if got != tc.Input[i] {
					t.Errorf("round-trip element %d: got %d, want %d", i, got, tc.Input[i])
				}
			}

			// Verify PopCount matches number of +1 signs
			expectedPopCount := 0
			for _, s := range tc.Input {
				if s == 1 {
					expectedPopCount++
				}
			}
			gotPopCount := PopCount(packed)
			if gotPopCount != expectedPopCount {
				t.Errorf("PopCount: got %d, want %d", gotPopCount, expectedPopCount)
			}
		})
	}
}

// TestGoldenBitPackingEmpty tests the empty input edge case.
func TestGoldenBitPackingEmpty(t *testing.T) {
	packed, err := Pack([]int8{})
	if err != nil {
		t.Fatalf("Pack(empty) returned error: %v", err)
	}
	if len(packed) != 0 {
		t.Errorf("Pack(empty) returned %d words, want 0", len(packed))
	}

	unpacked, err := Unpack(nil, 0)
	if err != nil {
		t.Fatalf("Unpack(nil, 0) returned error: %v", err)
	}
	if len(unpacked) != 0 {
		t.Errorf("Unpack(nil, 0) returned %d elements, want 0", len(unpacked))
	}
}

func TestGoldenBitPackingInvalid(t *testing.T) {
	_, err := Pack([]int8{1, 0, -1})
	if err == nil {
		t.Error("Pack with invalid sign value 0 should return error")
	}

	_, err = Unpack(nil, 5)
	if err == nil {
		t.Error("Unpack with insufficient data should return error")
	}

	_, err = Unpack(nil, -1)
	if err == nil {
		t.Error("Unpack with negative dimension should return error")
	}
}

func ExamplePack() {
	signs := []int8{1, -1, 1, 1, -1, -1, 1, -1}
	packed, _ := Pack(signs)
	fmt.Printf("0x%X\n", packed[0])
	// Output: 0x4D
}
