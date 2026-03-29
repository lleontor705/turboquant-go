package bits

import (
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// TestPack
// ---------------------------------------------------------------------------

// TestPack_Standard verifies the known bit pattern for a documented input.
//
// Input:  [+1, -1, +1, +1, -1, -1, +1, -1]
// Bit layout (bit 0 = LSB):
//
//	bit 0 = 1 (+1), bit 1 = 0 (-1), bit 2 = 1 (+1), bit 3 = 1 (+1),
//	bit 4 = 0 (-1), bit 5 = 0 (-1), bit 6 = 1 (+1), bit 7 = 0 (-1)
//
// Expected uint64: 1 + 4 + 8 + 64 = 77 = 0x4D
func TestPack_Standard(t *testing.T) {
	signs := []int8{+1, -1, +1, +1, -1, -1, +1, -1}
	packed, err := Pack(signs)
	if err != nil {
		t.Fatalf("Pack returned error: %v", err)
	}
	if len(packed) != 1 {
		t.Fatalf("expected 1 uint64, got %d", len(packed))
	}
	if packed[0] != 0x4D {
		t.Errorf("expected 0x4D, got 0x%X", packed[0])
	}
}

func TestPack_Empty(t *testing.T) {
	packed, err := Pack([]int8{})
	if err != nil {
		t.Fatalf("Pack([]) returned error: %v", err)
	}
	if len(packed) != 0 {
		t.Errorf("expected empty slice, got length %d", len(packed))
	}
}

func TestPack_InvalidValues(t *testing.T) {
	tests := []struct {
		name  string
		signs []int8
	}{
		{"zero value", []int8{+1, 0, -1}},
		{"positive two", []int8{+1, +2, -1}},
		{"negative two", []int8{-2}},
		{"int8 max", []int8{+1, -1, 127}},
		{"int8 min", []int8{-128}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Pack(tt.signs)
			if err == nil {
				t.Error("expected error for invalid sign values, got nil")
			}
			if err != ErrInvalidSignValue {
				t.Errorf("expected ErrInvalidSignValue, got %v", err)
			}
		})
	}
}

// TestPack_SingleElement verifies REQ-BITS-004: dim=1.
func TestPack_SingleElement(t *testing.T) {
	t.Run("+1", func(t *testing.T) {
		packed, err := Pack([]int8{+1})
		if err != nil {
			t.Fatal(err)
		}
		if len(packed) != 1 {
			t.Fatalf("expected 1 uint64, got %d", len(packed))
		}
		if packed[0] != 0x0000000000000001 {
			t.Errorf("expected 0x1, got 0x%X", packed[0])
		}
	})

	t.Run("-1", func(t *testing.T) {
		packed, err := Pack([]int8{-1})
		if err != nil {
			t.Fatal(err)
		}
		if len(packed) != 1 {
			t.Fatalf("expected 1 uint64, got %d", len(packed))
		}
		if packed[0] != 0 {
			t.Errorf("expected 0x0, got 0x%X", packed[0])
		}
	})
}

// TestPack_LittleEndianConvention verifies REQ-BITS-003:
// element 0 → bit 0 (LSB), element 1 → bit 1, etc.
func TestPack_LittleEndianConvention(t *testing.T) {
	// Helper: make a sign slice of length n with all -1, then set specific indices to +1.
	makeSigns := func(n int, plusOneAt ...int) []int8 {
		s := make([]int8, n)
		for i := range s {
			s[i] = -1
		}
		for _, idx := range plusOneAt {
			s[idx] = +1
		}
		return s
	}

	// Only element 0 is +1, rest are -1 (64 elements) → 0x0000000000000001
	packed, err := Pack(makeSigns(64, 0))
	if err != nil {
		t.Fatal(err)
	}
	if packed[0] != 0x0000000000000001 {
		t.Errorf("expected 0x1 for single-bit-at-position-0, got 0x%X", packed[0])
	}

	// Only element 1 is +1 → bit 1 set → 0x2
	packed2, err := Pack(makeSigns(64, 1))
	if err != nil {
		t.Fatal(err)
	}
	if packed2[0] != 0x0000000000000002 {
		t.Errorf("expected 0x2 for single-bit-at-position-1, got 0x%X", packed2[0])
	}

	// Only element 63 is +1 → bit 63 set → 0x8000000000000000
	packed63, err := Pack(makeSigns(64, 63))
	if err != nil {
		t.Fatal(err)
	}
	if packed63[0] != 0x8000000000000000 {
		t.Errorf("expected 0x8000000000000000 for bit 63, got 0x%X", packed63[0])
	}
}

// TestPack_CrossesWordBoundary verifies 65-element vectors (REQ-BITS-003):
// element 0 and element 64 are +1, rest are -1.
func TestPack_CrossesWordBoundary(t *testing.T) {
	signs := make([]int8, 65)
	signs[0] = +1
	signs[64] = +1
	for i := 1; i < 64; i++ {
		signs[i] = -1
	}

	packed, err := Pack(signs)
	if err != nil {
		t.Fatal(err)
	}
	if len(packed) != 2 {
		t.Fatalf("expected 2 uint64s, got %d", len(packed))
	}
	if packed[0] != 0x0000000000000001 {
		t.Errorf("word 0: expected 0x1, got 0x%X", packed[0])
	}
	if packed[1] != 0x0000000000000001 {
		t.Errorf("word 1: expected 0x1, got 0x%X", packed[1])
	}
}

// ---------------------------------------------------------------------------
// TestUnpack
// ---------------------------------------------------------------------------

func TestUnpack_Standard(t *testing.T) {
	// Pack a known sign vector, then unpack it and verify round-trip
	signs := []int8{+1, -1, +1, +1, -1, -1, +1, -1}
	packed, err := Pack(signs)
	if err != nil {
		t.Fatal(err)
	}

	unpacked, err := Unpack(packed, len(signs))
	if err != nil {
		t.Fatalf("Unpack returned error: %v", err)
	}
	if len(unpacked) != len(signs) {
		t.Fatalf("expected %d elements, got %d", len(signs), len(unpacked))
	}
	for i := range signs {
		if unpacked[i] != signs[i] {
			t.Errorf("element %d: expected %d, got %d", i, signs[i], unpacked[i])
		}
	}
}

func TestUnpack_NonMultipleOf64(t *testing.T) {
	tests := []struct {
		name string
		n    int
	}{
		{"dim=1", 1},
		{"dim=65", 65},
		{"dim=100", 100},
		{"dim=63", 63},
		{"dim=127", 127},
		{"dim=128", 128},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Generate a sign vector of length n
			signs := make([]int8, tt.n)
			for i := range signs {
				if i%2 == 0 {
					signs[i] = +1
				} else {
					signs[i] = -1
				}
			}

			packed, err := Pack(signs)
			if err != nil {
				t.Fatal(err)
			}

			unpacked, err := Unpack(packed, tt.n)
			if err != nil {
				t.Fatal(err)
			}

			if len(unpacked) != tt.n {
				t.Fatalf("expected %d elements, got %d", tt.n, len(unpacked))
			}
			for i := range signs {
				if unpacked[i] != signs[i] {
					t.Errorf("element %d: expected %d, got %d", i, signs[i], unpacked[i])
				}
			}
		})
	}
}

func TestUnpack_InsufficientData(t *testing.T) {
	tests := []struct {
		name   string
		packed []uint64
		n      int
	}{
		{
			name:   "need 2 words, have 1",
			packed: []uint64{0xFF},
			n:      100,
		},
		{
			name:   "empty packed, n=1",
			packed: []uint64{},
			n:      1,
		},
		{
			name:   "nil packed, n=1",
			packed: nil,
			n:      1,
		},
		{
			name:   "need 3 words, have 2",
			packed: []uint64{0, 0},
			n:      129,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Unpack(tt.packed, tt.n)
			if err == nil {
				t.Error("expected error for insufficient data, got nil")
			}
			if err != ErrInsufficientData {
				t.Errorf("expected ErrInsufficientData, got %v", err)
			}
		})
	}
}

func TestUnpack_ZeroLength(t *testing.T) {
	unpacked, err := Unpack([]uint64{}, 0)
	if err != nil {
		t.Fatalf("Unpack(_, 0) returned error: %v", err)
	}
	if len(unpacked) != 0 {
		t.Errorf("expected empty slice, got length %d", len(unpacked))
	}
}

func TestUnpack_NegativeLength(t *testing.T) {
	_, err := Unpack([]uint64{0xFF}, -1)
	if err == nil {
		t.Error("expected error for negative length, got nil")
	}
	if err != ErrInvalidDimension {
		t.Errorf("expected ErrInvalidDimension, got %v", err)
	}
}

// TestUnpack_PartialWord verifies that only the first n bits of a word
// are used when n < 64.
func TestUnpack_PartialWord(t *testing.T) {
	// packed = 0xFF (bits 0-7 set), n=5 → only bits 0-4 are used
	packed := []uint64{0xFF}
	unpacked, err := Unpack(packed, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(unpacked) != 5 {
		t.Fatalf("expected 5 elements, got %d", len(unpacked))
	}
	for i := 0; i < 5; i++ {
		if unpacked[i] != +1 {
			t.Errorf("element %d: expected +1, got %d", i, unpacked[i])
		}
	}
}

// ---------------------------------------------------------------------------
// TestBitPackingRoundTrip
// ---------------------------------------------------------------------------

// TestBitPackingRoundTrip verifies that Unpack(Pack(signs), len(signs))
// reproduces the original signs for dimensions 1-128.
func TestBitPackingRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	for n := 1; n <= 128; n++ {
		// Generate random signs
		signs := make([]int8, n)
		for i := range signs {
			if rng.Intn(2) == 0 {
				signs[i] = +1
			} else {
				signs[i] = -1
			}
		}

		packed, err := Pack(signs)
		if err != nil {
			t.Fatalf("Pack(dim=%d) error: %v", n, err)
		}

		unpacked, err := Unpack(packed, n)
		if err != nil {
			t.Fatalf("Unpack(dim=%d) error: %v", n, err)
		}

		for i := range signs {
			if unpacked[i] != signs[i] {
				t.Errorf("dim=%d element %d: expected %d, got %d",
					n, i, signs[i], unpacked[i])
				break // one failure per dim is enough
			}
		}
	}
}

// TestBitPackingRoundTrip_LargeDims verifies round-trip for selected
// large dimensions up to 4096.
func TestBitPackingRoundTrip_LargeDims(t *testing.T) {
	dims := []int{129, 200, 256, 512, 1000, 1024, 2048, 4096}
	rng := rand.New(rand.NewSource(123))

	for _, n := range dims {
		signs := make([]int8, n)
		for i := range signs {
			if rng.Intn(2) == 0 {
				signs[i] = +1
			} else {
				signs[i] = -1
			}
		}

		packed, err := Pack(signs)
		if err != nil {
			t.Fatalf("Pack(dim=%d) error: %v", n, err)
		}

		unpacked, err := Unpack(packed, n)
		if err != nil {
			t.Fatalf("Unpack(dim=%d) error: %v", n, err)
		}

		for i := range signs {
			if unpacked[i] != signs[i] {
				t.Errorf("dim=%d element %d: expected %d, got %d",
					n, i, signs[i], unpacked[i])
				break
			}
		}
	}
}

// TestBitPackingRoundTrip_PackUnpackPack verifies the idempotent property:
// Pack(Unpack(Pack(signs), len(signs))) == Pack(signs)
func TestBitPackingRoundTrip_PackUnpackPack(t *testing.T) {
	rng := rand.New(rand.NewSource(99))

	for n := 1; n <= 128; n++ {
		signs := make([]int8, n)
		for i := range signs {
			if rng.Intn(2) == 0 {
				signs[i] = +1
			} else {
				signs[i] = -1
			}
		}

		packed1, err := Pack(signs)
		if err != nil {
			t.Fatal(err)
		}

		unpacked, err := Unpack(packed1, n)
		if err != nil {
			t.Fatal(err)
		}

		packed2, err := Pack(unpacked)
		if err != nil {
			t.Fatal(err)
		}

		if len(packed1) != len(packed2) {
			t.Fatalf("dim=%d: packed lengths differ: %d vs %d",
				n, len(packed1), len(packed2))
		}

		for i := range packed1 {
			if packed1[i] != packed2[i] {
				t.Errorf("dim=%d word %d: 0x%X != 0x%X",
					n, i, packed1[i], packed2[i])
				break
			}
		}
	}
}

// ---------------------------------------------------------------------------
// TestPopCount
// ---------------------------------------------------------------------------

func TestPopCount(t *testing.T) {
	tests := []struct {
		name  string
		bits  []uint64
		count int
	}{
		{"empty", []uint64{}, 0},
		{"all zeros", []uint64{0, 0, 0}, 0},
		{"single bit", []uint64{1}, 1},
		{"all ones (64 bits)", []uint64{0xFFFFFFFFFFFFFFFF}, 64},
		{"0x4D (4 bits set)", []uint64{0x4D}, 4},
		{"two words", []uint64{0xFF, 0xFF}, 16},
		{"alternating bits", []uint64{0x5555555555555555}, 32},
		{"mixed", []uint64{0x0F, 0xF0}, 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PopCount(tt.bits)
			if got != tt.count {
				t.Errorf("PopCount(%v) = %d, want %d", tt.bits, got, tt.count)
			}
		})
	}
}

// TestPopCount_MatchesManual verifies PopCount matches manual bit counting
// after packing a known sign vector.
func TestPopCount_MatchesManual(t *testing.T) {
	// 8 signs: [+1, -1, +1, +1, -1, -1, +1, -1] → 4 bits set
	signs := []int8{+1, -1, +1, +1, -1, -1, +1, -1}
	packed, err := Pack(signs)
	if err != nil {
		t.Fatal(err)
	}

	count := PopCount(packed)
	if count != 4 {
		t.Errorf("PopCount after pack: expected 4, got %d", count)
	}
}
