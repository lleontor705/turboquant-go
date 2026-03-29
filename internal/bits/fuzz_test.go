package bits

import (
	"testing"
)

// FuzzBitPacking fuzzes the Pack/Unpack round-trip for the bits package.
// It checks the idempotent invariant:
//
//	Pack(Unpack(Pack(signs), len(signs))) == Pack(signs)
//
// The fuzzer provides raw bytes where each byte is converted to a sign:
// even byte → +1, odd byte → -1.
func FuzzBitPacking(f *testing.F) {
	// Seed corpus: various byte patterns producing different sign patterns.
	seeds := [][]byte{
		// All +1 (even bytes)
		{0, 2, 4, 6, 8, 10, 12, 14},
		// All -1 (odd bytes)
		{1, 3, 5, 7, 9, 11, 13, 15},
		// Alternating
		{0, 1, 0, 1, 0, 1, 0, 1},
		// Known pattern → [+1, -1, +1, +1, -1, -1, +1, -1] = 0x4D
		{0, 1, 0, 0, 1, 1, 0, 1},
		// Single +1
		{0},
		// Single -1
		{1},
		// 65 bytes (crosses word boundary)
		make([]byte, 65),
		// 64 bytes (exact word)
		make([]byte, 64),
		// 128 bytes (two words)
		make([]byte, 128),
	}

	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}

		// Convert bytes to signs: even → +1, odd → -1.
		signs := make([]int8, len(data))
		for i, b := range data {
			if b%2 == 0 {
				signs[i] = +1
			} else {
				signs[i] = -1
			}
		}

		// Step 1: Pack the signs.
		packed1, err := Pack(signs)
		if err != nil {
			t.Fatalf("Pack: %v", err)
		}

		// Step 2: Unpack back.
		unpacked, err := Unpack(packed1, len(signs))
		if err != nil {
			t.Fatalf("Unpack: %v", err)
		}

		// Step 3: Verify Unpack reproduces the original signs.
		if len(unpacked) != len(signs) {
			t.Fatalf("Unpack length: got %d, want %d", len(unpacked), len(signs))
		}
		for i := range signs {
			if unpacked[i] != signs[i] {
				t.Errorf("Unpack[%d]: got %d, want %d", i, unpacked[i], signs[i])
			}
		}

		// Step 4: Pack again and verify idempotent.
		packed2, err := Pack(unpacked)
		if err != nil {
			t.Fatalf("Pack (second): %v", err)
		}

		if len(packed1) != len(packed2) {
			t.Fatalf("packed length mismatch: %d vs %d", len(packed1), len(packed2))
		}
		for i := range packed1 {
			if packed1[i] != packed2[i] {
				t.Errorf("packed[%d]: 0x%X != 0x%X", i, packed1[i], packed2[i])
			}
		}
	})
}

// FuzzBitPacking_RandomDims is a separate fuzz target that generates
// arbitrary-length sign vectors from byte data and checks the full
// Pack → Unpack round-trip invariant.
func FuzzBitPacking_RandomDims(f *testing.F) {
	seeds := [][]byte{
		make([]byte, 7),
		make([]byte, 8),
		make([]byte, 9),
		make([]byte, 63),
		make([]byte, 64),
		make([]byte, 65),
		make([]byte, 100),
		make([]byte, 128),
		make([]byte, 200),
	}

	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}

		// Convert bytes to signs.
		signs := make([]int8, len(data))
		for i, b := range data {
			if b%2 == 0 {
				signs[i] = +1
			} else {
				signs[i] = -1
			}
		}

		packed, err := Pack(signs)
		if err != nil {
			t.Fatalf("Pack: %v", err)
		}

		// Verify expected packed length.
		expectedWords := (len(signs) + 63) / 64
		if len(packed) != expectedWords {
			t.Errorf("packed length: got %d, want %d", len(packed), expectedWords)
		}

		unpacked, err := Unpack(packed, len(signs))
		if err != nil {
			t.Fatalf("Unpack: %v", err)
		}

		for i := range signs {
			if unpacked[i] != signs[i] {
				t.Errorf("round-trip[%d]: got %d, want %d", i, unpacked[i], signs[i])
			}
		}
	})
}
