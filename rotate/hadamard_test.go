package rotate

import (
	"math"
	"math/bits"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// TestFWHT_KnownResult
// ---------------------------------------------------------------------------

// TestFWHT_KnownResult verifies REQ-ROTATE-002: FWHT([1,2,3,4]) == [10,-2,-4,0].
func TestFWHT_KnownResult(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	if err := FWHT(x); err != nil {
		t.Fatalf("FWHT returned error: %v", err)
	}

	expected := []float64{10, -2, -4, 0}
	for i := range expected {
		if x[i] != expected[i] {
			t.Errorf("x[%d] = %v, want %v", i, x[i], expected[i])
		}
	}
}

// ---------------------------------------------------------------------------
// TestFWHT_SingleElement
// ---------------------------------------------------------------------------

// TestFWHT_SingleElement verifies REQ-ROTATE-002: FWHT([7.5]) == [7.5].
// For n=1 the Hadamard transform is the identity.
func TestFWHT_SingleElement(t *testing.T) {
	x := []float64{7.5}
	if err := FWHT(x); err != nil {
		t.Fatalf("FWHT returned error: %v", err)
	}
	if x[0] != 7.5 {
		t.Errorf("x[0] = %v, want 7.5", x[0])
	}
}

// ---------------------------------------------------------------------------
// TestFWHT_NonPowerOf2
// ---------------------------------------------------------------------------

// TestFWHT_NonPowerOf2 verifies REQ-ROTATE-002: non-power-of-2 lengths
// return ErrNotPowerOfTwo without modifying the input.
func TestFWHT_NonPowerOf2(t *testing.T) {
	tests := []struct {
		name string
		n    int
	}{
		{"length 0", 0},
		{"length 3", 3},
		{"length 5", 5},
		{"length 6", 6},
		{"length 7", 7},
		{"length 9", 9},
		{"length 10", 10},
		{"length 100", 100},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := make([]float64, tt.n)
			for i := range x {
				x[i] = float64(i + 1)
			}

			err := FWHT(x)
			if err == nil {
				t.Error("expected error for non-power-of-2 length, got nil")
			}
			if err != ErrNotPowerOfTwo {
				t.Errorf("expected ErrNotPowerOfTwo, got %v", err)
			}

			// Verify input was NOT modified
			for i := range x {
				if x[i] != float64(i+1) {
					t.Errorf("input modified at index %d: got %v, want %v",
						i, x[i], float64(i+1))
					break
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestFWHT_Zeros
// ---------------------------------------------------------------------------

// TestFWHT_Zeros verifies REQ-ROTATE-004: zero vector → zero vector.
// The Hadamard transform of the zero vector is the zero vector.
func TestFWHT_Zeros(t *testing.T) {
	dims := []int{1, 2, 4, 8, 16, 64, 256, 1024}

	for _, n := range dims {
		t.Run(("dim=" + itoa(n)), func(t *testing.T) {
			x := make([]float64, n)
			if err := FWHT(x); err != nil {
				t.Fatalf("FWHT returned error: %v", err)
			}
			for i := range x {
				if x[i] != 0 {
					t.Errorf("x[%d] = %v, want 0", i, x[i])
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestFWHT_AgainstNaive
// ---------------------------------------------------------------------------

// TestFWHT_AgainstNaive verifies REQ-ROTATE-004:
// max|FWHT(x) - naive(x)| < 1e-12 for various power-of-2 dimensions.
//
// The naive Hadamard transform computes H*x where H_{ij} = (-1)^{popcount(i & j)}.
func TestFWHT_AgainstNaive(t *testing.T) {
	dims := []int{2, 4, 8, 16, 256, 1024}

	for _, n := range dims {
		t.Run(("dim=" + itoa(n)), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(n)))

			// Generate random vector
			x := make([]float64, n)
			for i := range x {
				x[i] = rng.NormFloat64()
			}

			// Compute FWHT
			fwhtResult := make([]float64, n)
			copy(fwhtResult, x)
			if err := FWHT(fwhtResult); err != nil {
				t.Fatalf("FWHT returned error: %v", err)
			}

			// Compute naive Hadamard transform
			naiveResult := naiveHadamard(x)

			// Compare
			maxErr := 0.0
			for i := 0; i < n; i++ {
				diff := math.Abs(fwhtResult[i] - naiveResult[i])
				if diff > maxErr {
					maxErr = diff
				}
			}

			if maxErr > 1e-12 {
				t.Errorf("max|FWHT - naive| = %v, want < 1e-12", maxErr)
			}
		})
	}
}

// naiveHadamard computes the Hadamard transform using the definition:
// H_{ij} = (-1)^{popcount(i & j)}.
// This is O(n^2) and used only for correctness verification.
func naiveHadamard(x []float64) []float64 {
	n := len(x)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			if bits.OnesCount64(uint64(i&j))%2 == 0 {
				sum += x[j]
			} else {
				sum -= x[j]
			}
		}
		result[i] = sum
	}
	return result
}

// ---------------------------------------------------------------------------
// TestIsPowerOfTwo
// ---------------------------------------------------------------------------

func TestIsPowerOfTwo(t *testing.T) {
	tests := []struct {
		n    int
		want bool
	}{
		{-1, false},
		{0, false},
		{1, true},
		{2, true},
		{3, false},
		{4, true},
		{5, false},
		{6, false},
		{7, false},
		{8, true},
		{16, true},
		{256, true},
		{1024, true},
		{1023, false},
		{1025, false},
	}

	for _, tt := range tests {
		t.Run(itoa(tt.n), func(t *testing.T) {
			if got := IsPowerOfTwo(tt.n); got != tt.want {
				t.Errorf("IsPowerOfTwo(%d) = %v, want %v", tt.n, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// BenchmarkFWHT
// ---------------------------------------------------------------------------

func BenchmarkFWHT(b *testing.B) {
	dims := []int{256, 1024, 4096}

	for _, n := range dims {
		b.Run(("dim=" + itoa(n)), func(b *testing.B) {
			x := make([]float64, n)
			for i := range x {
				x[i] = float64(i)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Reset input for each iteration
				for j := 0; j < n; j++ {
					x[j] = float64(j)
				}
				if err := FWHT(x); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// itoa converts an int to a string without importing strconv (avoids
// unnecessary import in test files that don't otherwise need it).
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	var buf [20]byte
	pos := len(buf)
	for n > 0 {
		pos--
		buf[pos] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
