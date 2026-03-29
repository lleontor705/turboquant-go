package quantize

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// ---------------------------------------------------------------------------
// TestNewTurboQuantizer
// ---------------------------------------------------------------------------

func TestNewTurboQuantizer(t *testing.T) {
	t.Run("valid construction", func(t *testing.T) {
		tq, err := NewTurboQuantizer(128, 4, 42)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if tq.Dim() != 128 {
			t.Errorf("Dim() = %d, want 128", tq.Dim())
		}
		if tq.Bits() != 4 {
			t.Errorf("Bits() = %d, want 4", tq.Bits())
		}
		if tq.codebook == nil {
			t.Error("codebook is nil")
		}
		if tq.rotation == nil {
			t.Error("rotation is nil")
		}
	})

	t.Run("invalid dim", func(t *testing.T) {
		tests := []struct {
			name string
			dim  int
		}{
			{"zero", 0},
			{"negative", -1},
			{"one", 1},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				_, err := NewTurboQuantizer(tt.dim, 4, 42)
				if err == nil {
					t.Error("expected error for invalid dim")
				}
			})
		}
	})

	t.Run("invalid bits", func(t *testing.T) {
		tests := []struct {
			name string
			bits int
		}{
			{"zero", 0},
			{"negative", -1},
			{"five", 5},
			{"eight", 8},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				_, err := NewTurboQuantizer(128, tt.bits, 42)
				if err == nil {
					t.Error("expected error for invalid bits")
				}
			})
		}
	})
}

// ---------------------------------------------------------------------------
// Round-trip tests with MSE targets
// ---------------------------------------------------------------------------

// generateUnitVectors creates n random unit-norm vectors of the given dimension.
func generateUnitVectors(rng *rand.Rand, n, dim int) [][]float64 {
	vecs := make([][]float64, n)
	for i := 0; i < n; i++ {
		v := make([]float64, dim)
		for j := range v {
			v[j] = rng.NormFloat64()
		}
		// Normalize.
		var norm float64
		for _, x := range v {
			norm += x * x
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		vecs[i] = v
	}
	return vecs
}

// computeMSE computes mean squared error between original and reconstructed vectors.
func computeMSE(original, reconstructed [][]float64) float64 {
	var totalSqErr float64
	var count float64
	for i := range original {
		for j := range original[i] {
			d := original[i][j] - reconstructed[i][j]
			totalSqErr += d * d
			count++
		}
	}
	return totalSqErr / count
}

func TestTurboQuantize_RoundTrip_1bit(t *testing.T) {
	testRoundTrip(t, 1, 0.45)
}

func TestTurboQuantize_RoundTrip_2bit(t *testing.T) {
	testRoundTrip(t, 2, 0.15)
}

func TestTurboQuantize_RoundTrip_3bit(t *testing.T) {
	testRoundTrip(t, 3, 0.05)
}

func TestTurboQuantize_RoundTrip_4bit(t *testing.T) {
	testRoundTrip(t, 4, 0.015)
}

func testRoundTrip(t *testing.T, bits int, maxMSE float64) {
	t.Helper()
	dim := 128
	numVecs := 50

	tq, err := NewTurboQuantizer(dim, bits, 42)
	if err != nil {
		t.Fatalf("NewTurboQuantizer(%d, %d): %v", dim, bits, err)
	}

	rng := rand.New(rand.NewSource(99))
	vecs := generateUnitVectors(rng, numVecs, dim)

	reconstructed := make([][]float64, numVecs)
	for i, v := range vecs {
		cv, err := tq.Quantize(v)
		if err != nil {
			t.Fatalf("Quantize vec %d: %v", i, err)
		}
		if cv.Dim != dim {
			t.Errorf("cv.Dim = %d, want %d", cv.Dim, dim)
		}
		if cv.BitsPer != bits {
			t.Errorf("cv.BitsPer = %d, want %d", cv.BitsPer, bits)
		}

		rec, err := tq.Dequantize(cv)
		if err != nil {
			t.Fatalf("Dequantize vec %d: %v", i, err)
		}
		if len(rec) != dim {
			t.Errorf("len(rec) = %d, want %d", len(rec), dim)
		}
		reconstructed[i] = rec
	}

	mse := computeMSE(vecs, reconstructed)
	t.Logf("bits=%d dim=%d numVecs=%d MSE=%.6f", bits, dim, numVecs, mse)

	if mse > maxMSE {
		t.Errorf("MSE = %.6f, want <= %.6f", mse, maxMSE)
	}
}

// ---------------------------------------------------------------------------
// TestTurboQuantize_Determinism
// ---------------------------------------------------------------------------

func TestTurboQuantize_Determinism(t *testing.T) {
	dim := 64
	bits := 4
	seed := int64(12345)

	tq1, err := NewTurboQuantizer(dim, bits, seed)
	if err != nil {
		t.Fatal(err)
	}
	tq2, err := NewTurboQuantizer(dim, bits, seed)
	if err != nil {
		t.Fatal(err)
	}

	// Generate a test vector.
	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	cv1, err := tq1.Quantize(vec)
	if err != nil {
		t.Fatal(err)
	}
	cv2, err := tq2.Quantize(vec)
	if err != nil {
		t.Fatal(err)
	}

	// Check packed data is identical.
	if len(cv1.Data) != len(cv2.Data) {
		t.Fatalf("data length mismatch: %d vs %d", len(cv1.Data), len(cv2.Data))
	}
	for i := range cv1.Data {
		if cv1.Data[i] != cv2.Data[i] {
			t.Errorf("byte %d: 0x%02X != 0x%02X", i, cv1.Data[i], cv2.Data[i])
		}
	}

	// Check rotation matrices are identical.
	r1 := tq1.RotationMatrix()
	r2 := tq2.RotationMatrix()
	rows, cols := r1.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if r1.At(i, j) != r2.At(i, j) {
				t.Errorf("rotation[%d][%d]: %v != %v", i, j, r1.At(i, j), r2.At(i, j))
			}
		}
	}
}

// ---------------------------------------------------------------------------
// TestTurboQuantize_DimMismatch
// ---------------------------------------------------------------------------

func TestTurboQuantize_DimMismatch(t *testing.T) {
	tq, err := NewTurboQuantizer(64, 4, 42)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("wrong dimension input", func(t *testing.T) {
		_, err := tq.Quantize(make([]float64, 32))
		if err == nil {
			t.Error("expected error for wrong dimension")
		}
	})

	t.Run("wrong dimension compressed vector", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{0}, Dim: 32, BitsPer: 4}
		_, err := tq.Dequantize(cv)
		if err == nil {
			t.Error("expected error for wrong dimension")
		}
	})

	t.Run("wrong bits in compressed vector", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{0}, Dim: 64, BitsPer: 2}
		_, err := tq.Dequantize(cv)
		if err == nil {
			t.Error("expected error for wrong bits")
		}
	})
}

// ---------------------------------------------------------------------------
// TestTurboQuantize_PreservesNorm
// ---------------------------------------------------------------------------

func TestTurboQuantize_PreservesNorm(t *testing.T) {
	dim := 128
	tq, err := NewTurboQuantizer(dim, 4, 42)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 20; trial++ {
		vec := make([]float64, dim)
		for i := range vec {
			vec[i] = rng.NormFloat64()
		}
		// Normalize to unit.
		var norm float64
		for _, v := range vec {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		for i := range vec {
			vec[i] /= norm
		}

		cv, err := tq.Quantize(vec)
		if err != nil {
			t.Fatal(err)
		}

		rec, err := tq.Dequantize(cv)
		if err != nil {
			t.Fatal(err)
		}

		var recNorm float64
		for _, v := range rec {
			recNorm += v * v
		}
		recNorm = math.Sqrt(recNorm)

		// Reconstructed norm should be close to 1.0 (within ~10% tolerance).
		if math.Abs(recNorm-1.0) > 0.15 {
			t.Errorf("trial %d: reconstructed norm = %.4f, expected ~1.0", trial, recNorm)
		}
	}
}

// ---------------------------------------------------------------------------
// TestTurboQuantize_ImplementsInterface
// ---------------------------------------------------------------------------

func TestTurboQuantize_ImplementsInterface(t *testing.T) {
	// Compile-time check that TurboQuantizer implements Quantizer.
	var _ Quantizer = (*TurboQuantizer)(nil)

	tq, err := NewTurboQuantizer(32, 4, 0)
	if err != nil {
		t.Fatal(err)
	}

	// Runtime check via interface assertion.
	var q Quantizer = tq
	if q.Bits() != 4 {
		t.Errorf("Bits() = %d, want 4", q.Bits())
	}
}

// ---------------------------------------------------------------------------
// TestPackUnpackIndices
// ---------------------------------------------------------------------------

func TestPackUnpackIndices(t *testing.T) {
	tests := []struct {
		bits    int
		indices []int
	}{
		{1, []int{0, 1, 1, 0, 1, 0, 0, 1}},
		{2, []int{0, 1, 2, 3, 0, 1, 2, 3}},
		{3, []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{4, []int{0, 5, 10, 15, 3, 7, 12, 8}},
		{1, []int{0, 1}},
		{2, []int{3}},
		{3, []int{0, 7, 3, 5}},
		{4, []int{15, 0, 8, 7}},
	}

	for _, tt := range tests {
		t.Run(fmtBits(tt.bits, tt.indices), func(t *testing.T) {
			packed := packIndices(tt.indices, tt.bits)
			unpacked := unpackIndices(packed, len(tt.indices), tt.bits)

			if len(unpacked) != len(tt.indices) {
				t.Fatalf("length mismatch: got %d, want %d", len(unpacked), len(tt.indices))
			}
			for i := range tt.indices {
				if unpacked[i] != tt.indices[i] {
					t.Errorf("index %d: got %d, want %d", i, unpacked[i], tt.indices[i])
				}
			}
		})
	}
}

func fmtBits(bits int, indices []int) string {
	return fmt.Sprintf("bits=%d_n=%d", bits, len(indices))
}

// TestPackUnpackIndices_EdgeCases tests boundary conditions.
func TestPackUnpackIndices_EdgeCases(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		packed := packIndices(nil, 4)
		if len(packed) != 0 {
			t.Errorf("expected empty, got %d bytes", len(packed))
		}
		unpacked := unpackIndices(nil, 0, 4)
		if len(unpacked) != 0 {
			t.Errorf("expected empty, got %d indices", len(unpacked))
		}
	})

	t.Run("single index", func(t *testing.T) {
		for bits := 1; bits <= 4; bits++ {
			maxIdx := (1 << bits) - 1
			for _, idx := range []int{0, maxIdx / 2, maxIdx} {
				packed := packIndices([]int{idx}, bits)
				unpacked := unpackIndices(packed, 1, bits)
				if unpacked[0] != idx {
					t.Errorf("bits=%d idx=%d: got %d, want %d", bits, idx, unpacked[0], idx)
				}
			}
		}
	})

	t.Run("large number of indices", func(t *testing.T) {
		rng := rand.New(rand.NewSource(42))
		for _, bits := range []int{1, 2, 3, 4} {
			n := 1024
			maxVal := 1<<bits - 1
			indices := make([]int, n)
			for i := range indices {
				indices[i] = rng.Intn(maxVal + 1)
			}
			packed := packIndices(indices, bits)
			unpacked := unpackIndices(packed, n, bits)

			for i := range indices {
				if unpacked[i] != indices[i] {
					t.Errorf("bits=%d index %d: got %d, want %d", bits, i, unpacked[i], indices[i])
					break
				}
			}
		}
	})
}

// ---------------------------------------------------------------------------
// TestTurboQuantize_Concurrent
// ---------------------------------------------------------------------------

func TestTurboQuantize_Concurrent(t *testing.T) {
	dim := 64
	bits := 4
	tq, err := NewTurboQuantizer(dim, bits, 42)
	if err != nil {
		t.Fatal(err)
	}

	const numGoroutines = 50

	type result struct {
		rec []float64
		err error
	}
	results := make([]result, numGoroutines)

	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			// Each goroutine gets its own unit-norm vector.
			localRng := rand.New(rand.NewSource(int64(idx)*1000 + 7))
			vec := make([]float64, dim)
			for j := range vec {
				vec[j] = localRng.NormFloat64()
			}
			// Normalize to unit norm (matching what Quantize expects).
			var norm float64
			for _, v := range vec {
				norm += v * v
			}
			norm = math.Sqrt(norm)
			if norm > 0 {
				for j := range vec {
					vec[j] /= norm
				}
			}

			cv, err := tq.Quantize(vec)
			if err != nil {
				results[idx] = result{err: err}
				return
			}
			rec, err := tq.Dequantize(cv)
			if err != nil {
				results[idx] = result{err: err}
				return
			}

			// Compute per-vector MSE between unit-norm input and reconstruction.
			var sqErr float64
			for j := range vec {
				d := vec[j] - rec[j]
				sqErr += d * d
			}
			mse := sqErr / float64(dim)
			if mse > 0.5 {
				results[idx] = result{err: fmt.Errorf("MSE = %.4f, too large", mse)}
				return
			}
			results[idx] = result{rec: rec}
		}(i)
	}
	wg.Wait()

	// Verify all succeeded.
	for i, r := range results {
		if r.err != nil {
			t.Errorf("goroutine %d: %v", i, r.err)
			continue
		}
		if len(r.rec) != dim {
			t.Errorf("goroutine %d: len(rec) = %d, want %d", i, len(r.rec), dim)
		}
	}
}

// ---------------------------------------------------------------------------
// TestTurboQuantize_NaNInput
// ---------------------------------------------------------------------------

func TestTurboQuantize_NaNInput(t *testing.T) {
	tq, err := NewTurboQuantizer(32, 4, 42)
	if err != nil {
		t.Fatal(err)
	}

	vec := make([]float64, 32)
	vec[10] = math.NaN()

	_, err = tq.Quantize(vec)
	if err == nil {
		t.Error("expected error for NaN input")
	}
	if err != ErrNaNInput {
		t.Errorf("expected ErrNaNInput, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// TestTurboQuantize_RotationMatrix
// ---------------------------------------------------------------------------

func TestTurboQuantize_RotationMatrix(t *testing.T) {
	tq, err := NewTurboQuantizer(16, 4, 42)
	if err != nil {
		t.Fatal(err)
	}

	rm := tq.RotationMatrix()
	rows, cols := rm.Dims()
	if rows != 16 || cols != 16 {
		t.Fatalf("rotation matrix dims: %d×%d, want 16×16", rows, cols)
	}

	// Verify orthogonality: R^T * R should be identity.
	// Build explicit transpose as a Dense matrix.
	rtData := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			rtData[j*rows+i] = rm.At(i, j)
		}
	}
	rt := mat.NewDense(rows, cols, rtData)

	var product mat.Dense
	product.Mul(rt, rm)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			got := product.At(i, j)
			want := 0.0
			if i == j {
				want = 1.0
			}
			if math.Abs(got-want) > 1e-10 {
				t.Errorf("R^T*R[%d][%d] = %.12f, want %.12f", i, j, got, want)
			}
		}
	}

	// Verify that modifying the returned copy doesn't affect the original.
	rm.Set(0, 0, 999.0)
	if tq.rotation.At(0, 0) == 999.0 {
		t.Error("modifying returned matrix affected internal state")
	}
}
