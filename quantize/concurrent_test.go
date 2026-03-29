package quantize

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
)

// TestConcurrentQuantize verifies that a single UniformQuantizer is safe for
// concurrent use. 50 goroutines each call Quantize and Dequantize on their own
// vectors; we check correctness and rely on the race detector to flag data races.
func TestConcurrentQuantize(t *testing.T) {
	const dim = 128
	const goroutines = 50
	const minVal, maxVal = -10.0, 10.0

	q8, err := NewUniformQuantizer(minVal, maxVal, 8)
	if err != nil {
		t.Fatalf("NewUniformQuantizer 8-bit: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	vecs := make([][]float64, goroutines)
	for i := range vecs {
		vecs[i] = make([]float64, dim)
		for j := range vecs[i] {
			vecs[i][j] = minVal + rng.Float64()*(maxVal-minVal)
		}
	}

	// Pre-compute expected results sequentially.
	type result struct {
		cv CompressedVector
		rt []float64 // round-trip values
	}
	expected := make([]result, goroutines)
	for i, v := range vecs {
		cv, err := q8.Quantize(v)
		if err != nil {
			t.Fatalf("Quantize[%d]: %v", i, err)
		}
		rt, err := q8.Dequantize(cv)
		if err != nil {
			t.Fatalf("Dequantize[%d]: %v", i, err)
		}
		expected[i] = result{cv: cv, rt: rt}
	}

	var wg sync.WaitGroup
	var errors atomic.Int64

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			cv, err := q8.Quantize(vecs[idx])
			if err != nil {
				t.Errorf("[%d] Quantize: %v", idx, err)
				errors.Add(1)
				return
			}

			if cv.Dim != expected[idx].cv.Dim {
				t.Errorf("[%d] Dim mismatch: got %d, want %d", idx, cv.Dim, expected[idx].cv.Dim)
				errors.Add(1)
				return
			}
			if len(cv.Data) != len(expected[idx].cv.Data) {
				t.Errorf("[%d] Data length mismatch: got %d, want %d", idx, len(cv.Data), len(expected[idx].cv.Data))
				errors.Add(1)
				return
			}
			for j := range cv.Data {
				if cv.Data[j] != expected[idx].cv.Data[j] {
					t.Errorf("[%d] Data[%d] mismatch", idx, j)
					errors.Add(1)
					return
				}
			}

			rt, err := q8.Dequantize(cv)
			if err != nil {
				t.Errorf("[%d] Dequantize: %v", idx, err)
				errors.Add(1)
				return
			}
			for j := range rt {
				if rt[j] != expected[idx].rt[j] {
					t.Errorf("[%d] rt[%d] mismatch: got %f, want %f", idx, j, rt[j], expected[idx].rt[j])
					errors.Add(1)
					return
				}
			}
		}(i)
	}

	wg.Wait()
	if n := errors.Load(); n > 0 {
		t.Fatalf("%d goroutines reported errors", n)
	}
}

// TestConcurrentQuantize4Bit tests 4-bit quantizer under concurrent access.
func TestConcurrentQuantize4Bit(t *testing.T) {
	const dim = 64
	const goroutines = 50
	const minVal, maxVal = -1.0, 1.0

	q4, err := NewUniformQuantizer(minVal, maxVal, 4)
	if err != nil {
		t.Fatalf("NewUniformQuantizer 4-bit: %v", err)
	}

	rng := rand.New(rand.NewSource(77))
	vecs := make([][]float64, goroutines)
	for i := range vecs {
		vecs[i] = make([]float64, dim)
		for j := range vecs[i] {
			vecs[i][j] = minVal + rng.Float64()*(maxVal-minVal)
		}
	}

	var wg sync.WaitGroup
	var errors atomic.Int64

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			cv, err := q4.Quantize(vecs[idx])
			if err != nil {
				t.Errorf("[%d] Quantize: %v", idx, err)
				errors.Add(1)
				return
			}

			rt, err := q4.Dequantize(cv)
			if err != nil {
				t.Errorf("[%d] Dequantize: %v", idx, err)
				errors.Add(1)
				return
			}

			maxErr := q4.MaxError()
			for j := range rt {
				diff := rt[j] - vecs[idx][j]
				if diff < 0 {
					diff = -diff
				}
				// Clamped values may have larger apparent error, so only check
				// values within the original range.
				if vecs[idx][j] >= minVal && vecs[idx][j] <= maxVal && diff > maxErr+1e-12 {
					t.Errorf("[%d] round-trip error at %d: |%f| > MaxError %f",
						idx, j, diff, maxErr)
					errors.Add(1)
					return
				}
			}
		}(i)
	}

	wg.Wait()
	if n := errors.Load(); n > 0 {
		t.Fatalf("%d goroutines reported errors", n)
	}
}
