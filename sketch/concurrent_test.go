package sketch

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
)

// TestConcurrentSketchSameVector verifies that a single QJLSketcher is safe
// when 100 goroutines all call Sketch with the exact same input vector.
// The race detector will flag any data races, and we verify all results
// are identical.
func TestConcurrentSketchSameVector(t *testing.T) {
	const dim = 64
	const sketchDim = 32
	const goroutines = 100

	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:       dim,
		SketchDim: sketchDim,
		Seed:      42,
	})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	// Build a deterministic input vector.
	rng := rand.New(rand.NewSource(7))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	// Compute the expected result.
	expected, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch reference: %v", err)
	}

	var wg sync.WaitGroup
	var errors atomic.Int64

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			bv, err := sketcher.Sketch(vec)
			if err != nil {
				t.Errorf("Sketch: %v", err)
				errors.Add(1)
				return
			}
			if bv.Dim != expected.Dim {
				t.Errorf("Dim mismatch: got %d, want %d", bv.Dim, expected.Dim)
				errors.Add(1)
				return
			}
			if len(bv.Bits) != len(expected.Bits) {
				t.Errorf("Bits length mismatch: got %d, want %d", len(bv.Bits), len(expected.Bits))
				errors.Add(1)
				return
			}
			for j := range expected.Bits {
				if bv.Bits[j] != expected.Bits[j] {
					t.Errorf("Bits[%d] mismatch: got 0x%X, want 0x%X", j, bv.Bits[j], expected.Bits[j])
					errors.Add(1)
					return
				}
			}
		}()
	}

	wg.Wait()
	if n := errors.Load(); n > 0 {
		t.Fatalf("%d goroutines reported errors", n)
	}
}

// TestConcurrentHamming verifies that HammingDistance and EstimateInnerProduct
// are safe to call concurrently from many goroutines on shared BitVectors.
func TestConcurrentHamming(t *testing.T) {
	const dim = 64
	const sketchDim = 32
	const goroutines = 50

	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:       dim,
		SketchDim: sketchDim,
		Seed:      13,
	})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	rng := rand.New(rand.NewSource(1))
	vecA := make([]float64, dim)
	vecB := make([]float64, dim)
	for i := 0; i < dim; i++ {
		vecA[i] = rng.NormFloat64()
		vecB[i] = rng.NormFloat64()
	}

	bvA, err := sketcher.Sketch(vecA)
	if err != nil {
		t.Fatalf("Sketch A: %v", err)
	}
	bvB, err := sketcher.Sketch(vecB)
	if err != nil {
		t.Fatalf("Sketch B: %v", err)
	}

	// Compute expected results once.
	expectedDist := HammingDistance(*bvA, *bvB)
	expectedIP, err := EstimateInnerProduct(*bvA, *bvB)
	if err != nil {
		t.Fatalf("EstimateInnerProduct: %v", err)
	}

	var wg sync.WaitGroup
	var errors atomic.Int64

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(iteration int) {
			defer wg.Done()

			// Half the goroutines test HammingDistance, half test EstimateInnerProduct.
			if iteration%2 == 0 {
				d := HammingDistance(*bvA, *bvB)
				if d != expectedDist {
					t.Errorf("HammingDistance: got %d, want %d", d, expectedDist)
					errors.Add(1)
				}
			} else {
				ip, err := EstimateInnerProduct(*bvA, *bvB)
				if err != nil {
					t.Errorf("EstimateInnerProduct: %v", err)
					errors.Add(1)
					return
				}
				if ip != expectedIP {
					t.Errorf("EstimateInnerProduct: got %f, want %f", ip, expectedIP)
					errors.Add(1)
				}
			}
		}(i)
	}

	wg.Wait()
	if n := errors.Load(); n > 0 {
		t.Fatalf("%d goroutines reported errors", n)
	}
}
