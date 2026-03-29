package quantize

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

// ---------------------------------------------------------------------------
// TestNewTurboProdQuantizer
// ---------------------------------------------------------------------------

func TestNewTurboProdQuantizer(t *testing.T) {
	t.Run("valid construction", func(t *testing.T) {
		tpq, err := NewTurboProdQuantizer(128, 4, 128, 42)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if tpq.Dim() != 128 {
			t.Errorf("Dim() = %d, want 128", tpq.Dim())
		}
		if tpq.Bits() != 4 {
			t.Errorf("Bits() = %d, want 4", tpq.Bits())
		}
		if tpq.SketchDim() != 128 {
			t.Errorf("SketchDim() = %d, want 128", tpq.SketchDim())
		}
		if tpq.mseQuant == nil {
			t.Error("mseQuant is nil")
		}
		if tpq.projector == nil {
			t.Error("projector is nil")
		}
	})

	t.Run("valid with reduced sketchDim", func(t *testing.T) {
		tpq, err := NewTurboProdQuantizer(128, 3, 64, 42)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if tpq.SketchDim() != 64 {
			t.Errorf("SketchDim() = %d, want 64", tpq.SketchDim())
		}
	})

	t.Run("invalid dim", func(t *testing.T) {
		for _, dim := range []int{0, -1, 1} {
			_, err := NewTurboProdQuantizer(dim, 3, 64, 42)
			if err == nil {
				t.Errorf("dim=%d: expected error", dim)
			}
		}
	})

	t.Run("invalid bits", func(t *testing.T) {
		for _, b := range []int{0, 1, 5, -1} {
			_, err := NewTurboProdQuantizer(64, b, 64, 42)
			if err == nil {
				t.Errorf("bits=%d: expected error", b)
			}
		}
	})

	t.Run("invalid sketchDim", func(t *testing.T) {
		for _, sd := range []int{0, -1, 65} {
			_, err := NewTurboProdQuantizer(64, 3, sd, 42)
			if err == nil {
				t.Errorf("sketchDim=%d: expected error", sd)
			}
		}
	})

	t.Run("all valid bit widths", func(t *testing.T) {
		for _, b := range []int{2, 3, 4} {
			tpq, err := NewTurboProdQuantizer(64, b, 64, 42)
			if err != nil {
				t.Errorf("bits=%d: unexpected error: %v", b, err)
			}
			if tpq.Bits() != b {
				t.Errorf("bits=%d: Bits() = %d, want %d", b, tpq.Bits(), b)
			}
		}
	})
}

// ---------------------------------------------------------------------------
// TestTurboProd_QuantizeDequantize
// ---------------------------------------------------------------------------

func TestTurboProd_QuantizeDequantize(t *testing.T) {
	dim := 128
	bits := 3 // 2-bit MSE + 1-bit QJL

	tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(99))
	vecs := generateUnitVectors(rng, 20, dim)

	for i, vec := range vecs {
		cv, err := tpq.Quantize(vec)
		if err != nil {
			t.Fatalf("Quantize vec %d: %v", i, err)
		}
		if cv.Dim != dim {
			t.Errorf("vec %d: cv.Dim = %d, want %d", i, cv.Dim, dim)
		}
		if cv.BitsPer != bits {
			t.Errorf("vec %d: cv.BitsPer = %d, want %d", i, cv.BitsPer, bits)
		}
		if len(cv.Data) == 0 {
			t.Errorf("vec %d: cv.Data is empty", i)
		}

		rec, err := tpq.Dequantize(cv)
		if err != nil {
			t.Fatalf("Dequantize vec %d: %v", i, err)
		}
		if len(rec) != dim {
			t.Errorf("vec %d: len(rec) = %d, want %d", i, len(rec), dim)
		}

		// Dequantize returns MSE reconstruction (2-bit), which should be
		// close to the original unit-norm vector.
		var sqErr float64
		for j := range vec {
			d := vec[j] - rec[j]
			sqErr += d * d
		}
		mse := sqErr / float64(dim)
		// 2-bit MSE should give MSE < 0.15; allow generous margin.
		if mse > 0.2 {
			t.Errorf("vec %d: MSE = %.6f, want < 0.2", i, mse)
		}
	}
}

// ---------------------------------------------------------------------------
// TestTurboProd_ResidualComputation
// ---------------------------------------------------------------------------

func TestTurboProd_ResidualComputation(t *testing.T) {
	dim := 64
	bits := 3 // 2-bit MSE + 1-bit QJL

	tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}
	// Normalize to unit norm.
	normalizeUnit(vec)

	// Manually compute expected residual norm.
	mseCV, err := tpq.mseQuant.Quantize(vec)
	if err != nil {
		t.Fatal(err)
	}
	xHat, err := tpq.mseQuant.Dequantize(mseCV)
	if err != nil {
		t.Fatal(err)
	}

	expectedGamma := 0.0
	for i := range vec {
		d := vec[i] - xHat[i]
		expectedGamma += d * d
	}
	expectedGamma = math.Sqrt(expectedGamma)

	// Quantize through TurboProd and decode wire format.
	cv, err := tpq.Quantize(vec)
	if err != nil {
		t.Fatal(err)
	}

	pd, err := decodeProdWire(cv.Data)
	if err != nil {
		t.Fatal(err)
	}

	// Gamma should match the manually computed value.
	if math.Abs(pd.gamma-expectedGamma) > 1e-10 {
		t.Errorf("gamma = %.10f, want %.10f", pd.gamma, expectedGamma)
	}

	// Gamma should be positive for non-trivial quantization.
	if pd.gamma <= 0 {
		t.Errorf("gamma = %.6f, expected > 0", pd.gamma)
	}

	// Gamma should be bounded (< 1.0 for reasonable MSE quantization).
	if pd.gamma > 1.0 {
		t.Errorf("gamma = %.6f, expected < 1.0", pd.gamma)
	}

	// Sketch dimension in wire format should match.
	if pd.sketchDim != dim {
		t.Errorf("wire sketchDim = %d, want %d", pd.sketchDim, dim)
	}

	// Sketch bits should have expected length: (dim + 63) / 64 uint64 words.
	expectedWords := (dim + 63) / 64
	if len(pd.sketchBits) != expectedWords {
		t.Errorf("sketch bits length = %d words, want %d", len(pd.sketchBits), expectedWords)
	}
}

// ---------------------------------------------------------------------------
// TestTurboProd_IPUnbiased
// ---------------------------------------------------------------------------

func TestTurboProd_IPUnbiased(t *testing.T) {
	dim := 128
	bits := 3 // 2-bit MSE + 1-bit QJL
	sketchDim := dim

	tpq, err := NewTurboProdQuantizer(dim, bits, sketchDim, 42)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(123))
	numPairs := 1000

	var sumTrueIP float64
	var sumEstIP float64

	for i := 0; i < numPairs; i++ {
		// Generate random unit-norm vectors.
		x := generateUnitVectors(rng, 1, dim)[0]
		y := generateUnitVectors(rng, 1, dim)[0]

		// True IP.
		trueIP := 0.0
		for j := range x {
			trueIP += x[j] * y[j]
		}

		// Encode x.
		cv, err := tpq.Quantize(x)
		if err != nil {
			t.Fatalf("Quantize pair %d: %v", i, err)
		}

		// Estimate IP.
		estIP, err := tpq.EstimateInnerProduct(y, cv)
		if err != nil {
			t.Fatalf("EstimateIP pair %d: %v", i, err)
		}

		sumTrueIP += trueIP
		sumEstIP += estIP
	}

	avgTrue := sumTrueIP / float64(numPairs)
	avgEst := sumEstIP / float64(numPairs)
	bias := math.Abs(avgEst - avgTrue)

	t.Logf("avg true IP = %.6f, avg estimated IP = %.6f, bias = %.6f",
		avgTrue, avgEst, bias)

	// The estimator is theoretically unbiased: E[estimate] = true IP.
	// With 1000 random pairs and a single projection matrix, allow
	// tolerance for finite-sample noise.
	if bias > 0.02 {
		t.Errorf("bias = %.6f, want < 0.02 (estimator may not be unbiased)", bias)
	}
}

// ---------------------------------------------------------------------------
// TestTurboProd_IPAccuracy
// ---------------------------------------------------------------------------

func TestTurboProd_IPAccuracy(t *testing.T) {
	dim := 128
	bits := 4 // 3-bit MSE + 1-bit QJL

	tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(456))

	// Test with high-similarity pairs (x ≈ y).
	for trial := 0; trial < 20; trial++ {
		// Generate a base vector.
		base := generateUnitVectors(rng, 1, dim)[0]

		// Create a similar vector by adding small perturbation.
		y := make([]float64, dim)
		copy(y, base)
		for j := range y {
			y[j] += rng.NormFloat64() * 0.1
		}
		normalizeUnit(y)

		// True IP.
		trueIP := 0.0
		for j := range base {
			trueIP += base[j] * y[j]
		}

		cv, err := tpq.Quantize(base)
		if err != nil {
			t.Fatal(err)
		}

		estIP, err := tpq.EstimateInnerProduct(y, cv)
		if err != nil {
			t.Fatal(err)
		}

		errAbs := math.Abs(estIP - trueIP)
		t.Logf("trial %d: true=%.4f est=%.4f err=%.4f", trial, trueIP, estIP, errAbs)

		// For high-similarity pairs with 4 bits, error should be bounded.
		if errAbs > 0.3 {
			t.Errorf("trial %d: |error| = %.4f, want < 0.3", trial, errAbs)
		}
	}
}

// ---------------------------------------------------------------------------
// TestTurboProd_Determinism
// ---------------------------------------------------------------------------

func TestTurboProd_Determinism(t *testing.T) {
	dim := 64
	bits := 3

	tpq1, err := NewTurboProdQuantizer(dim, bits, dim, 12345)
	if err != nil {
		t.Fatal(err)
	}
	tpq2, err := NewTurboProdQuantizer(dim, bits, dim, 12345)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	cv1, err := tpq1.Quantize(vec)
	if err != nil {
		t.Fatal(err)
	}
	cv2, err := tpq2.Quantize(vec)
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

	// IP estimates should also be identical.
	query := make([]float64, dim)
	for i := range query {
		query[i] = rng.NormFloat64()
	}

	ip1, err := tpq1.EstimateInnerProduct(query, cv1)
	if err != nil {
		t.Fatal(err)
	}
	ip2, err := tpq2.EstimateInnerProduct(query, cv2)
	if err != nil {
		t.Fatal(err)
	}
	if ip1 != ip2 {
		t.Errorf("IP estimates differ: %.10f vs %.10f", ip1, ip2)
	}
}

// ---------------------------------------------------------------------------
// TestTurboProd_Concurrent
// ---------------------------------------------------------------------------

func TestTurboProd_Concurrent(t *testing.T) {
	dim := 64
	bits := 3

	tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
	if err != nil {
		t.Fatal(err)
	}

	const numGoroutines = 50

	type result struct {
		ip  float64
		err error
	}
	results := make([]result, numGoroutines)

	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			localRng := rand.New(rand.NewSource(int64(idx)*1000 + 7))
			x := make([]float64, dim)
			y := make([]float64, dim)
			for j := range x {
				x[j] = localRng.NormFloat64()
				y[j] = localRng.NormFloat64()
			}
			normalizeUnit(x)
			normalizeUnit(y)

			cv, err := tpq.Quantize(x)
			if err != nil {
				results[idx] = result{err: err}
				return
			}

			ip, err := tpq.EstimateInnerProduct(y, cv)
			if err != nil {
				results[idx] = result{err: err}
				return
			}

			// IP estimate should be in reasonable range.
			if ip < -2 || ip > 2 {
				results[idx] = result{err: fmt.Errorf("IP = %.4f out of range [-2, 2]", ip)}
				return
			}

			results[idx] = result{ip: ip}
		}(i)
	}
	wg.Wait()

	failCount := 0
	for i, r := range results {
		if r.err != nil {
			t.Errorf("goroutine %d: %v", i, r.err)
			failCount++
		}
	}
	if failCount > 0 {
		t.Errorf("%d/%d goroutines failed", failCount, numGoroutines)
	}
}

// ---------------------------------------------------------------------------
// TestTurboProd_ImplementsInterface
// ---------------------------------------------------------------------------

func TestTurboProd_ImplementsInterface(t *testing.T) {
	// Compile-time check that TurboProdQuantizer implements Quantizer.
	var _ Quantizer = (*TurboProdQuantizer)(nil)

	tpq, err := NewTurboProdQuantizer(32, 3, 32, 0)
	if err != nil {
		t.Fatal(err)
	}

	// Runtime check via interface assertion.
	var q Quantizer = tpq
	if q.Bits() != 3 {
		t.Errorf("Bits() = %d, want 3", q.Bits())
	}
}

// ---------------------------------------------------------------------------
// TestTurboProd_NaNInput
// ---------------------------------------------------------------------------

func TestTurboProd_NaNInput(t *testing.T) {
	tpq, err := NewTurboProdQuantizer(32, 3, 32, 42)
	if err != nil {
		t.Fatal(err)
	}

	vec := make([]float64, 32)
	vec[10] = math.NaN()

	_, err = tpq.Quantize(vec)
	if err == nil {
		t.Error("expected error for NaN input")
	}
	if err != ErrNaNInput {
		t.Errorf("expected ErrNaNInput, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// TestTurboProd_DimMismatch
// ---------------------------------------------------------------------------

func TestTurboProd_DimMismatch(t *testing.T) {
	tpq, err := NewTurboProdQuantizer(64, 3, 64, 42)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("wrong dimension input", func(t *testing.T) {
		_, err := tpq.Quantize(make([]float64, 32))
		if err == nil {
			t.Error("expected error for wrong dimension")
		}
	})

	t.Run("wrong dimension compressed vector", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{0}, Dim: 32, BitsPer: 3}
		_, err := tpq.Dequantize(cv)
		if err == nil {
			t.Error("expected error for wrong dimension")
		}
	})

	t.Run("wrong bits in compressed vector", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{0}, Dim: 64, BitsPer: 4}
		_, err := tpq.Dequantize(cv)
		if err == nil {
			t.Error("expected error for wrong bits")
		}
	})

	t.Run("wrong query dimension for IP", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{0}, Dim: 64, BitsPer: 3}
		_, err := tpq.EstimateInnerProduct(make([]float64, 32), cv)
		if err == nil {
			t.Error("expected error for wrong query dimension")
		}
	})
}

// ---------------------------------------------------------------------------
// TestTurboProd_ParseProdVector
// ---------------------------------------------------------------------------

func TestTurboProd_ParseProdVector(t *testing.T) {
	dim := 64
	bits := 3

	tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(7))
	vec := generateUnitVectors(rng, 1, dim)[0]

	cv, err := tpq.Quantize(vec)
	if err != nil {
		t.Fatal(err)
	}

	pv, err := tpq.ParseProdVector(cv)
	if err != nil {
		t.Fatal(err)
	}

	if pv.Dim != dim {
		t.Errorf("Dim = %d, want %d", pv.Dim, dim)
	}
	if pv.Bits != bits {
		t.Errorf("Bits = %d, want %d", pv.Bits, bits)
	}
	if pv.ResidualNorm <= 0 {
		t.Errorf("ResidualNorm = %.6f, expected > 0", pv.ResidualNorm)
	}
	if pv.MSEData.Dim != dim {
		t.Errorf("MSEData.Dim = %d, want %d", pv.MSEData.Dim, dim)
	}
	if pv.MSEData.BitsPer != bits-1 {
		t.Errorf("MSEData.BitsPer = %d, want %d", pv.MSEData.BitsPer, bits-1)
	}
	if pv.Residual.Dim != dim {
		t.Errorf("Residual.Dim = %d, want %d", pv.Residual.Dim, dim)
	}
	if len(pv.Residual.Bits) == 0 {
		t.Error("Residual.Bits is empty")
	}
}

// ---------------------------------------------------------------------------
// TestTurboProd_MultipleBitWidths
// ---------------------------------------------------------------------------

func TestTurboProd_MultipleBitWidths(t *testing.T) {
	dim := 64

	for _, bits := range []int{2, 3, 4} {
		t.Run(fmt.Sprintf("bits=%d", bits), func(t *testing.T) {
			tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
			if err != nil {
				t.Fatal(err)
			}

			rng := rand.New(rand.NewSource(99))
			vecs := generateUnitVectors(rng, 10, dim)

			var totalMSE float64
			for _, vec := range vecs {
				cv, err := tpq.Quantize(vec)
				if err != nil {
					t.Fatal(err)
				}

				rec, err := tpq.Dequantize(cv)
				if err != nil {
					t.Fatal(err)
				}

				var sqErr float64
				for j := range vec {
					d := vec[j] - rec[j]
					sqErr += d * d
				}
				totalMSE += sqErr / float64(dim)
			}
			avgMSE := totalMSE / float64(len(vecs))
			t.Logf("bits=%d: avg MSE=%.6f", bits, avgMSE)
		})
	}
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

// normalizeUnit normalizes a vector to unit length in-place.
func normalizeUnit(v []float64) {
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}
