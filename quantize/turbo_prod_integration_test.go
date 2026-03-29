package quantize

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// TurboQuant_prod integration tests — end-to-end quality verification
// ---------------------------------------------------------------------------

// TestTurboProd_IPUnbiased_LargeScale verifies that inner product estimation
// is unbiased over 5000 random unit-vector pairs. The mean bias (average
// estimated IP minus average true IP) must be < 0.01.
func TestTurboProd_IPUnbiased_LargeScale(t *testing.T) {
	const dim = 128
	const bits = 3 // 2-bit MSE + 1-bit QJL
	const numPairs = 5000

	tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(12345))

	var sumTrueIP, sumEstIP float64

	for i := 0; i < numPairs; i++ {
		x := generateUnitVectors(rng, 1, dim)[0]
		y := generateUnitVectors(rng, 1, dim)[0]

		trueIP := 0.0
		for j := range x {
			trueIP += x[j] * y[j]
		}

		cv, err := tpq.Quantize(x)
		if err != nil {
			t.Fatalf("Quantize pair %d: %v", i, err)
		}

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

	t.Logf("5000 pairs: avg true IP = %.6f, avg estimated IP = %.6f, |bias| = %.6f",
		avgTrue, avgEst, bias)

	if bias >= 0.01 {
		t.Errorf("IP bias = %.6f, want < 0.01 (estimator is not unbiased)", bias)
	}
}

// TestTurboProd_Vs_MSE_Only compares IP estimation between TurboQuant_prod
// and plain TurboQuant_mse at the same MSE quality level.
//
// TurboQuant_prod(bits=3) = 2-bit MSE + 1-bit QJL.
// TurboQuant_mse(bits=2) = 2-bit MSE only.
//
// Both use 2 bits for MSE quantization, so the reconstruction quality is
// comparable. TurboQuant_prod adds a 1-bit QJL residual sketch on top, which
// should improve IP estimation by capturing the residual direction.
func TestTurboProd_Vs_MSE_Only(t *testing.T) {
	const dim = 128
	const prodBits = 3 // 2-bit MSE + 1-bit QJL = 3 total bits
	const mseBits = 2  // 2-bit MSE only (same MSE quality)
	const numPairs = 2000

	tpq, err := NewTurboProdQuantizer(dim, prodBits, dim, 42)
	if err != nil {
		t.Fatal(err)
	}

	// MSE-only at 2 bits (same MSE part as prod).
	tq, err := NewTurboQuantizer(dim, mseBits, 42)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(9999))

	var prodTotalErr, mseTotalErr float64
	var prodSumErr, mseSumErr float64 // signed errors for bias measurement

	for i := 0; i < numPairs; i++ {
		x := generateUnitVectors(rng, 1, dim)[0]
		y := generateUnitVectors(rng, 1, dim)[0]

		trueIP := 0.0
		for j := range x {
			trueIP += x[j] * y[j]
		}

		// TurboQuant_prod: encode → EstimateIP.
		prodCV, err := tpq.Quantize(x)
		if err != nil {
			t.Fatalf("Prod Quantize pair %d: %v", i, err)
		}
		prodIP, err := tpq.EstimateInnerProduct(y, prodCV)
		if err != nil {
			t.Fatalf("Prod EstimateIP pair %d: %v", i, err)
		}

		// TurboQuant_mse: encode → dequantize → dot product.
		mseCV, err := tq.Quantize(x)
		if err != nil {
			t.Fatalf("MSE Quantize pair %d: %v", i, err)
		}
		mseRec, err := tq.Dequantize(mseCV)
		if err != nil {
			t.Fatalf("MSE Dequantize pair %d: %v", i, err)
		}
		mseIP := 0.0
		for j := range y {
			mseIP += y[j] * mseRec[j]
		}

		prodErr := prodIP - trueIP
		mseErr := mseIP - trueIP

		prodTotalErr += math.Abs(prodErr)
		mseTotalErr += math.Abs(mseErr)
		prodSumErr += prodErr
		mseSumErr += mseErr
	}

	prodAvgErr := prodTotalErr / float64(numPairs)
	mseAvgErr := mseTotalErr / float64(numPairs)
	prodBias := math.Abs(prodSumErr / float64(numPairs))
	mseBias := math.Abs(mseSumErr / float64(numPairs))

	t.Logf("TurboQuant_prod (2-bit MSE + 1-bit QJL): avg |IP error|=%.6f, |bias|=%.6f", prodAvgErr, prodBias)
	t.Logf("TurboQuant_mse  (2-bit MSE only):         avg |IP error|=%.6f, |bias|=%.6f", mseAvgErr, mseBias)

	// TurboQuant_prod should have comparable or better absolute IP error
	// since it adds QJL correction on top of the same MSE quality.
	t.Logf("Absolute error ratio (mse/prod): %.2fx", mseAvgErr/prodAvgErr)
	t.Logf("Bias ratio (mse/prod):           %.2fx", mseBias/prodBias)

	// Both should have reasonable absolute error.
	if prodAvgErr > 0.5 {
		t.Errorf("TurboQuant_prod avg |IP error| = %.6f, want < 0.5", prodAvgErr)
	}
	if mseAvgErr > 0.5 {
		t.Errorf("TurboQuant_mse avg |IP error| = %.6f, want < 0.5", mseAvgErr)
	}
}

// TestTurboProd_CompressionRatio verifies that the compressed representation
// achieves the expected effective bits per coordinate.
//
// The TurboQuant_prod wire format includes: MSE data (dim*(bits-1)/8 bytes),
// plus 1 byte version, 4 bytes sketchDim, 8 bytes gamma, 4 bytes mseDataLen,
// 4 bytes numSketchWords, and sketchDim/8 bytes for the QJL sketch.
// The total should correspond to approximately `bits` bits per coordinate.
func TestTurboProd_CompressionRatio(t *testing.T) {
	tests := []struct {
		dim  int
		bits int
	}{
		{128, 2},
		{128, 3},
		{128, 4},
		{768, 3},
		{768, 4},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("d=%d_b=%d", tt.dim, tt.bits), func(t *testing.T) {
			tpq, err := NewTurboProdQuantizer(tt.dim, tt.bits, tt.dim, 42)
			if err != nil {
				t.Fatal(err)
			}

			rng := rand.New(rand.NewSource(7))
			vec := make([]float64, tt.dim)
			for i := range vec {
				vec[i] = rng.NormFloat64()
			}

			cv, err := tpq.Quantize(vec)
			if err != nil {
				t.Fatal(err)
			}

			originalBytes := tt.dim * 8 // float64
			compressedBytes := len(cv.Data)
			ratio := float64(originalBytes) / float64(compressedBytes)

			// Compute effective bits per coordinate.
			effectiveBits := float64(compressedBytes*8) / float64(tt.dim)

			t.Logf("dim=%d bits=%d: compressed=%d bytes, original=%d bytes, ratio=%.1fx, effective=%.2f bits/coord",
				tt.dim, tt.bits, compressedBytes, originalBytes, ratio, effectiveBits)

			// The effective bits per coordinate should be roughly the configured bits
			// plus overhead from the wire format header (~17 bytes) and QJL sketch.
			// Allow generous tolerance.
			minEffective := float64(tt.bits) * 0.5
			maxEffective := float64(tt.bits) * 3.0
			if effectiveBits < minEffective || effectiveBits > maxEffective {
				t.Errorf("effective bits/coord = %.2f, expected in [%.1f, %.1f]",
					effectiveBits, minEffective, maxEffective)
			}

			// Compression ratio should be meaningful (> 2x).
			if ratio < 2.0 {
				t.Errorf("compression ratio %.1fx < 2.0x", ratio)
			}
		})
	}
}

// TestTurboProd_MultiBitWidths verifies that bits=2,3,4 all produce
// reasonable IP estimation accuracy.
func TestTurboProd_MultiBitWidths(t *testing.T) {
	const dim = 128
	const numPairs = 500

	for _, bits := range []int{2, 3, 4} {
		t.Run(fmt.Sprintf("bits=%d", bits), func(t *testing.T) {
			tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
			if err != nil {
				t.Fatal(err)
			}

			rng := rand.New(rand.NewSource(int64(bits * 1000)))

			var totalAbsErr float64
			for i := 0; i < numPairs; i++ {
				x := generateUnitVectors(rng, 1, dim)[0]
				y := generateUnitVectors(rng, 1, dim)[0]

				trueIP := 0.0
				for j := range x {
					trueIP += x[j] * y[j]
				}

				cv, err := tpq.Quantize(x)
				if err != nil {
					t.Fatalf("Quantize pair %d: %v", i, err)
				}

				estIP, err := tpq.EstimateInnerProduct(y, cv)
				if err != nil {
					t.Fatalf("EstimateIP pair %d: %v", i, err)
				}

				totalAbsErr += math.Abs(estIP - trueIP)
			}

			avgErr := totalAbsErr / float64(numPairs)
			t.Logf("bits=%d: avg |IP error| = %.6f", bits, avgErr)

			// All bit widths should produce reasonable IP estimates.
			// With 2 bits (1-bit MSE + 1-bit QJL), accuracy is limited but should
			// still be bounded. Allow generous tolerance.
			maxErr := map[int]float64{2: 0.5, 3: 0.3, 4: 0.2}[bits]
			if avgErr > maxErr {
				t.Errorf("bits=%d: avg |IP error| = %.6f, want < %.2f", bits, avgErr, maxErr)
			}
		})
	}
}

// TestTurboProd_MultiDimensions validates that TurboQuant_prod works correctly
// across common embedding dimensions.
func TestTurboProd_MultiDimensions(t *testing.T) {
	const bits = 3
	const numPairs = 200

	dims := []int{64, 128, 256, 512, 768}

	for _, dim := range dims {
		t.Run(fmt.Sprintf("d=%d", dim), func(t *testing.T) {
			tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
			if err != nil {
				t.Fatalf("NewTurboProdQuantizer(%d, %d): %v", dim, bits, err)
			}

			rng := rand.New(rand.NewSource(int64(dim)))

			var totalAbsErr float64
			var sumBias float64

			for i := 0; i < numPairs; i++ {
				x := generateUnitVectors(rng, 1, dim)[0]
				y := generateUnitVectors(rng, 1, dim)[0]

				trueIP := 0.0
				for j := range x {
					trueIP += x[j] * y[j]
				}

				cv, err := tpq.Quantize(x)
				if err != nil {
					t.Fatalf("Quantize pair %d: %v", i, err)
				}

				estIP, err := tpq.EstimateInnerProduct(y, cv)
				if err != nil {
					t.Fatalf("EstimateIP pair %d: %v", i, err)
				}

				totalAbsErr += math.Abs(estIP - trueIP)
				sumBias += estIP - trueIP
			}

			avgErr := totalAbsErr / float64(numPairs)
			bias := sumBias / float64(numPairs)

			t.Logf("dim=%d bits=%d: avg |IP error| = %.6f, bias = %.6f",
				dim, bits, avgErr, bias)

			// Bias should be small (unbiased estimator).
			if math.Abs(bias) > 0.02 {
				t.Errorf("dim=%d: bias = %.6f, want |bias| < 0.02", dim, bias)
			}

			// Average error should be reasonable.
			if avgErr > 0.4 {
				t.Errorf("dim=%d: avg |IP error| = %.6f, want < 0.4", dim, avgErr)
			}
		})
	}
}
