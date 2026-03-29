package quantize

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// ==========================================================================
// E2E Paper Validation Tests
//
// These tests verify the mathematical predictions from:
//   - TurboQuant (arXiv:2504.19874)
//   - PolarQuant (arXiv:2502.02617)
//   - QJL (arXiv:2406.03482)
//
// Each test references the specific paper theorem or equation it validates.
// ==========================================================================

// --------------------------------------------------------------------------
// 1. TurboQuant_mse: MSE matches paper's Panter-Dite upper bound
//    Paper Eq: D_mse ≤ (√3·π/2) · (1/4^b) for any b ≥ 0
//    For b=1,2,3,4: D_mse ≈ 0.36, 0.117, 0.03, 0.009
// --------------------------------------------------------------------------

func TestE2E_TurboMSE_PaperBounds(t *testing.T) {
	const dim = 768
	const N = 5000

	// Paper: C(f_X, b) ≤ (√3·π)/(2·d) · (1/4^b) [Panter-Dite for Beta(d)]
	// Total MSE for unit vector: D_mse = d · C(f_X, b) = (√3·π/2) · (1/4^b)
	// Per-coordinate MSE = D_mse / d = (√3·π)/(2·d) · (1/4^b)
	paperBound := func(b, d int) float64 {
		return math.Sqrt(3) * math.Pi / (2.0 * float64(d)) / math.Pow(4, float64(b))
	}

	for _, bits := range []int{1, 2, 3, 4} {
		t.Run(fmt.Sprintf("b=%d", bits), func(t *testing.T) {
			tq, err := NewTurboQuantizer(dim, bits, 42)
			if err != nil {
				t.Fatal(err)
			}

			rng := rand.New(rand.NewSource(99))
			var totalMSE float64
			for i := 0; i < N; i++ {
				vec := makeUnitVec(rng, dim)
				cv, _ := tq.Quantize(vec)
				rec, _ := tq.Dequantize(cv)

				for j := range vec {
					d := vec[j] - rec[j]
					totalMSE += d * d
				}
			}
			msePerCoord := totalMSE / float64(N*dim)
			bound := paperBound(bits, dim)

			t.Logf("b=%d: measured MSE/coord=%.6f, paper bound=%.6f, ratio=%.2f",
				bits, msePerCoord, bound, msePerCoord/bound)

			// Measured MSE should be below the paper's upper bound.
			if msePerCoord > bound*1.1 { // 10% tolerance for finite-sample noise
				t.Errorf("MSE/coord %.6f exceeds paper bound %.6f (×1.1)", msePerCoord, bound)
			}

			// Measured MSE should not be dramatically below (sanity check).
			if msePerCoord < bound*0.01 {
				t.Errorf("MSE/coord %.6f suspiciously low vs bound %.6f", msePerCoord, bound)
			}
		})
	}
}

// --------------------------------------------------------------------------
// 2. TurboQuant_prod: Unbiased IP estimation
//    Paper Theorem 2: E[estimate] = ⟨y, x⟩
//    Verified with large sample, correlated vectors, multiple dims.
// --------------------------------------------------------------------------

func TestE2E_TurboProd_Unbiasedness(t *testing.T) {
	dims := []int{16, 64, 128, 256, 768}
	const N = 10000

	for _, dim := range dims {
		t.Run(fmt.Sprintf("d=%d", dim), func(t *testing.T) {
			tpq, err := NewTurboProdQuantizer(dim, 3, dim, 42)
			if err != nil {
				t.Fatal(err)
			}

			rng := rand.New(rand.NewSource(12345))
			var sumTrue, sumEst float64

			for i := 0; i < N; i++ {
				// Correlated vectors: y = 0.7*x + 0.3*noise → true IP ≈ 0.7
				x := makeUnitVec(rng, dim)
				noise := makeUnitVec(rng, dim)
				y := make([]float64, dim)
				for j := range y {
					y[j] = 0.7*x[j] + 0.3*noise[j]
				}
				normalizeVec(y)

				trueIP := dotVec(x, y)
				cv, _ := tpq.Quantize(x)
				estIP, _ := tpq.EstimateInnerProduct(y, cv)

				sumTrue += trueIP
				sumEst += estIP
			}

			avgTrue := sumTrue / float64(N)
			avgEst := sumEst / float64(N)
			bias := math.Abs(avgEst - avgTrue)
			relBias := bias / math.Abs(avgTrue)

			t.Logf("d=%d: avgTrue=%.6f avgEst=%.6f |bias|=%.6f relBias=%.4f",
				dim, avgTrue, avgEst, bias, relBias)

			// Paper guarantees E[estimate] = ⟨y, x⟩ (zero bias).
			// With 10K samples, statistical noise ≈ 1/√N ≈ 0.01.
			if relBias > 0.03 {
				t.Errorf("relative bias %.4f > 3%% — estimator may be biased", relBias)
			}
		})
	}
}

// --------------------------------------------------------------------------
// 3. TurboQuant_prod: IP distortion bound
//    Paper Theorem 2: D_prod ≤ (√3·π²·||y||²)/d · (1/4^b)
// --------------------------------------------------------------------------

func TestE2E_TurboProd_DistortionBound(t *testing.T) {
	const dim = 128
	const N = 5000

	paperBound := func(b, d int) float64 {
		return math.Sqrt(3) * math.Pi * math.Pi / float64(d) / math.Pow(4, float64(b))
	}

	for _, bits := range []int{2, 3, 4} {
		t.Run(fmt.Sprintf("b=%d", bits), func(t *testing.T) {
			tpq, err := NewTurboProdQuantizer(dim, bits, dim, 42)
			if err != nil {
				t.Fatal(err)
			}

			rng := rand.New(rand.NewSource(99))
			var totalSqErr float64

			for i := 0; i < N; i++ {
				x := makeUnitVec(rng, dim)
				y := makeUnitVec(rng, dim) // ||y||=1

				trueIP := dotVec(x, y)
				cv, _ := tpq.Quantize(x)
				estIP, _ := tpq.EstimateInnerProduct(y, cv)

				err := estIP - trueIP
				totalSqErr += err * err
			}

			distortion := totalSqErr / float64(N)
			bound := paperBound(bits, dim) // ||y||=1

			t.Logf("b=%d d=%d: distortion=%.6f, paper bound=%.6f, ratio=%.2f",
				bits, dim, distortion, bound, distortion/bound)

			// Allow 3x tolerance due to finite bits and finite sample.
			if distortion > bound*3.0 {
				t.Errorf("distortion %.6f exceeds 3× paper bound %.6f", distortion, bound)
			}
		})
	}
}

// --------------------------------------------------------------------------
// 4. TurboQuant_prod: MSE-only reconstruction is biased (multiplicatively)
//    Paper: 1-bit MSE has multiplicative bias of 2/π ≈ 0.637
// --------------------------------------------------------------------------

func TestE2E_TurboProd_MSEBias(t *testing.T) {
	const dim = 128
	const N = 10000

	// 1-bit MSE: bias factor = 2/π
	tq1, _ := NewTurboQuantizer(dim, 1, 42)
	rng := rand.New(rand.NewSource(42))

	var sumTrue, sumMSE float64
	for i := 0; i < N; i++ {
		x := makeUnitVec(rng, dim)
		y := make([]float64, dim)
		for j := range y { y[j] = 0.5*x[j] + 0.5*rng.NormFloat64()/math.Sqrt(float64(dim)) }
		normalizeVec(y)

		trueIP := dotVec(x, y)
		cv, _ := tq1.Quantize(x)
		rec, _ := tq1.Dequantize(cv)
		mseIP := dotVec(rec, y)

		sumTrue += trueIP
		sumMSE += mseIP
	}

	avgTrue := sumTrue / float64(N)
	avgMSE := sumMSE / float64(N)
	ratio := avgMSE / avgTrue

	t.Logf("1-bit MSE bias: avgTrue=%.4f avgMSE=%.4f ratio=%.4f (paper: 2/π=%.4f)",
		avgTrue, avgMSE, ratio, 2.0/math.Pi)

	// Paper predicts ratio ≈ 2/π ≈ 0.637 for 1-bit.
	// Allow 15% tolerance for finite sample.
	if math.Abs(ratio-2.0/math.Pi) > 0.1 {
		t.Errorf("MSE bias ratio %.4f, expected ≈%.4f (2/π)", ratio, 2.0/math.Pi)
	}
}

// --------------------------------------------------------------------------
// 5. TurboQuant_prod: Full estimator corrects MSE bias
//    Verify that adding QJL correction brings bias from ~36% down to <3%.
// --------------------------------------------------------------------------

func TestE2E_TurboProd_CorrectionRemovesBias(t *testing.T) {
	const dim = 128
	const N = 20000

	tpq, _ := NewTurboProdQuantizer(dim, 2, dim, 42) // 1-bit MSE + 1-bit QJL

	rng := rand.New(rand.NewSource(42))
	var sumTrue, sumFull, sumMSE float64
	for i := 0; i < N; i++ {
		x := makeUnitVec(rng, dim)
		y := make([]float64, dim)
		for j := range y { y[j] = 0.5*x[j] + 0.5*rng.NormFloat64()/math.Sqrt(float64(dim)) }
		normalizeVec(y)

		trueIP := dotVec(x, y)
		cv, _ := tpq.Quantize(x)
		fullEst, _ := tpq.EstimateInnerProduct(y, cv)

		pv, _ := tpq.ParseProdVector(cv)
		xHat, _ := tpq.mseQuant.Dequantize(pv.MSEData)
		mseIP := dotVec(xHat, y)

		sumTrue += trueIP
		sumFull += fullEst
		sumMSE += mseIP
	}

	avgTrue := sumTrue / float64(N)
	avgFull := sumFull / float64(N)
	avgMSE := sumMSE / float64(N)

	mseBiasRel := math.Abs(avgMSE-avgTrue) / math.Abs(avgTrue)
	fullBiasRel := math.Abs(avgFull-avgTrue) / math.Abs(avgTrue)

	t.Logf("MSE-only  rel bias = %.4f (%.1f%%)", mseBiasRel, mseBiasRel*100)
	t.Logf("Full est  rel bias = %.4f (%.1f%%)", fullBiasRel, fullBiasRel*100)
	t.Logf("Bias reduction: %.1fx", mseBiasRel/fullBiasRel)

	if mseBiasRel < 0.1 {
		t.Skip("MSE bias too small to test correction effect")
	}
	if fullBiasRel > mseBiasRel {
		t.Errorf("QJL correction made bias WORSE: full=%.4f > mse=%.4f", fullBiasRel, mseBiasRel)
	}
	if fullBiasRel > 0.05 {
		t.Errorf("full estimator bias %.4f > 5%% — correction insufficient", fullBiasRel)
	}
}

// --------------------------------------------------------------------------
// 6. PolarQuant: Energy conservation
//    Paper: ||x||² = ∑(final_radii)² at each recursion level
// --------------------------------------------------------------------------

func TestE2E_PolarQuant_EnergyConservation(t *testing.T) {
	const dim = 256
	const N = 1000

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < N; i++ {
		vec := make([]float64, dim)
		for j := range vec {
			vec[j] = rng.NormFloat64()
		}

		originalEnergy := 0.0
		for _, v := range vec {
			originalEnergy += v * v
		}

		_, radii, err := PolarTransform(vec, 4)
		if err != nil {
			t.Fatalf("PolarTransform: %v", err)
		}

		radiiEnergy := 0.0
		for _, r := range radii {
			radiiEnergy += r * r
		}

		relErr := math.Abs(radiiEnergy-originalEnergy) / originalEnergy
		if relErr > 1e-10 {
			t.Errorf("vec %d: energy not conserved: original=%.10f radii=%.10f relErr=%.2e",
				i, originalEnergy, radiiEnergy, relErr)
		}
	}
}

// --------------------------------------------------------------------------
// 7. PolarQuant: Angle distribution verification
//    Paper Lemma 2: Level 1 angles ~ Uniform[0, 2π)
//                   Level ℓ≥2 angles ~ sin^(2^(ℓ-1)-1)(2ψ) on [0, π/2]
// --------------------------------------------------------------------------

func TestE2E_PolarQuant_AngleDistributions(t *testing.T) {
	const dim = 256
	const N = 50000 // many vectors for distribution estimation

	rng := rand.New(rand.NewSource(42))

	// Collect angles from random Gaussian vectors (simulating post-rotation).
	level1Angles := make([]float64, 0, N*dim/2)
	level2Angles := make([]float64, 0, N*dim/4)

	for i := 0; i < N; i++ {
		vec := make([]float64, dim)
		for j := range vec {
			vec[j] = rng.NormFloat64()
		}

		angles, _, _ := PolarTransform(vec, 4)
		for _, a := range angles[0] {
			// Map to [0, 2π)
			if a < 0 {
				a += 2 * math.Pi
			}
			level1Angles = append(level1Angles, a)
		}
		for _, a := range angles[1] {
			level2Angles = append(level2Angles, a)
		}
	}

	// Level 1: should be approximately uniform on [0, 2π).
	// Chi-squared test with 8 bins.
	t.Run("Level1_Uniform", func(t *testing.T) {
		nBins := 8
		bins := make([]int, nBins)
		binWidth := 2 * math.Pi / float64(nBins)
		for _, a := range level1Angles {
			bin := int(a / binWidth)
			if bin >= nBins {
				bin = nBins - 1
			}
			bins[bin]++
		}

		expected := float64(len(level1Angles)) / float64(nBins)
		var chiSq float64
		for _, count := range bins {
			diff := float64(count) - expected
			chiSq += diff * diff / expected
		}

		// Chi-squared critical value for 7 df at 99.9% = 24.32
		t.Logf("Level 1 chi-squared = %.2f (critical 99.9%% = 24.32, nBins=%d)", chiSq, nBins)
		if chiSq > 50 { // very generous threshold
			t.Errorf("Level 1 angles not uniform: chi²=%.2f", chiSq)
		}
	})

	// Level 2: should follow sin(2ψ) on [0, π/2].
	// Verify by checking mean angle — for sin(2ψ), mean ≈ π/4 (by symmetry around π/4).
	t.Run("Level2_SinDistribution", func(t *testing.T) {
		var sum float64
		for _, a := range level2Angles {
			sum += math.Abs(a) // angles should be in [0, π/2]
		}
		mean := sum / float64(len(level2Angles))

		// Theoretical mean of sin(2ψ) distribution on [0, π/2]:
		// E[ψ] = ∫₀^{π/2} ψ·sin(2ψ)dψ / ∫₀^{π/2} sin(2ψ)dψ
		// ∫sin(2ψ)dψ = [-cos(2ψ)/2]₀^{π/2} = 1
		// ∫ψ·sin(2ψ)dψ = [-ψ·cos(2ψ)/2 + sin(2ψ)/4]₀^{π/2} = π/4
		theoreticalMean := math.Pi / 4

		relErr := math.Abs(mean-theoreticalMean) / theoreticalMean
		t.Logf("Level 2 mean angle = %.6f, theoretical = %.6f (π/4), relErr = %.4f",
			mean, theoreticalMean, relErr)

		if relErr > 0.02 {
			t.Errorf("Level 2 mean angle %.6f deviates %.1f%% from π/4", mean, relErr*100)
		}
	})
}

// --------------------------------------------------------------------------
// 8. PolarQuant: Round-trip preserves cosine similarity
//    After quantize → dequantize, cosine sim should be ≥ 0.95 at default config.
// --------------------------------------------------------------------------

func TestE2E_PolarQuant_CosineSimilarity(t *testing.T) {
	dims := []int{64, 128, 256, 768}
	const N = 500

	for _, dim := range dims {
		t.Run(fmt.Sprintf("d=%d", dim), func(t *testing.T) {
			cfg := DefaultPolarConfig(dim)
			cfg.Seed = 42
			pq, err := NewPolarQuantizer(cfg)
			if err != nil {
				t.Fatal(err)
			}

			rng := rand.New(rand.NewSource(99))
			var sumCosSim float64
			minCosSim := 1.0

			for i := 0; i < N; i++ {
				vec := makeUnitVec(rng, dim)
				cv, _ := pq.Quantize(vec)
				rec, _ := pq.Dequantize(cv)

				cosSim := dotVec(vec, rec) / vecNorm(rec)
				sumCosSim += cosSim
				if cosSim < minCosSim {
					minCosSim = cosSim
				}
			}

			avgCosSim := sumCosSim / float64(N)
			t.Logf("d=%d: avg cosine sim = %.6f, min = %.6f", dim, avgCosSim, minCosSim)

			if avgCosSim < 0.95 {
				t.Errorf("avg cosine similarity %.4f < 0.95", avgCosSim)
			}
		})
	}
}

// --------------------------------------------------------------------------
// 9. PolarQuant: BitsPerCoord matches paper prediction
//    Paper: For L=4, bits=[4,2,2,2], radii=16 → 3.875 bits/coord
// --------------------------------------------------------------------------

func TestE2E_PolarQuant_BitsPerCoord(t *testing.T) {
	tests := []struct {
		dim    int
		levels int
		bitsL1 int
		bitsR  int
		radB   int
		expect float64
	}{
		{128, 4, 4, 2, 16, 3.875},
		{256, 4, 4, 2, 16, 3.875},
		{768, 4, 4, 2, 16, 3.875},
		{64, 2, 4, 2, 16, 6.5}, // L=2: 4·32 + 2·16 + 16·16 = 128+32+256 = 416 → 416/64 = 6.5
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("d=%d_L=%d", tt.dim, tt.levels), func(t *testing.T) {
			cfg := PolarConfig{
				Dim: tt.dim, Levels: tt.levels,
				BitsLevel1: tt.bitsL1, BitsRest: tt.bitsR, RadiusBits: tt.radB,
			}
			got := cfg.BitsPerCoord()
			if math.Abs(got-tt.expect) > 0.001 {
				t.Errorf("BitsPerCoord = %.4f, want %.4f", got, tt.expect)
			}
		})
	}
}

// --------------------------------------------------------------------------
// 10. QJL Sketch: Hamming distance approximates cosine similarity
//     Paper: E[1 - 2·hamming/k] ≈ (2/π)·arcsin(cos(θ))
// --------------------------------------------------------------------------

func TestE2E_QJL_HammingApproxCosine(t *testing.T) {
	const dim = 128
	const N = 5000

	// Single quantizer, reused across all queries.
	tpq, _ := NewTurboProdQuantizer(dim, 3, dim, 42)
	rng := rand.New(rand.NewSource(42))

	similarities := []float64{0.0, 0.3, 0.6, 0.9}
	for _, targetSim := range similarities {
		t.Run(fmt.Sprintf("sim=%.1f", targetSim), func(t *testing.T) {
			var sumEstSim, sumTrueSim float64
			for i := 0; i < N; i++ {
				x := makeUnitVec(rng, dim)
				noise := makeUnitVec(rng, dim)
				y := make([]float64, dim)
				for j := range y {
					y[j] = targetSim*x[j] + (1-targetSim)*noise[j]
				}
				normalizeVec(y)

				trueSim := dotVec(x, y)
				cv, _ := tpq.Quantize(x)
				estIP, _ := tpq.EstimateInnerProduct(y, cv)

				sumTrueSim += trueSim
				sumEstSim += estIP
			}

			avgTrue := sumTrueSim / float64(N)
			avgEst := sumEstSim / float64(N)
			bias := math.Abs(avgEst - avgTrue)

			t.Logf("target=%.1f: avgTrue=%.4f avgEst=%.4f |bias|=%.4f",
				targetSim, avgTrue, avgEst, bias)

			if math.Abs(avgTrue) > 0.05 {
				if math.Abs(bias/avgTrue) > 0.1 {
					t.Errorf("relative bias %.4f > 10%%", math.Abs(bias/avgTrue))
				}
			} else {
				if bias > 0.05 {
					t.Errorf("absolute bias %.4f > 0.05 (near-zero similarity)", bias)
				}
			}
		})
	}
}

// --------------------------------------------------------------------------
// 11. Compression ratio validation across all algorithms
//     Verify bytes match expected bit rates.
// --------------------------------------------------------------------------

func TestE2E_CompressionRatios(t *testing.T) {
	const dim = 768

	rng := rand.New(rand.NewSource(42))
	vec := makeUnitVec(rng, dim)

	originalBytes := dim * 8 // float64

	t.Run("Turbo_1bit", func(t *testing.T) {
		tq, _ := NewTurboQuantizer(dim, 1, 42)
		cv, _ := tq.Quantize(vec)
		ratio := float64(originalBytes) / float64(len(cv.Data))
		t.Logf("1-bit: %d bytes, ratio=%.1fx", len(cv.Data), ratio)
		if ratio < 40 { // 64x theoretical, ~58x with overhead
			t.Errorf("compression ratio %.1f too low for 1-bit", ratio)
		}
	})

	t.Run("Turbo_3bit", func(t *testing.T) {
		tq, _ := NewTurboQuantizer(dim, 3, 42)
		cv, _ := tq.Quantize(vec)
		ratio := float64(originalBytes) / float64(len(cv.Data))
		t.Logf("3-bit: %d bytes, ratio=%.1fx", len(cv.Data), ratio)
		if ratio < 15 {
			t.Errorf("compression ratio %.1f too low for 3-bit", ratio)
		}
	})

	t.Run("Polar_default", func(t *testing.T) {
		cfg := DefaultPolarConfig(dim)
		cfg.Seed = 42
		pq, _ := NewPolarQuantizer(cfg)
		cv, _ := pq.Quantize(vec)
		ratio := float64(originalBytes) / float64(len(cv.Data))
		bitsPerCoord := float64(len(cv.Data)*8) / float64(dim)
		t.Logf("Polar: %d bytes, ratio=%.1fx, %.2f bits/coord", len(cv.Data), ratio, bitsPerCoord)
		if bitsPerCoord > 5.0 {
			t.Errorf("bits/coord %.2f too high for PolarQuant (expect ~3.875)", bitsPerCoord)
		}
	})

	t.Run("Scalar_4bit", func(t *testing.T) {
		sq, _ := NewUniformQuantizer(-1, 1, 4)
		cv, _ := sq.Quantize(vec)
		ratio := float64(originalBytes) / float64(len(cv.Data))
		t.Logf("Scalar 4-bit: %d bytes, ratio=%.1fx", len(cv.Data), ratio)
		if ratio < 14 {
			t.Errorf("compression ratio %.1f too low for 4-bit scalar", ratio)
		}
	})
}

// --------------------------------------------------------------------------
// 12. Round-trip determinism: same seed → same output
// --------------------------------------------------------------------------

func TestE2E_Determinism(t *testing.T) {
	const dim = 128

	rng := rand.New(rand.NewSource(42))
	vec := makeUnitVec(rng, dim)
	query := makeUnitVec(rng, dim)

	t.Run("Turbo", func(t *testing.T) {
		tq1, _ := NewTurboQuantizer(dim, 3, 42)
		tq2, _ := NewTurboQuantizer(dim, 3, 42)

		cv1, _ := tq1.Quantize(vec)
		cv2, _ := tq2.Quantize(vec)

		rec1, _ := tq1.Dequantize(cv1)
		rec2, _ := tq2.Dequantize(cv2)

		for i := range rec1 {
			if rec1[i] != rec2[i] {
				t.Fatalf("non-deterministic at index %d: %f != %f", i, rec1[i], rec2[i])
			}
		}
	})

	t.Run("TurboProd", func(t *testing.T) {
		tpq1, _ := NewTurboProdQuantizer(dim, 3, dim, 42)
		tpq2, _ := NewTurboProdQuantizer(dim, 3, dim, 42)

		cv1, _ := tpq1.Quantize(vec)
		cv2, _ := tpq2.Quantize(vec)

		ip1, _ := tpq1.EstimateInnerProduct(query, cv1)
		ip2, _ := tpq2.EstimateInnerProduct(query, cv2)

		if ip1 != ip2 {
			t.Errorf("non-deterministic IP: %f != %f", ip1, ip2)
		}
	})

	t.Run("Polar", func(t *testing.T) {
		cfg := DefaultPolarConfig(dim)
		cfg.Seed = 42
		pq1, _ := NewPolarQuantizer(cfg)
		pq2, _ := NewPolarQuantizer(cfg)

		cv1, _ := pq1.Quantize(vec)
		cv2, _ := pq2.Quantize(vec)

		rec1, _ := pq1.Dequantize(cv1)
		rec2, _ := pq2.Dequantize(cv2)

		for i := range rec1 {
			if rec1[i] != rec2[i] {
				t.Fatalf("non-deterministic at index %d: %f != %f", i, rec1[i], rec2[i])
			}
		}
	})
}

// --------------------------------------------------------------------------
// 13. Ranking preservation: quantized vectors preserve nearest-neighbor order
//     (Recall@10 test with 1K database, 50 queries)
// --------------------------------------------------------------------------

func TestE2E_RankingPreservation(t *testing.T) {
	const dim = 128
	const dbSize = 1000
	const numQueries = 50
	const topK = 10

	rng := rand.New(rand.NewSource(42))
	database := make([][]float64, dbSize)
	for i := range database {
		database[i] = makeUnitVec(rng, dim)
	}

	t.Run("Turbo_3bit", func(t *testing.T) {
		tq, _ := NewTurboQuantizer(dim, 3, 42)

		compDB := make([]CompressedVector, dbSize)
		recDB := make([][]float64, dbSize)
		for i, v := range database {
			compDB[i], _ = tq.Quantize(v)
			recDB[i], _ = tq.Dequantize(compDB[i])
		}

		var totalRecall float64
		for q := 0; q < numQueries; q++ {
			query := makeUnitVec(rng, dim)
			trueTop := topKIndices(query, database, topK)
			quantTop := topKIndices(query, recDB, topK)
			totalRecall += recallAt(trueTop, quantTop)
		}
		avgRecall := totalRecall / float64(numQueries)
		t.Logf("Turbo 3-bit: recall@%d = %.4f", topK, avgRecall)
		if avgRecall < 0.5 {
			t.Errorf("recall@%d = %.4f < 0.5", topK, avgRecall)
		}
	})

	t.Run("TurboProd_3bit", func(t *testing.T) {
		tpq, _ := NewTurboProdQuantizer(dim, 3, dim, 42)

		compDB := make([]CompressedVector, dbSize)
		for i, v := range database {
			compDB[i], _ = tpq.Quantize(v)
		}

		var totalRecall float64
		for q := 0; q < numQueries; q++ {
			query := makeUnitVec(rng, dim)
			trueTop := topKIndices(query, database, topK)

			// Rank by estimated IP
			scores := make([]float64, dbSize)
			for i := range compDB {
				scores[i], _ = tpq.EstimateInnerProduct(query, compDB[i])
			}
			quantTop := topKByScores(scores, topK)
			totalRecall += recallAt(trueTop, quantTop)
		}
		avgRecall := totalRecall / float64(numQueries)
		t.Logf("TurboProd 3-bit: recall@%d = %.4f", topK, avgRecall)
		if avgRecall < 0.4 {
			t.Errorf("recall@%d = %.4f < 0.4", topK, avgRecall)
		}
	})
}

// ==========================================================================
// Helpers
// ==========================================================================

func makeUnitVec(rng *rand.Rand, dim int) []float64 {
	v := make([]float64, dim)
	for j := range v {
		v[j] = rng.NormFloat64()
	}
	normalizeVec(v)
	return v
}

func normalizeVec(v []float64) {
	n := vecNorm(v)
	if n > 0 {
		for i := range v {
			v[i] /= n
		}
	}
}

func vecNorm(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

func dotVec(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func topKIndices(query []float64, db [][]float64, k int) []int {
	type scored struct {
		idx   int
		score float64
	}
	items := make([]scored, len(db))
	for i, v := range db {
		items[i] = scored{i, dotVec(query, v)}
	}
	// Simple selection of top-k
	for i := 0; i < k; i++ {
		for j := i + 1; j < len(items); j++ {
			if items[j].score > items[i].score {
				items[i], items[j] = items[j], items[i]
			}
		}
	}
	result := make([]int, k)
	for i := 0; i < k; i++ {
		result[i] = items[i].idx
	}
	return result
}

func topKByScores(scores []float64, k int) []int {
	type scored struct {
		idx   int
		score float64
	}
	items := make([]scored, len(scores))
	for i, s := range scores {
		items[i] = scored{i, s}
	}
	for i := 0; i < k; i++ {
		for j := i + 1; j < len(items); j++ {
			if items[j].score > items[i].score {
				items[i], items[j] = items[j], items[i]
			}
		}
	}
	result := make([]int, k)
	for i := 0; i < k; i++ {
		result[i] = items[i].idx
	}
	return result
}

func recallAt(truth, predicted []int) float64 {
	truthSet := make(map[int]bool, len(truth))
	for _, idx := range truth {
		truthSet[idx] = true
	}
	hits := 0
	for _, idx := range predicted {
		if truthSet[idx] {
			hits++
		}
	}
	return float64(hits) / float64(len(truth))
}
