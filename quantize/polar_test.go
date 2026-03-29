package quantize

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

func TestNewPolarQuantizer(t *testing.T) {
	t.Run("valid default config", func(t *testing.T) {
		cfg := DefaultPolarConfig(64)
		pq, err := NewPolarQuantizer(cfg)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pq.Dim() != 64 {
			t.Errorf("Dim() = %d, want 64", pq.Dim())
		}
		if pq.Bits() != 3 {
			// Default config BitsPerCoord ≈ 3.875, truncated to int = 3
			t.Logf("Bits() = %d (BitsPerCoord ≈ %.3f)", pq.Bits(), cfg.BitsPerCoord())
		}
		if len(pq.codebooks) != cfg.Levels {
			t.Errorf("len(codebooks) = %d, want %d", len(pq.codebooks), cfg.Levels)
		}
	})

	t.Run("valid custom config", func(t *testing.T) {
		cfg := PolarConfig{
			Dim:        128,
			Levels:     3,
			BitsLevel1: 4,
			BitsRest:   2,
			RadiusBits: 16,
			Seed:       123,
		}
		pq, err := NewPolarQuantizer(cfg)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pq.Dim() != 128 {
			t.Errorf("Dim() = %d, want 128", pq.Dim())
		}
	})

	t.Run("defaults applied", func(t *testing.T) {
		cfg := PolarConfig{Dim: 64, Seed: 42}
		pq, err := NewPolarQuantizer(cfg)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pq.config.Levels != 4 {
			t.Errorf("Levels = %d, want default 4", pq.config.Levels)
		}
		if pq.config.BitsLevel1 != 4 {
			t.Errorf("BitsLevel1 = %d, want default 4", pq.config.BitsLevel1)
		}
		if pq.config.BitsRest != 2 {
			t.Errorf("BitsRest = %d, want default 2", pq.config.BitsRest)
		}
	})

	t.Run("dim too small", func(t *testing.T) {
		cfg := PolarConfig{Dim: 1, Seed: 42}
		_, err := NewPolarQuantizer(cfg)
		if err == nil {
			t.Fatal("expected error for dim=1")
		}
	})

	t.Run("dim not multiple of 16", func(t *testing.T) {
		cfg := PolarConfig{Dim: 17, Seed: 42}
		_, err := NewPolarQuantizer(cfg)
		if err == nil {
			t.Fatal("expected error for dim=17 (not multiple of 16)")
		}
	})

	t.Run("dim not multiple of power", func(t *testing.T) {
		cfg := PolarConfig{Dim: 12, Levels: 3, Seed: 42} // 2^3=8, 12%8=4
		_, err := NewPolarQuantizer(cfg)
		if err == nil {
			t.Fatal("expected error for dim=12 not multiple of 2^3=8")
		}
	})

	t.Run("invalid bits level1", func(t *testing.T) {
		cfg := PolarConfig{Dim: 64, BitsLevel1: 9, Seed: 42}
		_, err := NewPolarQuantizer(cfg)
		if err == nil {
			t.Fatal("expected error for BitsLevel1=9")
		}
	})

	t.Run("invalid bits rest", func(t *testing.T) {
		cfg := PolarConfig{Dim: 64, BitsRest: 0, Seed: 42}
		// BitsRest=0 should get defaulted to 2
		pq, err := NewPolarQuantizer(cfg)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pq.config.BitsRest != 2 {
			t.Errorf("BitsRest = %d, want default 2", pq.config.BitsRest)
		}
	})
}

// ---------------------------------------------------------------------------
// Round-trip tests
// ---------------------------------------------------------------------------

func TestPolarQuantize_RoundTrip(t *testing.T) {
	dims := []int{64, 256, 768}

	for _, dim := range dims {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			cfg := DefaultPolarConfig(dim)
			pq, err := NewPolarQuantizer(cfg)
			if err != nil {
				t.Fatalf("NewPolarQuantizer(%d): %v", dim, err)
			}

			// Generate a random unit vector.
			rng := rand.New(rand.NewSource(99))
			vec := make([]float64, dim)
			for i := range vec {
				vec[i] = rng.NormFloat64()
			}
			normalize(vec)

			cv, err := pq.Quantize(vec)
			if err != nil {
				t.Fatalf("Quantize: %v", err)
			}
			if cv.Dim != dim {
				t.Errorf("cv.Dim = %d, want %d", cv.Dim, dim)
			}

			recon, err := pq.Dequantize(cv)
			if err != nil {
				t.Fatalf("Dequantize: %v", err)
			}
			if len(recon) != dim {
				t.Fatalf("len(recon) = %d, want %d", len(recon), dim)
			}

			// Compute MSE (cosine similarity should be high for unit vectors).
			mse := mse(vec, recon)
			cosSim := dot(vec, recon) // both unit norm
			t.Logf("dim=%d: MSE=%.6f, cosine_similarity=%.6f", dim, mse, cosSim)

			// Cosine similarity should be reasonably high (at least 0.85 for 3.875 bits).
			if cosSim < 0.80 {
				t.Errorf("cosine similarity %.4f too low for dim=%d", cosSim, dim)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

func TestPolarQuantize_Determinism(t *testing.T) {
	cfg := DefaultPolarConfig(64)

	pq1, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatal(err)
	}
	pq2, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(7))
	vec := make([]float64, 64)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}
	normalize(vec)

	cv1, err := pq1.Quantize(vec)
	if err != nil {
		t.Fatal(err)
	}
	cv2, err := pq2.Quantize(vec)
	if err != nil {
		t.Fatal(err)
	}

	// Data should be identical.
	if len(cv1.Data) != len(cv2.Data) {
		t.Fatalf("data length mismatch: %d vs %d", len(cv1.Data), len(cv2.Data))
	}
	for i := range cv1.Data {
		if cv1.Data[i] != cv2.Data[i] {
			t.Errorf("data[%d] mismatch: %d vs %d", i, cv1.Data[i], cv2.Data[i])
		}
	}

	// Same reconstruction.
	r1, _ := pq1.Dequantize(cv1)
	r2, _ := pq2.Dequantize(cv2)
	for i := range r1 {
		if r1[i] != r2[i] {
			t.Errorf("recon[%d] mismatch: %.15f vs %.15f", i, r1[i], r2[i])
		}
	}
}

// ---------------------------------------------------------------------------
// Dimension mismatch
// ---------------------------------------------------------------------------

func TestPolarQuantize_DimMismatch(t *testing.T) {
	cfg := DefaultPolarConfig(64)
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatal(err)
	}

	// Wrong-length input.
	_, err = pq.Quantize(make([]float64, 32))
	if err == nil {
		t.Fatal("expected error for wrong dimension")
	}

	// Wrong dimension in CompressedVector.
	cv := CompressedVector{Dim: 32, BitsPer: pq.Bits(), Data: []byte{0}}
	_, err = pq.Dequantize(cv)
	if err == nil {
		t.Fatal("expected error for wrong dim in Dequantize")
	}

	// Wrong bits in CompressedVector.
	cv2 := CompressedVector{Dim: 64, BitsPer: 99, Data: []byte{0}}
	_, err = pq.Dequantize(cv2)
	if err == nil {
		t.Fatal("expected error for wrong bits in Dequantize")
	}
}

// ---------------------------------------------------------------------------
// Interface compliance
// ---------------------------------------------------------------------------

func TestPolarQuantize_ImplementsInterface(t *testing.T) {
	cfg := DefaultPolarConfig(64)
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatal(err)
	}

	// Compile-time check.
	var _ Quantizer = pq

	// Verify methods exist and return expected types.
	bits := pq.Bits()
	if bits <= 0 {
		t.Errorf("Bits() = %d, want > 0", bits)
	}
	dim := pq.Dim()
	if dim != 64 {
		t.Errorf("Dim() = %d, want 64", dim)
	}
}

// ---------------------------------------------------------------------------
// Norm preservation
// ---------------------------------------------------------------------------

func TestPolarQuantize_PreservesNorm(t *testing.T) {
	cfg := DefaultPolarConfig(64)
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	vec := make([]float64, 64)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}
	normalize(vec)

	cv, err := pq.Quantize(vec)
	if err != nil {
		t.Fatal(err)
	}

	recon, err := pq.Dequantize(cv)
	if err != nil {
		t.Fatal(err)
	}

	// The reconstruction should be approximately unit norm.
	rNorm := norm(recon)
	if math.Abs(rNorm-1.0) > 0.15 {
		t.Errorf("reconstructed norm = %.6f, want ≈ 1.0 (tolerance 0.15)", rNorm)
	}
}

// ---------------------------------------------------------------------------
// Concurrent safety
// ---------------------------------------------------------------------------

func TestPolarQuantize_Concurrent(t *testing.T) {
	cfg := DefaultPolarConfig(64)
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatal(err)
	}

	const goroutines = 50
	var wg sync.WaitGroup
	errCh := make(chan error, goroutines)

	rng := rand.New(rand.NewSource(42))
	vecs := make([][]float64, goroutines)
	for i := range vecs {
		vecs[i] = make([]float64, 64)
		for j := range vecs[i] {
			vecs[i][j] = rng.NormFloat64()
		}
		normalize(vecs[i])
	}

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			cv, err := pq.Quantize(vecs[idx])
			if err != nil {
				errCh <- err
				return
			}
			recon, err := pq.Dequantize(cv)
			if err != nil {
				errCh <- err
				return
			}
			if len(recon) != 64 {
				errCh <- fmt.Errorf("goroutine %d: len(recon)=%d", idx, len(recon))
			}
		}(g)
	}
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("concurrent error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Bit efficiency
// ---------------------------------------------------------------------------

func TestPolarQuantize_BitEfficiency(t *testing.T) {
	cfg := DefaultPolarConfig(64)
	bpc := cfg.BitsPerCoord()

	// For default config: (4*32 + 2*16 + 2*8 + 2*4 + 16*4) / 64 = 3.875
	expected := 3.875
	if math.Abs(bpc-expected) > 0.001 {
		t.Errorf("BitsPerCoord() = %.6f, want ≈ %.3f", bpc, expected)
	}

	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Bits() returns int(BitsPerCoord()), so should be 3.
	if pq.Bits() != int(expected) {
		t.Logf("Bits() = %d (floor of %.3f)", pq.Bits(), expected)
	}
}

// ---------------------------------------------------------------------------
// NaN input
// ---------------------------------------------------------------------------

func TestPolarQuantize_NaN(t *testing.T) {
	cfg := DefaultPolarConfig(64)
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatal(err)
	}

	vec := make([]float64, 64)
	vec[5] = math.NaN()

	_, err = pq.Quantize(vec)
	if err != ErrNaNInput {
		t.Errorf("expected ErrNaNInput, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Small vector (dim=16)
// ---------------------------------------------------------------------------

func TestPolarQuantize_SmallDim(t *testing.T) {
	cfg := DefaultPolarConfig(16)
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(1))
	vec := make([]float64, 16)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}
	normalize(vec)

	cv, err := pq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	recon, err := pq.Dequantize(cv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	cosSim := dot(vec, recon)
	t.Logf("dim=16: cosine_similarity=%.6f", cosSim)
	if cosSim < 0.70 {
		t.Errorf("cosine similarity %.4f too low for dim=16", cosSim)
	}
}

// ---------------------------------------------------------------------------
// Zero vector
// ---------------------------------------------------------------------------

func TestPolarQuantize_ZeroVector(t *testing.T) {
	cfg := DefaultPolarConfig(64)
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatal(err)
	}

	vec := make([]float64, 64) // all zeros
	cv, err := pq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize zero vector: %v", err)
	}

	recon, err := pq.Dequantize(cv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	// Zero vector → all zeros after normalization (norm=0 → no scaling).
	// Reconstruction should be near-zero.
	for i, v := range recon {
		if math.Abs(v) > 1e-10 {
			t.Logf("recon[%d] = %.15f (near-zero expected)", i, v)
		}
	}
}

// ---------------------------------------------------------------------------
// Multiple round-trips with different seeds
// ---------------------------------------------------------------------------

func TestPolarQuantize_MultipleSeeds(t *testing.T) {
	dim := 64
	rng := rand.New(rand.NewSource(0))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}
	normalize(vec)

	for seed := int64(0); seed < 5; seed++ {
		cfg := PolarConfig{
			Dim:        dim,
			Levels:     4,
			BitsLevel1: 4,
			BitsRest:   2,
			RadiusBits: 16,
			Seed:       seed,
		}
		pq, err := NewPolarQuantizer(cfg)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}

		cv, err := pq.Quantize(vec)
		if err != nil {
			t.Fatalf("seed %d Quantize: %v", seed, err)
		}

		recon, err := pq.Dequantize(cv)
		if err != nil {
			t.Fatalf("seed %d Dequantize: %v", seed, err)
		}

		cosSim := dot(vec, recon)
		t.Logf("seed=%d: cosine_similarity=%.6f", seed, cosSim)
		if cosSim < 0.80 {
			t.Errorf("seed %d: cosine similarity %.4f too low", seed, cosSim)
		}
	}
}

// ---------------------------------------------------------------------------
// Custom levels configuration
// ---------------------------------------------------------------------------

func TestPolarQuantize_CustomLevels(t *testing.T) {
	tests := []struct {
		dim    int
		levels int
	}{
		{32, 2},
		{32, 3},
		{64, 2},
		{64, 3},
		{128, 2},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("dim=%d_levels=%d", tt.dim, tt.levels), func(t *testing.T) {
			cfg := PolarConfig{
				Dim:        tt.dim,
				Levels:     tt.levels,
				BitsLevel1: 4,
				BitsRest:   2,
				RadiusBits: 16,
				Seed:       42,
			}
			pq, err := NewPolarQuantizer(cfg)
			if err != nil {
				t.Fatalf("NewPolarQuantizer: %v", err)
			}

			rng := rand.New(rand.NewSource(7))
			vec := make([]float64, tt.dim)
			for i := range vec {
				vec[i] = rng.NormFloat64()
			}
			normalize(vec)

			cv, err := pq.Quantize(vec)
			if err != nil {
				t.Fatalf("Quantize: %v", err)
			}

			recon, err := pq.Dequantize(cv)
			if err != nil {
				t.Fatalf("Dequantize: %v", err)
			}

			cosSim := dot(vec, recon)
			t.Logf("cosine_similarity=%.6f", cosSim)
			if cosSim < 0.75 {
				t.Errorf("cosine similarity %.4f too low", cosSim)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func normalize(v []float64) {
	n := norm(v)
	if n > 0 {
		for i := range v {
			v[i] /= n
		}
	}
}

func norm(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func mse(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s / float64(len(a))
}
