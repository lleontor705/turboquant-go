package quantize

import (
	"math"
	"math/rand"
	"testing"
)

// TestTurboProd_MSE_Compatibility verifies that the MSE sub-quantizer inside
// TurboProdQuantizer produces the same reconstruction as a standalone
// TurboQuantizer with the same seed and (bits-1).
func TestTurboProd_MSE_Compatibility(t *testing.T) {
	const dim = 64
	const bits = 3
	const seed int64 = 42

	// Create standalone TurboQuantizer with bits-1.
	turbo, err := NewTurboQuantizer(dim, bits-1, seed)
	if err != nil {
		t.Fatalf("NewTurboQuantizer: %v", err)
	}

	// Create TurboProdQuantizer (uses bits-1 internally for MSE).
	prod, err := NewTurboProdQuantizer(dim, bits, dim, seed)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	for trial := 0; trial < 10; trial++ {
		vec := make([]float64, dim)
		for i := range vec {
			vec[i] = rng.NormFloat64()
		}

		// Quantize with standalone turbo.
		cvTurbo, err := turbo.Quantize(vec)
		if err != nil {
			t.Fatalf("turbo.Quantize: %v", err)
		}
		reconTurbo, err := turbo.Dequantize(cvTurbo)
		if err != nil {
			t.Fatalf("turbo.Dequantize: %v", err)
		}

		// Quantize with prod, then dequantize (MSE only).
		cvProd, err := prod.Quantize(vec)
		if err != nil {
			t.Fatalf("prod.Quantize: %v", err)
		}
		reconProd, err := prod.Dequantize(cvProd)
		if err != nil {
			t.Fatalf("prod.Dequantize: %v", err)
		}

		// The MSE reconstructions should match (both use same seed, same bits-1).
		for i := range reconTurbo {
			if math.Abs(reconTurbo[i]-reconProd[i]) > 1e-10 {
				t.Errorf("trial %d, dim %d: turbo=%.10f, prod=%.10f",
					trial, i, reconTurbo[i], reconProd[i])
			}
		}
	}
}
