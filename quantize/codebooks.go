package quantize

import (
	"fmt"
	"math"
	"sync"
)

// BetaPDF returns the probability density function of the coordinate distribution
// after applying a random rotation to a unit-norm vector in d dimensions.
//
// Each coordinate follows:
//
//	f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}   for x ∈ (-1, 1)
//
// This is related to the Beta distribution: the coordinate follows a
// Beta((d-1)/2, (d-1)/2) distribution scaled to [-1, 1].
//
// Returns ErrInvalidConfig if d < 2.
func BetaPDF(d int) (func(float64) float64, error) {
	if d < 2 {
		return nil, fmt.Errorf("%w: BetaPDF requires d >= 2, got %d", ErrInvalidConfig, d)
	}

	halfD := float64(d) / 2.0
	halfDMinus1 := float64(d-1) / 2.0
	exponent := (float64(d) - 3.0) / 2.0

	// Compute log of normalization constant for numerical stability.
	// For large d, Gamma(d/2) would overflow float64, so we use Lgamma.
	logGamma1, _ := math.Lgamma(halfD)
	logGamma2, _ := math.Lgamma(halfDMinus1)
	logNorm := logGamma1 - 0.5*math.Log(math.Pi) - logGamma2

	return func(x float64) float64 {
		if x <= -1 || x >= 1 {
			return 0.0
		}
		xsq := x * x
		if xsq == 0 {
			return math.Exp(logNorm)
		}
		return math.Exp(logNorm + exponent*math.Log(1-xsq))
	}, nil
}

// precomputedCodebooks caches Lloyd-Max centroids keyed by (d, b).
// Access is synchronized via codebookMu.
var precomputedCodebooks = make(map[[2]int][]float64)

var codebookMu sync.Mutex

// TurboCodebook returns precomputed Lloyd-Max centroids for the Beta(d) distribution
// at the given bit-width b.
//
// Valid bit-widths are 1, 2, 3, or 4 (producing 2, 4, 8, or 16 centroids).
// The dimension d must be at least 2.
//
// Codebooks are computed on first access and cached for subsequent calls.
// The returned centroids are sorted, lie within [-1, 1], and have length 2^b.
func TurboCodebook(d int, b int) ([]float64, error) {
	if d < 2 {
		return nil, fmt.Errorf("quantize: TurboCodebook requires d >= 2, got %d", d)
	}
	if b < 1 || b > 4 {
		return nil, fmt.Errorf("quantize: TurboCodebook requires 1 <= b <= 4, got %d", b)
	}

	key := [2]int{d, b}

	codebookMu.Lock()
	defer codebookMu.Unlock()

	if cb, ok := precomputedCodebooks[key]; ok {
		return cb, nil
	}

	levels := 1 << b // 2^b
	// Cannot fail: d >= 2 validated above.
	pdf, _ := BetaPDF(d)
	// Cannot fail: pdf is valid, [-1,1] is valid range, levels >= 2, iterations > 0.
	centroids, _, _ := LloydMax(pdf, -1.0, 1.0, levels, 100)

	precomputedCodebooks[key] = centroids
	return centroids, nil
}
