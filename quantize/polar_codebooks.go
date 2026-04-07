package quantize

import (
	"fmt"
	"math"
	"sync"
)

// sinPowerPDF returns the probability density function f(ψ) ∝ sin^n(2ψ)
// for ψ ∈ [0, π/2].
//
// This is the distribution of angles at level ℓ ≥ 2 of the polar decomposition,
// where n = 2^{ℓ-1} - 1:
//
//	Level 2: n=1 → sin(2ψ)
//	Level 3: n=3 → sin³(2ψ)
//	Level 4: n=7 → sin⁷(2ψ)
//
// The function is unnormalized; the caller can normalize if needed.
// Returns 0 outside [0, π/2].
func sinPowerPDF(n int) func(float64) float64 {
	return func(psi float64) float64 {
		if psi < 0 || psi > math.Pi/2 {
			return 0.0
		}
		s2p := math.Sin(2 * psi)
		if s2p <= 0 {
			return 0.0
		}
		// Use log to avoid overflow for large n.
		if n == 1 {
			return s2p
		}
		return math.Pow(s2p, float64(n))
	}
}

// polarCodebookCache caches precomputed Lloyd-Max centroids for polar angle
// distributions, keyed by (level, n, bits). Thread-safe via sync.Map.
var polarCodebookCache sync.Map

// polarCodebookKey is the cache key for polar codebooks.
type polarCodebookKey struct {
	Level int
	N     int
	Bits  int
}

// LevelCodebook returns precomputed Lloyd-Max centroids for a given polar
// decomposition level.
//
//   - Level 1: uniform distribution on [0, 2π) with bitsLevel1 bits → n is ignored
//   - Levels 2+: sin^n(2ψ) on [0, π/2] with bitsRest bits
//
// Codebooks are computed on first access and cached for subsequent calls.
// The returned centroids are sorted and lie within the appropriate range.
func LevelCodebook(level, n, bits int) ([]float64, error) {
	if bits < 1 {
		return nil, fmt.Errorf("quantize: LevelCodebook requires bits >= 1, got %d", bits)
	}
	if level < 1 {
		return nil, fmt.Errorf("quantize: LevelCodebook requires level >= 1, got %d", level)
	}

	key := polarCodebookKey{Level: level, N: n, Bits: bits}

	if cached, ok := polarCodebookCache.Load(key); ok {
		return cached.([]float64), nil
	}

	levels := 1 << bits

	var pdf func(float64) float64
	var minVal, maxVal float64

	if level == 1 {
		// Uniform distribution on [0, 2π).
		pdf = func(float64) float64 { return 1.0 }
		minVal = 0.0
		maxVal = 2 * math.Pi
	} else {
		// sin^n(2ψ) on [0, π/2].
		if n < 1 {
			return nil, fmt.Errorf("quantize: LevelCodebook requires n >= 1 for level >= 2, got %d", n)
		}
		pdf = sinPowerPDF(n)
		minVal = 0.0
		maxVal = math.Pi / 2
	}

	// Cannot fail: pdf is valid, range is valid, levels >= 2, iterations > 0.
	centroids, _, _ := LloydMax(pdf, minVal, maxVal, levels, 200)

	// Store in cache (use LoadOrStore to handle concurrent computation).
	actual, _ := polarCodebookCache.LoadOrStore(key, centroids)
	return actual.([]float64), nil
}

// PolarSinExponent returns the sin exponent n = 2^{level-1} - 1 for a given
// polar decomposition level (1-indexed). Level 1 returns 0 (uniform).
//
//	Level 1: n=0 (uniform)
//	Level 2: n=1 (sin(2ψ))
//	Level 3: n=3 (sin³(2ψ))
//	Level 4: n=7 (sin⁷(2ψ))
func PolarSinExponent(level int) int {
	if level <= 1 {
		return 0
	}
	return (1 << (level - 1)) - 1
}
