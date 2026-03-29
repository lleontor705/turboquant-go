package quantize

import (
	"fmt"
	"math"
)

// PolarTransform performs the recursive atan2 decomposition of a vector.
//
// Input: d-dimensional vector (after rotation), d must be a multiple of 2^L.
// At each level ℓ, consecutive pairs of radii are converted to an angle (via
// atan2) and a new radius (via hypot), halving the count. After L levels:
//
//	Level 1: d/2 angles in [0, 2π)
//	Level 2: d/4 angles in [0, π/2]
//	...
//	Level L: d/2^L angles in [0, π/2]
//	Final:   d/2^L radii
//
// Energy is conserved: ‖x‖² = Σ (final_radii)² at each level.
func PolarTransform(vec []float64, levels int) (angles [][]float64, radii []float64, err error) {
	n := len(vec)
	if n == 0 {
		return nil, nil, fmt.Errorf("quantize: PolarTransform requires non-empty input")
	}
	if levels < 1 {
		return nil, nil, fmt.Errorf("quantize: PolarTransform requires levels >= 1, got %d", levels)
	}

	// Validate that n is a multiple of 2^levels.
	blockSize := 1 << levels
	if n%blockSize != 0 {
		return nil, nil, fmt.Errorf("quantize: PolarTransform requires dim=%d to be a multiple of %d (2^%d)", n, blockSize, levels)
	}

	// Working copy of radii, starting with the input vector.
	r := make([]float64, n)
	copy(r, vec)

	angles = make([][]float64, levels)

	for ℓ := 0; ℓ < levels; ℓ++ {
		halfN := n >> (ℓ + 1)
		angles[ℓ] = make([]float64, halfN)
		newR := make([]float64, halfN)

		for j := 0; j < halfN; j++ {
			a := r[2*j]
			b := r[2*j+1]
			angles[ℓ][j] = math.Atan2(b, a)
			newR[j] = math.Hypot(a, b)
		}

		r = newR
	}

	return angles, r, nil
}

// InversePolarTransform reverses the polar decomposition, reconstructing a
// d-dimensional vector from quantized angle centroids and final radii.
//
// The reconstruction proceeds from the final level back to level 0:
//
//	For each pair (r_j, θ_j) at level ℓ:
//	  r_{2j-1} = r_j · cos(θ_j)
//	  r_{2j}   = r_j · sin(θ_j)
//
// The dim parameter specifies the original vector length, which must equal
// len(finalRadii) * 2^levels.
func InversePolarTransform(angleCentroids [][]float64, finalRadii []float64, levels int, dim int) ([]float64, error) {
	if levels < 1 {
		return nil, fmt.Errorf("quantize: InversePolarTransform requires levels >= 1, got %d", levels)
	}
	if len(angleCentroids) != levels {
		return nil, fmt.Errorf("quantize: InversePolarTransform expects %d angle levels, got %d", levels, len(angleCentroids))
	}

	expectedDim := len(finalRadii) << levels
	if dim != expectedDim {
		return nil, fmt.Errorf("quantize: InversePolarTransform dim=%d does not match len(radii)=%d * 2^%d=%d", dim, len(finalRadii), levels, expectedDim)
	}

	// Start from final radii, work backwards.
	r := make([]float64, len(finalRadii))
	copy(r, finalRadii)

	for ℓ := levels - 1; ℓ >= 0; ℓ-- {
		if len(angleCentroids[ℓ]) != len(r) {
			return nil, fmt.Errorf("quantize: InversePolarTransform level %d: expected %d angle centroids, got %d", ℓ, len(r), len(angleCentroids[ℓ]))
		}

		newR := make([]float64, 2*len(r))
		for j := 0; j < len(r); j++ {
			θ := angleCentroids[ℓ][j]
			cosθ := math.Cos(θ)
			sinθ := math.Sin(θ)
			newR[2*j] = r[j] * cosθ
			newR[2*j+1] = r[j] * sinθ
		}
		r = newR
	}

	return r, nil
}
