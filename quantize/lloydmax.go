package quantize

import (
	"fmt"
	"math"
	"sort"
)

// LloydMax computes optimal quantization centroids and boundaries for a given PDF
// using the Lloyd-Max (Lloyd-II) iterative algorithm.
//
// The algorithm alternates between computing Voronoi boundaries (midpoints between
// adjacent centroids) and recomputing centroids as conditional means over each
// region. Centroids are initialized at equally-spaced quantiles of the CDF for
// robust convergence even with highly concentrated distributions.
//
// Parameters:
//   - pdf: probability density function evaluated on [min, max]
//   - min, max: support range of the distribution
//   - levels: number of quantization levels (2^bits)
//   - iterations: number of Lloyd iterations (typically 100)
//
// Returns centroids (len = levels) and boundaries (len = levels+1, includes min and max).
func LloydMax(pdf func(float64) float64, min, max float64, levels int, iterations int) ([]float64, []float64, error) {
	if pdf == nil {
		return nil, nil, fmt.Errorf("lloydmax: pdf must not be nil")
	}
	if levels < 1 {
		return nil, nil, fmt.Errorf("lloydmax: levels must be >= 1, got %d", levels)
	}
	if min >= max {
		return nil, nil, fmt.Errorf("lloydmax: min (%.6f) must be less than max (%.6f)", min, max)
	}
	if iterations < 1 {
		return nil, nil, fmt.Errorf("lloydmax: iterations must be >= 1, got %d", iterations)
	}

	n := levels

	// Compute a rough CDF via trapezoidal integration for smart initialization.
	// This ensures centroids are placed where the PDF has mass, which is critical
	// for highly concentrated distributions (e.g. Beta(1024) on [-1,1]).
	const cdfN = 10000
	cdfDx := (max - min) / float64(cdfN)
	cdf := make([]float64, cdfN+1)
	for i := 1; i <= cdfN; i++ {
		xPrev := min + float64(i-1)*cdfDx
		xCurr := min + float64(i)*cdfDx
		cdf[i] = cdf[i-1] + (pdf(xPrev)+pdf(xCurr))*cdfDx/2
	}
	totalArea := cdf[cdfN]

	// Initialize centroids at equally-spaced quantiles of the CDF.
	centroids := make([]float64, n)
	if totalArea > 0 {
		for i := 0; i < n; i++ {
			target := totalArea * (float64(i) + 0.5) / float64(n)
			// searchCDF returns at most len(cdf)-1 = cdfN via binary search bounds.
			idx := searchCDF(cdf, target)
			centroids[i] = min + float64(idx)*cdfDx
		}
	} else {
		// Fallback: uniform initialization when PDF is identically zero.
		step := (max - min) / float64(n)
		for i := 0; i < n; i++ {
			centroids[i] = min + step*(float64(i)+0.5)
		}
	}

	// Run Lloyd iterations.
	const numIntegPts = 1000
	for iter := 0; iter < iterations; iter++ {
		// Step 1: Compute boundaries as midpoints between adjacent centroids.
		boundaries := make([]float64, n+1)
		boundaries[0] = min
		boundaries[n] = max
		for i := 1; i < n; i++ {
			boundaries[i] = (centroids[i-1] + centroids[i]) / 2.0
		}

		// Step 2: Recompute each centroid as the conditional mean over its region,
		// using trapezoidal numerical integration.
		newCentroids := make([]float64, n)
		for i := 0; i < n; i++ {
			a, b := boundaries[i], boundaries[i+1]
			if a >= b {
				newCentroids[i] = (a + b) / 2.0
				continue
			}

			dx := (b - a) / float64(numIntegPts)
			var sumXW, sumW float64

			for j := 0; j <= numIntegPts; j++ {
				x := a + float64(j)*dx
				w := pdf(x)
				if j == 0 || j == numIntegPts {
					w *= 0.5
				}
				sumXW += x * w
				sumW += w
			}
			sumXW *= dx
			sumW *= dx

			if sumW > 0 {
				newCentroids[i] = sumXW / sumW
			} else {
				// Region has negligible probability — place centroid at region midpoint
				// to maintain sorted order.
				newCentroids[i] = (a + b) / 2.0
			}
		}

		centroids = newCentroids
	}

	// Compute final boundaries from converged centroids.
	boundaries := make([]float64, n+1)
	boundaries[0] = min
	boundaries[n] = max
	for i := 1; i < n; i++ {
		boundaries[i] = (centroids[i-1] + centroids[i]) / 2.0
	}

	return centroids, boundaries, nil
}

// searchCDF returns the smallest index i such that cdf[i] >= target.
func searchCDF(cdf []float64, target float64) int {
	lo, hi := 0, len(cdf)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if cdf[mid] < target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// NearestCentroid finds the index of the nearest centroid to value x.
// Centroids must be sorted in ascending order. Uses binary search for O(log k).
// Returns -1 if centroids is empty.
func NearestCentroid(x float64, centroids []float64) int {
	n := len(centroids)
	if n == 0 {
		return -1
	}
	// Binary search: find the insertion point.
	i := sort.SearchFloat64s(centroids, x)
	if i == 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	// Compare distances to centroids[i-1] and centroids[i].
	if math.Abs(x-centroids[i-1]) <= math.Abs(x-centroids[i]) {
		return i - 1
	}
	return i
}

// QuantizeWithCodebook maps each value to the index of its nearest centroid.
func QuantizeWithCodebook(values []float64, centroids []float64) []int {
	indices := make([]int, len(values))
	for i, v := range values {
		indices[i] = NearestCentroid(v, centroids)
	}
	return indices
}
