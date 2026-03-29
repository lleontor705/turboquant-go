package quantize

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// LloydMax core tests
// ---------------------------------------------------------------------------

// TestLloydMax_Uniform verifies that uniform PDF produces approximately
// uniformly spaced centroids.
func TestLloydMax_Uniform(t *testing.T) {
	uniformPDF := func(float64) float64 { return 1.0 }

	centroids, boundaries, err := LloydMax(uniformPDF, 0.0, 1.0, 4, 100)
	if err != nil {
		t.Fatalf("LloydMax: %v", err)
	}

	if len(centroids) != 4 {
		t.Fatalf("len(centroids) = %d, want 4", len(centroids))
	}
	if len(boundaries) != 5 {
		t.Fatalf("len(boundaries) = %d, want 5", len(boundaries))
	}

	// For a uniform PDF, centroids should be at the midpoint of each cell:
	// 0.125, 0.375, 0.625, 0.875
	expectedCentroids := []float64{0.125, 0.375, 0.625, 0.875}
	for i, c := range centroids {
		if math.Abs(c-expectedCentroids[i]) > 0.01 {
			t.Errorf("centroids[%d] = %.6f, want ~%.6f", i, c, expectedCentroids[i])
		}
	}

	// Boundaries: 0, 0.25, 0.5, 0.75, 1.0
	expectedBounds := []float64{0.0, 0.25, 0.5, 0.75, 1.0}
	for i, b := range boundaries {
		if math.Abs(b-expectedBounds[i]) > 0.01 {
			t.Errorf("boundaries[%d] = %.6f, want ~%.6f", i, b, expectedBounds[i])
		}
	}
}

// TestLloydMax_Convergence verifies that more iterations do not increase
// total quantization distortion.
func TestLloydMax_Convergence(t *testing.T) {
	// Gaussian-like PDF centered at 0 with σ² = 0.25
	pdf := func(x float64) float64 {
		return math.Exp(-x * x / 0.5)
	}

	// Run with fewer iterations.
	c10, _, err := LloydMax(pdf, -2.0, 2.0, 4, 10)
	if err != nil {
		t.Fatalf("LloydMax (10 iter): %v", err)
	}

	// Run with more iterations.
	c100, _, err := LloydMax(pdf, -2.0, 2.0, 4, 100)
	if err != nil {
		t.Fatalf("LloydMax (100 iter): %v", err)
	}

	dist10 := computeDistortion(pdf, c10, -2.0, 2.0)
	dist100 := computeDistortion(pdf, c100, -2.0, 2.0)

	t.Logf("Distortion: 10 iter = %.8f, 100 iter = %.8f", dist10, dist100)
	// More iterations should not significantly increase distortion
	// (allow small tolerance for numerical noise in integration).
	if dist100 > dist10*1.01 {
		t.Errorf("more iterations should not increase distortion: 10 iter = %.8f, 100 iter = %.8f",
			dist10, dist100)
	}
}

// computeDistortion computes ∫(x - Q(x))² · pdf(x) dx over [min, max]
// where Q(x) maps x to the centroid of the region containing x.
func computeDistortion(pdf func(float64) float64, centroids []float64, min, max float64) float64 {
	n := len(centroids)
	boundaries := make([]float64, n+1)
	boundaries[0] = min
	boundaries[n] = max
	for i := 1; i < n; i++ {
		boundaries[i] = (centroids[i-1] + centroids[i]) / 2.0
	}

	const numPts = 1000
	total := 0.0
	for i := 0; i < n; i++ {
		a, b := boundaries[i], boundaries[i+1]
		if a >= b {
			continue
		}
		dx := (b - a) / float64(numPts)
		sum := 0.0
		for j := 0; j <= numPts; j++ {
			x := a + float64(j)*dx
			w := pdf(x)
			if j == 0 || j == numPts {
				w *= 0.5
			}
			diff := x - centroids[i]
			sum += diff * diff * w
		}
		total += sum * dx
	}
	return total
}

// ---------------------------------------------------------------------------
// NearestCentroid tests
// ---------------------------------------------------------------------------

func TestNearestCentroid(t *testing.T) {
	centroids := []float64{-1.0, 0.0, 1.0}

	tests := []struct {
		x    float64
		want int
	}{
		{-0.8, 0}, // closest to -1.0 (dist 0.2 vs 0.8 vs 1.8)
		{-0.4, 1}, // closest to 0.0 (dist 0.6 vs 0.4 vs 1.4)
		{0.0, 1},  // exactly at centroid 1
		{0.2, 1},  // closest to 0.0 (dist 1.2 vs 0.2 vs 0.8)
		{0.8, 2},  // closest to 1.0 (dist 1.8 vs 0.8 vs 0.2)
		{1.5, 2},  // closest to 1.0
		{-1.5, 0}, // closest to -1.0
	}

	for _, tt := range tests {
		got := NearestCentroid(tt.x, centroids)
		if got != tt.want {
			t.Errorf("NearestCentroid(%.1f, %v) = %d, want %d", tt.x, centroids, got, tt.want)
		}
	}

	// Empty centroids.
	if idx := NearestCentroid(0.0, nil); idx != -1 {
		t.Errorf("NearestCentroid with nil centroids = %d, want -1", idx)
	}
}

// ---------------------------------------------------------------------------
// QuantizeWithCodebook tests
// ---------------------------------------------------------------------------

func TestQuantizeWithCodebook(t *testing.T) {
	centroids := []float64{-0.5, 0.0, 0.5}

	tests := []struct {
		values []float64
		want   []int
	}{
		{
			values: []float64{-1.0, -0.3, 0.2, 1.0},
			want:   []int{0, 0, 1, 2},
		},
		{
			values: []float64{0.0, 0.5, -0.5},
			want:   []int{1, 2, 0},
		},
	}

	for i, tt := range tests {
		got := QuantizeWithCodebook(tt.values, centroids)
		if len(got) != len(tt.want) {
			t.Errorf("test %d: len(got) = %d, want %d", i, len(got), len(tt.want))
			continue
		}
		for j := range got {
			if got[j] != tt.want[j] {
				t.Errorf("test %d: got[%d] = %d, want %d", i, j, got[j], tt.want[j])
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

func TestLloydMax_SingleLevel(t *testing.T) {
	uniformPDF := func(float64) float64 { return 1.0 }
	centroids, boundaries, err := LloydMax(uniformPDF, 0.0, 1.0, 1, 10)
	if err != nil {
		t.Fatalf("LloydMax: %v", err)
	}
	if len(centroids) != 1 {
		t.Fatalf("len(centroids) = %d, want 1", len(centroids))
	}
	// Single centroid should be the mean of the range = 0.5
	if math.Abs(centroids[0]-0.5) > 0.01 {
		t.Errorf("centroid = %.6f, want ~0.5", centroids[0])
	}
	if len(boundaries) != 2 {
		t.Fatalf("len(boundaries) = %d, want 2", len(boundaries))
	}
}

func TestLloydMax_Errors(t *testing.T) {
	pdf := func(float64) float64 { return 1.0 }

	_, _, err := LloydMax(nil, 0, 1, 4, 10)
	if err == nil {
		t.Error("expected error for nil pdf")
	}

	_, _, err = LloydMax(pdf, 0, 1, 0, 10)
	if err == nil {
		t.Error("expected error for levels < 1")
	}

	_, _, err = LloydMax(pdf, 1, 1, 4, 10)
	if err == nil {
		t.Error("expected error for min >= max")
	}

	_, _, err = LloydMax(pdf, 0, 1, 4, 0)
	if err == nil {
		t.Error("expected error for iterations < 1")
	}
}
