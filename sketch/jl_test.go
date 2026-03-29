package sketch

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// GaussianProjection tests
// ---------------------------------------------------------------------------

func TestGaussianProjection_Determinism(t *testing.T) {
	p1, err := NewGaussianProjection(64, 16, 42)
	if err != nil {
		t.Fatalf("NewGaussianProjection: %v", err)
	}
	p2, err := NewGaussianProjection(64, 16, 42)
	if err != nil {
		t.Fatalf("NewGaussianProjection (2nd): %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, 64)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	r1, err := p1.Project(vec)
	if err != nil {
		t.Fatalf("Project: %v", err)
	}
	r2, err := p2.Project(vec)
	if err != nil {
		t.Fatalf("Project (2nd): %v", err)
	}

	for i := range r1 {
		if r1[i] != r2[i] {
			t.Errorf("determinism violation at index %d: %v != %v", i, r1[i], r2[i])
		}
	}
}

func TestGaussianProjection_InvalidDims(t *testing.T) {
	tests := []struct {
		name     string
		src, tgt int
	}{
		{"source zero", 0, 4},
		{"source negative", -1, 4},
		{"target zero", 4, 0},
		{"target negative", 4, -1},
		{"target exceeds source", 4, 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewGaussianProjection(tt.src, tt.tgt, 42)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, ErrInvalidDimension) {
				t.Fatalf("expected ErrInvalidDimension, got %v", err)
			}
		})
	}
}

func TestGaussianProjection_Project(t *testing.T) {
	srcDim, tgtDim := 8, 4
	p, err := NewGaussianProjection(srcDim, tgtDim, 123)
	if err != nil {
		t.Fatalf("NewGaussianProjection: %v", err)
	}

	if p.SourceDim() != srcDim {
		t.Errorf("SourceDim = %d, want %d", p.SourceDim(), srcDim)
	}
	if p.TargetDim() != tgtDim {
		t.Errorf("TargetDim = %d, want %d", p.TargetDim(), tgtDim)
	}

	vec := []float64{1, 2, 3, 4, 5, 6, 7, 8.0}
	out, err := p.Project(vec)
	if err != nil {
		t.Fatalf("Project: %v", err)
	}
	if len(out) != tgtDim {
		t.Fatalf("output length = %d, want %d", len(out), tgtDim)
	}

	// Wrong-length vector should error
	_, err = p.Project([]float64{1, 2, 3})
	if err == nil {
		t.Fatal("expected error for wrong-length input")
	}
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestGaussianProjection_JLProperty(t *testing.T) {
	srcDim, tgtDim := 256, 128
	p, err := NewGaussianProjection(srcDim, tgtDim, 7)
	if err != nil {
		t.Fatalf("NewGaussianProjection: %v", err)
	}

	rng := rand.New(rand.NewSource(2024))
	epsilon := 0.35
	violations := 0
	numPairs := 100

	for k := 0; k < numPairs; k++ {
		u := make([]float64, srcDim)
		v := make([]float64, srcDim)
		for i := 0; i < srcDim; i++ {
			u[i] = rng.NormFloat64()
			v[i] = rng.NormFloat64()
		}

		origDist := euclideanDist(u, v)
		if origDist == 0 {
			continue
		}

		pu, _ := p.Project(u)
		pv, _ := p.Project(v)
		projDist := euclideanDist(pu, pv)

		// With N(0, 1/sourceDim) entries in the k×d matrix, the expected
		// squared projected distance is (k/d) * original squared distance.
		expectedSq := float64(tgtDim) / float64(srcDim) * origDist * origDist
		projSq := projDist * projDist

		if expectedSq == 0 {
			continue
		}
		ratio := projSq / expectedSq
		if math.Abs(ratio-1) > epsilon {
			violations++
		}
	}

	failureRate := float64(violations) / float64(numPairs)
	if failureRate > 0.15 {
		t.Fatalf("JL property violated: %.0f%% of pairs outside (1±%.2f) bound",
			failureRate*100, epsilon)
	}
}

// ---------------------------------------------------------------------------
// SRHT tests
// ---------------------------------------------------------------------------

func TestSRHT_Determinism(t *testing.T) {
	p1, err := NewSRHT(64, 16, 42)
	if err != nil {
		t.Fatalf("NewSRHT: %v", err)
	}
	p2, err := NewSRHT(64, 16, 42)
	if err != nil {
		t.Fatalf("NewSRHT (2nd): %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, 64)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	r1, err := p1.Project(vec)
	if err != nil {
		t.Fatalf("Project: %v", err)
	}
	r2, err := p2.Project(vec)
	if err != nil {
		t.Fatalf("Project (2nd): %v", err)
	}

	for i := range r1 {
		if r1[i] != r2[i] {
			t.Errorf("determinism violation at index %d: %v != %v", i, r1[i], r2[i])
		}
	}
}

func TestSRHT_TargetExceedsSource(t *testing.T) {
	_, err := NewSRHT(64, 128, 42)
	if err == nil {
		t.Fatal("expected error for targetDim > sourceDim")
	}
	if !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("expected ErrInvalidDimension, got %v", err)
	}
}

func TestSRHT_NonPowerOf2(t *testing.T) {
	_, err := NewSRHT(100, 50, 42)
	if err == nil {
		t.Fatal("expected error for non-power-of-2 sourceDim")
	}
	if !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("expected ErrInvalidDimension, got %v", err)
	}
}

func TestSRHT_JLProperty(t *testing.T) {
	srcDim, tgtDim := 512, 128
	p, err := NewSRHT(srcDim, tgtDim, 7)
	if err != nil {
		t.Fatalf("NewSRHT: %v", err)
	}

	rng := rand.New(rand.NewSource(2025))
	epsilon := 0.35
	violations := 0
	numPairs := 100

	for k := 0; k < numPairs; k++ {
		u := make([]float64, srcDim)
		v := make([]float64, srcDim)
		for i := 0; i < srcDim; i++ {
			u[i] = rng.NormFloat64()
			v[i] = rng.NormFloat64()
		}

		origDist := euclideanDist(u, v)
		if origDist == 0 {
			continue
		}

		pu, _ := p.Project(u)
		pv, _ := p.Project(v)
		projDist := euclideanDist(pu, pv)

		// SRHT with normalization preserves norms, so we check ratio directly.
		// But since we subsample (tgtDim < srcDim), the effective norm shrinks
		// by a factor of sqrt(tgtDim/srcDim). We need to account for this.
		//
		// After FWHT + 1/sqrt(n) normalization, the full vector preserves norms.
		// Subsampling first tgtDim entries: expected squared norm ≈ (tgtDim/srcDim) * ||v||².
		// So projected distance² ≈ (tgtDim/srcDim) * orig distance².
		// Ratio of squared distances ≈ tgtDim/srcDim.
		//
		// We compare relative to the expected subsampled distance.
		expectedSq := float64(tgtDim) / float64(srcDim) * origDist * origDist
		projSq := projDist * projDist

		if expectedSq == 0 {
			continue
		}
		ratio := projSq / expectedSq
		if math.Abs(ratio-1) > epsilon {
			violations++
		}
	}

	failureRate := float64(violations) / float64(numPairs)
	if failureRate > 0.15 {
		t.Fatalf("SRHT JL property violated: %.0f%% of pairs outside (1±%.2f) bound",
			failureRate*100, epsilon)
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func euclideanDist(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}
