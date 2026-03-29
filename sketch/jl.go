package sketch

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"

	"github.com/lleontor705/turboquant-go/rotate"
)

// Projector projects a vector from sourceDim to targetDim.
// Must be goroutine-safe (immutable after construction).
type Projector interface {
	// Project applies the random projection to vec, returning a
	// targetDim-length vector. The input vec must have length sourceDim.
	Project(vec []float64) ([]float64, error)
	// SourceDim returns the input dimension.
	SourceDim() int
	// TargetDim returns the output (projected) dimension.
	TargetDim() int
}

// ---------------------------------------------------------------------------
// GaussianProjection
// ---------------------------------------------------------------------------

// gaussianProjector stores a dense targetDim × sourceDim matrix with
// entries drawn from N(0, 1/sourceDim). Because the matrix is created once
// in the constructor and never mutated, concurrent calls to Project are safe.
type gaussianProjector struct {
	mat    *mat.Dense
	srcDim int
	tgtDim int
}

// NewGaussianProjection creates a Gaussian random projection matrix.
// Generates a targetDim × sourceDim matrix with N(0,1/sourceDim) entries.
// Deterministic for a given seed.
func NewGaussianProjection(sourceDim, targetDim int, seed int64) (Projector, error) {
	if sourceDim <= 0 || targetDim <= 0 {
		return nil, ErrInvalidDimension
	}
	if targetDim > sourceDim {
		return nil, ErrInvalidDimension
	}

	rng := rand.New(rand.NewSource(seed))
	rows := targetDim
	cols := sourceDim

	data := make([]float64, rows*cols)
	stddev := 1.0 / math.Sqrt(float64(sourceDim))
	for i := range data {
		data[i] = rng.NormFloat64() * stddev
	}

	m := mat.NewDense(rows, cols, data)

	return &gaussianProjector{
		mat:    m,
		srcDim: sourceDim,
		tgtDim: targetDim,
	}, nil
}

func (g *gaussianProjector) SourceDim() int { return g.srcDim }
func (g *gaussianProjector) TargetDim() int { return g.tgtDim }

func (g *gaussianProjector) Project(vec []float64) ([]float64, error) {
	if len(vec) != g.srcDim {
		return nil, fmt.Errorf("sketch: input vector length %d != source dim %d: %w",
			len(vec), g.srcDim, ErrDimensionMismatch)
	}

	// mat-vector multiply: y = M * x
	result := make([]float64, g.tgtDim)
	for i := 0; i < g.tgtDim; i++ {
		var sum float64
		for j := 0; j < g.srcDim; j++ {
			sum += g.mat.At(i, j) * vec[j]
		}
		result[i] = sum
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// SRHT (Subsampled Randomized Hadamard Transform)
// ---------------------------------------------------------------------------

// srhtProjector stores the random sign-flip vector and dimensions.
// No dense matrix is stored; the transform is applied on-the-fly via FWHT.
type srhtProjector struct {
	signs  []float64 // random ±1, length sourceDim
	srcDim int
	tgtDim int
}

// NewSRHT creates a Subsampled Randomized Hadamard Transform projector.
// Algorithm: random sign flip → FWHT → subsample first targetDim coordinates.
// sourceDim must be a power of 2. targetDim must be ≤ sourceDim.
func NewSRHT(sourceDim, targetDim int, seed int64) (Projector, error) {
	if sourceDim <= 0 || targetDim <= 0 {
		return nil, ErrInvalidDimension
	}
	if targetDim > sourceDim {
		return nil, ErrInvalidDimension
	}
	if !rotate.IsPowerOfTwo(sourceDim) {
		return nil, ErrInvalidDimension
	}

	rng := rand.New(rand.NewSource(seed))
	signs := make([]float64, sourceDim)
	for i := range signs {
		if rng.Float64() < 0.5 {
			signs[i] = -1.0
		} else {
			signs[i] = 1.0
		}
	}

	return &srhtProjector{
		signs:  signs,
		srcDim: sourceDim,
		tgtDim: targetDim,
	}, nil
}

func (s *srhtProjector) SourceDim() int { return s.srcDim }
func (s *srhtProjector) TargetDim() int { return s.tgtDim }

func (s *srhtProjector) Project(vec []float64) ([]float64, error) {
	if len(vec) != s.srcDim {
		return nil, fmt.Errorf("sketch: input vector length %d != source dim %d: %w",
			len(vec), s.srcDim, ErrDimensionMismatch)
	}

	// Step 1: element-wise sign flip (copy to avoid mutating input)
	buf := make([]float64, s.srcDim)
	for i, v := range vec {
		buf[i] = v * s.signs[i]
	}

	// Step 2: in-place FWHT (unnormalized)
	if err := rotate.FWHT(buf); err != nil {
		// Should never happen since sourceDim is validated as power-of-2
		// in the constructor, but handle defensively.
		return nil, fmt.Errorf("sketch: FWHT failed: %w", err)
	}

	// Step 3: normalize by 1/sqrt(n) and subsample first targetDim coords.
	scale := 1.0 / math.Sqrt(float64(s.srcDim))
	out := make([]float64, s.tgtDim)
	for i := 0; i < s.tgtDim; i++ {
		out[i] = buf[i] * scale
	}

	return out, nil
}
