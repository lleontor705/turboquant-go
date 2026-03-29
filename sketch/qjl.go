package sketch

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"sort"

	"github.com/lleontor705/turboquant-go/internal/bits"
	"github.com/lleontor705/turboquant-go/internal/serial"
)

// ---------------------------------------------------------------------------
// QJLOptions
// ---------------------------------------------------------------------------

// QJLOptions configures the QJL sketcher.
type QJLOptions struct {
	// Dim is the input vector dimension. Must be > 0.
	Dim int
	// SketchDim is the number of projection dimensions (output sketch size).
	// Must be > 0 and <= Dim.
	SketchDim int
	// Seed is the random seed for reproducibility. Same seed always produces
	// the same projection matrix.
	Seed int64
	// OutlierK is the number of outlier projections to store in full precision.
	// 0 disables outlier handling. Must be <= SketchDim.
	OutlierK int
	// UseSRHT selects SRHT instead of Gaussian projection. Requires Dim to be
	// a power of 2. SRHT is faster (O(n log n)) but less flexible.
	UseSRHT bool
}

// ---------------------------------------------------------------------------
// QJLSketcher
// ---------------------------------------------------------------------------

// QJLSketcher encodes vectors into 1-bit sketches using the QJL algorithm.
//
// Algorithm: Johnson-Lindenstrauss random projection → sign quantization →
// bit packing into []uint64. The result is a BitVector where bit=1 means
// positive projection and bit=0 means non-positive projection.
//
// All types are immutable after construction — goroutine-safe.
type QJLSketcher struct {
	projector Projector
	dim       int
	sketchDim int
	outlierK  int
}

// NewQJLSketcher creates a new QJL sketcher from the given options.
//
// Default: Gaussian projection. Set UseSRHT=true for SRHT (faster, power-of-2
// dims only).
//
// Returns ErrInvalidDimension if Dim <= 0, SketchDim <= 0, or SketchDim > Dim.
// Returns ErrInvalidConfiguration if OutlierK < 0 or OutlierK > SketchDim.
func NewQJLSketcher(opts QJLOptions) (*QJLSketcher, error) {
	if opts.Dim <= 0 || opts.SketchDim <= 0 {
		return nil, fmt.Errorf("sketch: dim=%d sketchDim=%d: %w",
			opts.Dim, opts.SketchDim, ErrInvalidDimension)
	}
	if opts.SketchDim > opts.Dim {
		return nil, fmt.Errorf("sketch: sketchDim=%d > dim=%d: %w",
			opts.SketchDim, opts.Dim, ErrInvalidDimension)
	}
	if opts.OutlierK < 0 {
		return nil, fmt.Errorf("sketch: outlierK=%d: %w",
			opts.OutlierK, ErrInvalidConfiguration)
	}
	if opts.OutlierK > opts.SketchDim {
		return nil, fmt.Errorf("sketch: outlierK=%d > sketchDim=%d: %w",
			opts.OutlierK, opts.SketchDim, ErrInvalidConfiguration)
	}

	var proj Projector
	var err error
	if opts.UseSRHT {
		proj, err = NewSRHT(opts.Dim, opts.SketchDim, opts.Seed)
	} else {
		proj, err = NewGaussianProjection(opts.Dim, opts.SketchDim, opts.Seed)
	}
	if err != nil {
		return nil, err
	}

	return &QJLSketcher{
		projector: proj,
		dim:       opts.Dim,
		sketchDim: opts.SketchDim,
		outlierK:  opts.OutlierK,
	}, nil
}

// Dim returns the input dimension.
func (s *QJLSketcher) Dim() int { return s.dim }

// SketchDim returns the sketch dimension.
func (s *QJLSketcher) SketchDim() int { return s.sketchDim }

// Sketch encodes a vector into a 1-bit sketch (BitVector).
//
// Algorithm:
//  1. Project vec through the random projection matrix → sketchDim-dimensional vector.
//  2. If OutlierK > 0: find top-K values by absolute magnitude, store indices + values.
//  3. Sign quantize: positive → +1, non-positive → -1.
//  4. If OutlierK > 0: zero out outlier positions in the sign vector (set to -1 → bit=0).
//  5. Pack signs into BitVector.
//
// Returns ErrDimensionMismatch if len(vec) != Dim().
func (s *QJLSketcher) Sketch(vec []float64) (*BitVector, error) {
	if len(vec) != s.dim {
		return nil, fmt.Errorf("sketch: input vector length %d != source dim %d: %w",
			len(vec), s.dim, ErrDimensionMismatch)
	}

	// Step 1: Project
	projected, err := s.projector.Project(vec)
	if err != nil {
		return nil, fmt.Errorf("sketch: projection failed: %w", err)
	}

	// Step 2: Find outliers (top-K by absolute value)
	var outlierIndices []int
	var outlierValues []float64
	outlierSet := make(map[int]bool)

	if s.outlierK > 0 {
		type iv struct {
			idx int
			abs float64
		}
		candidates := make([]iv, s.sketchDim)
		for i, v := range projected {
			candidates[i] = iv{i, math.Abs(v)}
		}
		// Sort descending by absolute value.
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].abs > candidates[j].abs
		})

		outlierIndices = make([]int, s.outlierK)
		outlierValues = make([]float64, s.outlierK)
		for i := 0; i < s.outlierK; i++ {
			outlierIndices[i] = candidates[i].idx
			outlierValues[i] = projected[candidates[i].idx]
			outlierSet[candidates[i].idx] = true
		}
	}

	// Step 3: Sign quantize (positive → +1, non-positive → -1)
	signs := make([]int8, s.sketchDim)
	for i, v := range projected {
		if v > 0 {
			signs[i] = 1
		} else {
			signs[i] = -1
		}
	}

	// Step 4: Zero out outlier positions in the sign vector (bit=0 → sign=-1)
	for idx := range outlierSet {
		signs[idx] = -1
	}

	// Step 5: Pack into BitVector
	packed, err := bits.Pack(signs)
	if err != nil {
		return nil, fmt.Errorf("sketch: packing failed: %w", err)
	}

	return &BitVector{
		Bits:           packed,
		Dim:            s.sketchDim,
		OutlierIndices: outlierIndices,
		OutlierValues:  outlierValues,
	}, nil
}

// ---------------------------------------------------------------------------
// BitVector serialization (encoding.BinaryMarshaler / BinaryUnmarshaler)
// ---------------------------------------------------------------------------

// MarshalBinary implements encoding.BinaryMarshaler.
//
// Wire format (all little-endian):
//
//	[version:1] [dim:4] [bits: length-prefixed uint64s] [has_outliers:1]
//	[if has_outliers: [outlier_indices: length-prefixed uint64s] [outlier_values: length-prefixed float64s]]
func (bv BitVector) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	// Version byte
	if err := serial.WriteVersion(buf); err != nil {
		return nil, err
	}

	// Dim as uint32 LE
	var dimBuf [4]byte
	binary.LittleEndian.PutUint32(dimBuf[:], uint32(bv.Dim))
	if _, err := buf.Write(dimBuf[:]); err != nil {
		return nil, err
	}

	// Packed bits
	if err := serial.WriteUint64Slice(buf, bv.Bits); err != nil {
		return nil, err
	}

	// Outlier flag
	hasOutliers := byte(0)
	if len(bv.OutlierIndices) > 0 {
		hasOutliers = 1
	}
	if err := buf.WriteByte(hasOutliers); err != nil {
		return nil, err
	}

	// Outlier data (only if present)
	if hasOutliers == 1 {
		// Convert indices to uint64 for serialization
		idxSlice := make([]uint64, len(bv.OutlierIndices))
		for i, idx := range bv.OutlierIndices {
			idxSlice[i] = uint64(idx)
		}
		if err := serial.WriteUint64Slice(buf, idxSlice); err != nil {
			return nil, err
		}
		if err := serial.WriteFloat64Slice(buf, bv.OutlierValues); err != nil {
			return nil, err
		}
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler.
func (bv *BitVector) UnmarshalBinary(data []byte) error {
	buf := bytes.NewReader(data)

	// Version byte
	if err := serial.ReadVersion(buf); err != nil {
		return err
	}

	// Dim
	var dimBuf [4]byte
	if _, err := buf.Read(dimBuf[:]); err != nil {
		return fmt.Errorf("sketch: reading dim: %w", err)
	}
	bv.Dim = int(binary.LittleEndian.Uint32(dimBuf[:]))

	// Packed bits
	bitsData, err := serial.ReadUint64Slice(buf)
	if err != nil {
		return fmt.Errorf("sketch: reading bits: %w", err)
	}
	bv.Bits = bitsData

	// Outlier flag
	hasOutliers, err := buf.ReadByte()
	if err != nil {
		return fmt.Errorf("sketch: reading outlier flag: %w", err)
	}

	if hasOutliers == 1 {
		// Outlier indices (stored as uint64)
		idxSlice, err := serial.ReadUint64Slice(buf)
		if err != nil {
			return fmt.Errorf("sketch: reading outlier indices: %w", err)
		}
		bv.OutlierIndices = make([]int, len(idxSlice))
		for i, v := range idxSlice {
			bv.OutlierIndices[i] = int(v)
		}

		// Outlier values
		bv.OutlierValues, err = serial.ReadFloat64Slice(buf)
		if err != nil {
			return fmt.Errorf("sketch: reading outlier values: %w", err)
		}
	} else {
		bv.OutlierIndices = nil
		bv.OutlierValues = nil
	}

	return nil
}
