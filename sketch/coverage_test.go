package sketch

import (
	"errors"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// SRHT accessor coverage
// ---------------------------------------------------------------------------

func TestSRHT_SourceDim(t *testing.T) {
	p, err := NewSRHT(64, 16, 42)
	if err != nil {
		t.Fatalf("NewSRHT: %v", err)
	}
	if p.SourceDim() != 64 {
		t.Errorf("SourceDim() = %d, want 64", p.SourceDim())
	}
}

func TestSRHT_TargetDim(t *testing.T) {
	p, err := NewSRHT(64, 16, 42)
	if err != nil {
		t.Fatalf("NewSRHT: %v", err)
	}
	if p.TargetDim() != 16 {
		t.Errorf("TargetDim() = %d, want 16", p.TargetDim())
	}
}

// ---------------------------------------------------------------------------
// SRHT Project dimension mismatch
// ---------------------------------------------------------------------------

func TestSRHT_Project_DimMismatch(t *testing.T) {
	p, err := NewSRHT(64, 16, 42)
	if err != nil {
		t.Fatalf("NewSRHT: %v", err)
	}
	_, err = p.Project([]float64{1, 2, 3}) // wrong length
	if err == nil {
		t.Fatal("expected error for dimension mismatch")
	}
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("expected ErrDimensionMismatch, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// NewSRHT invalid params
// ---------------------------------------------------------------------------

func TestNewSRHT_InvalidParams(t *testing.T) {
	tests := []struct {
		name     string
		src, tgt int
	}{
		{"source zero", 0, 4},
		{"source negative", -1, 4},
		{"target zero", 64, 0},
		{"target negative", 64, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewSRHT(tt.src, tt.tgt, 42)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, ErrInvalidDimension) {
				t.Fatalf("expected ErrInvalidDimension, got %v", err)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// NewQJLSketcher — projector creation error forwarding (SRHT with non-power-of-2)
// ---------------------------------------------------------------------------

func TestNewQJLSketcher_SRHTNonPowerOf2(t *testing.T) {
	// Dim=100 passes the generic validation (> 0, SketchDim <= Dim) but
	// NewSRHT rejects non-power-of-2, exercising the error forwarding path.
	_, err := NewQJLSketcher(QJLOptions{
		Dim:       100,
		SketchDim: 50,
		Seed:      42,
		UseSRHT:   true,
	})
	if err == nil {
		t.Fatal("expected error for SRHT with non-power-of-2 dim")
	}
	if !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("expected ErrInvalidDimension, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// NewQJLSketcher validation branches
// ---------------------------------------------------------------------------

func TestNewQJLSketcher_DuplicateOutlierIndex(t *testing.T) {
	_, err := NewQJLSketcher(QJLOptions{
		Dim:            128,
		SketchDim:      64,
		Seed:           42,
		OutlierIndices: []int{3, 10, 3}, // duplicate
	})
	if err == nil {
		t.Fatal("expected error for duplicate outlier index")
	}
	if !errors.Is(err, ErrInvalidConfiguration) {
		t.Fatalf("expected ErrInvalidConfiguration, got %v", err)
	}
}

func TestNewQJLSketcher_OutlierIndexOutOfRange(t *testing.T) {
	tests := []struct {
		name    string
		indices []int
	}{
		{"negative index", []int{-1}},
		{"index equals sketchDim", []int{64}},
		{"index exceeds sketchDim", []int{100}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewQJLSketcher(QJLOptions{
				Dim:            128,
				SketchDim:      64,
				Seed:           42,
				OutlierIndices: tt.indices,
			})
			if err == nil {
				t.Fatal("expected error for out-of-range outlier index")
			}
			if !errors.Is(err, ErrInvalidConfiguration) {
				t.Fatalf("expected ErrInvalidConfiguration, got %v", err)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// QJL Sketch with OutlierK > 0 (dynamic top-K outlier mode)
// ---------------------------------------------------------------------------

func TestQJLSketch_OutlierK_Dynamic(t *testing.T) {
	const dim = 64
	const sketchDim = 32
	const outlierK = 4

	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:       dim,
		SketchDim: sketchDim,
		Seed:      42,
		OutlierK:  outlierK,
	})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	bv, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch: %v", err)
	}

	// Verify outlier metadata
	if len(bv.OutlierIndices) != outlierK {
		t.Fatalf("len(OutlierIndices) = %d, want %d", len(bv.OutlierIndices), outlierK)
	}
	if len(bv.OutlierValues) != outlierK {
		t.Fatalf("len(OutlierValues) = %d, want %d", len(bv.OutlierValues), outlierK)
	}

	// Verify indices are in valid range
	for _, idx := range bv.OutlierIndices {
		if idx < 0 || idx >= sketchDim {
			t.Errorf("outlier index %d out of range [0, %d)", idx, sketchDim)
		}
	}

	// Verify outlier bits are zeroed out
	for _, idx := range bv.OutlierIndices {
		word := idx / 64
		bit := uint(idx % 64)
		if bv.Bits[word]&(1<<bit) != 0 {
			t.Errorf("outlier at index %d has bit set, expected 0", idx)
		}
	}

	// Determinism check
	bv2, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch (2nd): %v", err)
	}
	if len(bv2.OutlierIndices) != outlierK {
		t.Fatalf("len(OutlierIndices) = %d on 2nd call, want %d", len(bv2.OutlierIndices), outlierK)
	}
	for i := range bv.OutlierIndices {
		if bv.OutlierIndices[i] != bv2.OutlierIndices[i] {
			t.Errorf("OutlierIndices[%d] = %d, want %d", i, bv2.OutlierIndices[i], bv.OutlierIndices[i])
		}
	}
}

// ---------------------------------------------------------------------------
// QJL Sketch with fixed OutlierIndices — verify values match projection
// ---------------------------------------------------------------------------

func TestQJLSketch_FixedOutliers_ValuesPopulated(t *testing.T) {
	const dim = 64
	const sketchDim = 32
	indices := []int{0, 5, 31}

	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:            dim,
		SketchDim:      sketchDim,
		Seed:           42,
		OutlierIndices: indices,
	})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	bv, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch: %v", err)
	}

	if len(bv.OutlierValues) != len(indices) {
		t.Fatalf("len(OutlierValues) = %d, want %d", len(bv.OutlierValues), len(indices))
	}

	// Each outlier value should be non-zero (vanishingly unlikely for random input)
	for i, v := range bv.OutlierValues {
		if v == 0 {
			t.Errorf("OutlierValues[%d] = 0, expected non-zero for random input", i)
		}
	}
}

// ---------------------------------------------------------------------------
// BitVector MarshalBinary/UnmarshalBinary round-trip with outlier data
// ---------------------------------------------------------------------------

func TestSerialization_OutlierRoundTrip(t *testing.T) {
	// Create a sketcher with dynamic outliers and do a full round-trip
	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:       64,
		SketchDim: 32,
		Seed:      42,
		OutlierK:  4,
	})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, 64)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	original, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch: %v", err)
	}

	data, err := original.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	var restored BitVector
	if err := restored.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary: %v", err)
	}

	// Verify all fields
	if restored.Dim != original.Dim {
		t.Errorf("Dim = %d, want %d", restored.Dim, original.Dim)
	}
	for i := range original.Bits {
		if restored.Bits[i] != original.Bits[i] {
			t.Errorf("Bits[%d] = %v, want %v", i, restored.Bits[i], original.Bits[i])
		}
	}
	if len(restored.OutlierIndices) != len(original.OutlierIndices) {
		t.Fatalf("len(OutlierIndices) = %d, want %d", len(restored.OutlierIndices), len(original.OutlierIndices))
	}
	for i := range original.OutlierIndices {
		if restored.OutlierIndices[i] != original.OutlierIndices[i] {
			t.Errorf("OutlierIndices[%d] = %d, want %d", i, restored.OutlierIndices[i], original.OutlierIndices[i])
		}
	}
	if len(restored.OutlierValues) != len(original.OutlierValues) {
		t.Fatalf("len(OutlierValues) = %d, want %d", len(restored.OutlierValues), len(original.OutlierValues))
	}
	for i := range original.OutlierValues {
		if restored.OutlierValues[i] != original.OutlierValues[i] {
			t.Errorf("OutlierValues[%d] = %v, want %v", i, restored.OutlierValues[i], original.OutlierValues[i])
		}
	}
}

// ---------------------------------------------------------------------------
// BitVector UnmarshalBinary with truncated data at each field boundary
// ---------------------------------------------------------------------------

func TestUnmarshalBinary_TruncatedData(t *testing.T) {
	// Build valid data with outliers, then truncate at various points.
	bv := BitVector{
		Bits:           []uint64{0xDEADBEEF, 0xCAFEBABE},
		Dim:            128,
		OutlierIndices: []int{0, 5},
		OutlierValues:  []float64{1.5, -2.5},
	}
	fullData, err := bv.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	tests := []struct {
		name   string
		length int
	}{
		{"empty", 0},
		{"version only", 1},
		{"partial dim", 3},
		{"dim complete, no bits length", 5},
		{"partial bits data", 15},
		{"bits complete, no outlier flag", len(fullData) - 1 - 4 - 8 - 8 - 4 - 8 - 8}, // approximate
		{"outlier flag set, no outlier indices", len(fullData) - 20},
		{"partial outlier values", len(fullData) - 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.length < 0 || tt.length >= len(fullData) {
				t.Skipf("truncation length %d out of range for data of size %d", tt.length, len(fullData))
			}
			var decoded BitVector
			err := decoded.UnmarshalBinary(fullData[:tt.length])
			if err == nil {
				t.Error("expected error for truncated data")
			}
		})
	}
}

func TestUnmarshalBinary_TruncatedAtOutlierFlag(t *testing.T) {
	// Build valid data without outliers, truncate before the outlier flag byte.
	bv := BitVector{
		Bits: []uint64{0xDEADBEEF},
		Dim:  64,
	}
	data, err := bv.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	// Remove just the last byte (the outlier flag)
	truncated := data[:len(data)-1]
	var decoded BitVector
	err = decoded.UnmarshalBinary(truncated)
	if err == nil {
		t.Error("expected error for truncated outlier flag")
	}
}

func TestUnmarshalBinary_TruncatedOutlierIndices(t *testing.T) {
	bv := BitVector{
		Bits:           []uint64{0xFF},
		Dim:            8,
		OutlierIndices: []int{0, 3},
		OutlierValues:  []float64{1.0, 2.0},
	}
	data, err := bv.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	// Truncate in the middle of outlier indices data
	// Format: version(1) + dim(4) + bitsLen(4) + bits(8) + outlierFlag(1) + outlierIdxLen(4) + ...
	// = 1 + 4 + 4 + 8 + 1 = 18 bytes before outlier indices length prefix
	// We want to cut after the length prefix but before all index data
	cutPoint := 18 + 4 + 4 // after outlier indices length prefix + partial data
	if cutPoint >= len(data) {
		cutPoint = len(data) - 4 // fall back
	}
	var decoded BitVector
	err = decoded.UnmarshalBinary(data[:cutPoint])
	if err == nil {
		t.Error("expected error for truncated outlier indices")
	}
}

func TestUnmarshalBinary_TruncatedOutlierValues(t *testing.T) {
	bv := BitVector{
		Bits:           []uint64{0xFF},
		Dim:            8,
		OutlierIndices: []int{0, 3},
		OutlierValues:  []float64{1.0, 2.0},
	}
	data, err := bv.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	// Remove last few bytes to truncate outlier values
	var decoded BitVector
	err = decoded.UnmarshalBinary(data[:len(data)-3])
	if err == nil {
		t.Error("expected error for truncated outlier values")
	}
}

// ---------------------------------------------------------------------------
// hammingDistanceUnchecked
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Sketch — projector error forwarding and bits.Pack error forwarding
// ---------------------------------------------------------------------------

// errorProjector is a test-only Projector that always returns an error.
type errorProjector struct {
	srcDim int
	tgtDim int
}

func (e *errorProjector) SourceDim() int { return e.srcDim }
func (e *errorProjector) TargetDim() int { return e.tgtDim }
func (e *errorProjector) Project(vec []float64) ([]float64, error) {
	return nil, errors.New("forced projection error")
}

func TestQJLSketch_ProjectorError(t *testing.T) {
	// Construct a QJLSketcher with an error-returning projector to exercise
	// the projector.Project error-forwarding path in Sketch.
	s := &QJLSketcher{
		projector: &errorProjector{srcDim: 4, tgtDim: 4},
		dim:       4,
		sketchDim: 4,
	}
	_, err := s.Sketch([]float64{1, 2, 3, 4})
	if err == nil {
		t.Fatal("expected error from projector")
	}
}

// badSignProjector returns projections with NaN values to trigger bits.Pack
// error (NaN is not > 0, so sign will be -1; but we can't easily trigger
// invalid sign values through this path). Instead we directly test that
// bits.Pack errors propagate if they ever occurred.
// Since the sign quantization always produces +1/-1 and bits.Pack only fails
// on values other than +1/-1, this path is truly unreachable. We cover it
// via the projector error path above instead.

// ---------------------------------------------------------------------------
// SRHT Project — FWHT error path (defensive, unreachable normally)
// The FWHT error path at jl.go:148-152 is unreachable because NewSRHT
// validates that sourceDim is a power of 2, and FWHT only fails for
// non-power-of-2 input. This is intentional defensive programming.
// We cannot test it without constructing an srhtProjector with invalid state,
// which we do below.
// ---------------------------------------------------------------------------

func TestSRHT_Project_FWHTError(t *testing.T) {
	// Construct an srhtProjector with non-power-of-2 srcDim to trigger
	// the defensive FWHT error path.
	s := &srhtProjector{
		signs:  make([]float64, 5), // 5 is not power of 2
		srcDim: 5,
		tgtDim: 3,
	}
	for i := range s.signs {
		s.signs[i] = 1.0
	}
	_, err := s.Project([]float64{1, 2, 3, 4, 5})
	if err == nil {
		t.Fatal("expected error from FWHT with non-power-of-2 input")
	}
}

func TestHammingDistanceUnchecked(t *testing.T) {
	a := BitVector{Bits: []uint64{0x0F0F0F0F0F0F0F0F}, Dim: 64}
	b := BitVector{Bits: []uint64{0xF0F0F0F0F0F0F0F0}, Dim: 64}

	got := hammingDistanceUnchecked(a, b)
	want := 64 // all bits differ
	if got != want {
		t.Errorf("hammingDistanceUnchecked = %d, want %d", got, want)
	}

	// Identical vectors
	got = hammingDistanceUnchecked(a, a)
	if got != 0 {
		t.Errorf("hammingDistanceUnchecked(identical) = %d, want 0", got)
	}
}
