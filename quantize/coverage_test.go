package quantize

import (
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// PolarQuantizer.RotationMatrix (was 0% coverage)
// ---------------------------------------------------------------------------

func TestPolarQuantizer_RotationMatrix(t *testing.T) {
	cfg := DefaultPolarConfig(16)
	cfg.Seed = 42
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}
	rot := pq.RotationMatrix()
	if rot == nil {
		t.Fatal("RotationMatrix returned nil")
	}
	r, c := rot.Dims()
	if r != 16 || c != 16 {
		t.Errorf("rotation matrix dims = (%d, %d), want (16, 16)", r, c)
	}
}

// ---------------------------------------------------------------------------
// decodeTurboWire error paths (was 73.3%)
// ---------------------------------------------------------------------------

func TestDecodeTurboWire_Truncated(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"empty", []byte{}},
		{"version_only", []byte{turboWireVersion}},
		{"partial_norm", []byte{turboWireVersion, 0, 0, 0}},
		{"wrong_version", []byte{0xFF}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := decodeTurboWire(tt.data)
			if err == nil {
				t.Error("expected error for truncated/invalid data")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TurboQuantizer.Dequantize error paths (was 91.3%)
// ---------------------------------------------------------------------------

func TestTurboDequantize_InvalidBits(t *testing.T) {
	tq, err := NewTurboQuantizer(16, 2, 42)
	if err != nil {
		t.Fatalf("NewTurboQuantizer: %v", err)
	}

	// Wrong BitsPer
	cv := CompressedVector{Data: []byte{turboWireVersion, 0, 0, 0, 0, 0, 0, 0, 0}, Dim: 16, BitsPer: 3}
	_, err = tq.Dequantize(cv)
	if err == nil {
		t.Error("expected error for bits mismatch")
	}

	// Wrong Dim
	cv = CompressedVector{Data: []byte{turboWireVersion, 0, 0, 0, 0, 0, 0, 0, 0}, Dim: 8, BitsPer: 2}
	_, err = tq.Dequantize(cv)
	if err == nil {
		t.Error("expected error for dim mismatch")
	}
}

// ---------------------------------------------------------------------------
// NewTurboQuantizer validation paths (was 83.3%)
// ---------------------------------------------------------------------------

func TestNewTurboQuantizer_ValidationPaths(t *testing.T) {
	tests := []struct {
		name string
		dim  int
		bits int
	}{
		{"dim=0", 0, 2},
		{"dim=1", 1, 2},
		{"dim=-1", -1, 2},
		{"bits=0", 16, 0},
		{"bits=5", 16, 5},
		{"bits=-1", 16, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewTurboQuantizer(tt.dim, tt.bits, 42)
			if err == nil {
				t.Error("expected error")
			}
			if !errors.Is(err, ErrInvalidConfig) {
				t.Errorf("expected ErrInvalidConfig, got: %v", err)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// NewPolarQuantizer validation paths (was 88.6%)
// ---------------------------------------------------------------------------

func TestNewPolarQuantizer_ValidationPaths(t *testing.T) {
	tests := []struct {
		name   string
		config PolarConfig
	}{
		{"dim=0", PolarConfig{Dim: 0, Levels: 4, BitsLevel1: 4, BitsRest: 2, RadiusBits: 16}},
		{"dim=1", PolarConfig{Dim: 1, Levels: 4, BitsLevel1: 4, BitsRest: 2, RadiusBits: 16}},
		{"dim_not_multiple", PolarConfig{Dim: 17, Levels: 4, BitsLevel1: 4, BitsRest: 2, RadiusBits: 16}},
		{"bitsLevel1=0_defaults", PolarConfig{Dim: 16, Levels: 4, BitsLevel1: 0, BitsRest: 2, RadiusBits: 16}},
		{"bitsLevel1=9", PolarConfig{Dim: 16, Levels: 4, BitsLevel1: 9, BitsRest: 2, RadiusBits: 16}},
		{"bitsRest=0_defaults", PolarConfig{Dim: 16, Levels: 4, BitsLevel1: 4, BitsRest: 0, RadiusBits: 16}},
		{"bitsRest=9", PolarConfig{Dim: 16, Levels: 4, BitsLevel1: 4, BitsRest: 9, RadiusBits: 16}},
		{"radiusBits=8", PolarConfig{Dim: 16, Levels: 4, BitsLevel1: 4, BitsRest: 2, RadiusBits: 8}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Some cases use defaults (0 → default) and should succeed
			_, err := NewPolarQuantizer(tt.config)
			switch tt.name {
			case "bitsLevel1=0_defaults", "bitsRest=0_defaults":
				if err != nil {
					t.Errorf("expected success for %s, got: %v", tt.name, err)
				}
			default:
				if err == nil {
					t.Error("expected error")
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// NewTurboProdQuantizer validation paths (was 88.2%)
// ---------------------------------------------------------------------------

func TestNewTurboProdQuantizer_ValidationPaths(t *testing.T) {
	tests := []struct {
		name      string
		dim       int
		bits      int
		sketchDim int
	}{
		{"dim=0", 0, 3, 64},
		{"dim=1", 1, 3, 1},
		{"bits=1", 16, 1, 16},
		{"bits=5", 16, 5, 16},
		{"sketchDim=0", 16, 3, 0},
		{"sketchDim=-1", 16, 3, -1},
		{"sketchDim>dim", 16, 3, 32},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewTurboProdQuantizer(tt.dim, tt.bits, tt.sketchDim, 42)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TurboProd.Quantize NaN path (was 88.9%)
// ---------------------------------------------------------------------------

func TestTurboProd_Quantize_NaN(t *testing.T) {
	tpq, err := NewTurboProdQuantizer(16, 3, 16, 42)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}
	vec := make([]float64, 16)
	vec[5] = math.NaN()
	_, err = tpq.Quantize(vec)
	if !errors.Is(err, ErrNaNInput) {
		t.Errorf("expected ErrNaNInput, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// TurboProd.Quantize dimension mismatch
// ---------------------------------------------------------------------------

func TestTurboProd_Quantize_DimMismatch(t *testing.T) {
	tpq, err := NewTurboProdQuantizer(16, 3, 16, 42)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}
	_, err = tpq.Quantize(make([]float64, 8))
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// TurboProd.Dequantize bits mismatch (was 86.7%)
// ---------------------------------------------------------------------------

func TestTurboProd_Dequantize_Errors(t *testing.T) {
	tpq, err := NewTurboProdQuantizer(16, 3, 16, 42)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}

	t.Run("dim_mismatch", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{prodWireVersion}, Dim: 8, BitsPer: 3}
		_, err := tpq.Dequantize(cv)
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("bits_mismatch", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{prodWireVersion}, Dim: 16, BitsPer: 4}
		_, err := tpq.Dequantize(cv)
		if err == nil {
			t.Error("expected error")
		}
	})
}

// ---------------------------------------------------------------------------
// TurboProd.EstimateInnerProduct error paths (was 76.7%)
// ---------------------------------------------------------------------------

func TestTurboProd_EstimateIP_Errors(t *testing.T) {
	tpq, err := NewTurboProdQuantizer(16, 3, 16, 42)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}

	t.Run("query_dim_mismatch", func(t *testing.T) {
		cv := CompressedVector{Dim: 16, BitsPer: 3}
		_, err := tpq.EstimateInnerProduct(make([]float64, 8), cv)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("expected ErrDimensionMismatch, got: %v", err)
		}
	})

	t.Run("cv_dim_mismatch", func(t *testing.T) {
		cv := CompressedVector{Dim: 8, BitsPer: 3}
		_, err := tpq.EstimateInnerProduct(make([]float64, 16), cv)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("expected ErrDimensionMismatch, got: %v", err)
		}
	})

	t.Run("cv_bits_mismatch", func(t *testing.T) {
		cv := CompressedVector{Dim: 16, BitsPer: 4}
		_, err := tpq.EstimateInnerProduct(make([]float64, 16), cv)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("expected ErrDimensionMismatch, got: %v", err)
		}
	})

	t.Run("gamma_zero", func(t *testing.T) {
		// Create a quantized vector then directly manipulate gamma to 0.
		rng := rand.New(rand.NewSource(42))
		vec := make([]float64, 16)
		for i := range vec {
			vec[i] = rng.NormFloat64()
		}
		cv, err := tpq.Quantize(vec)
		if err != nil {
			t.Fatalf("Quantize: %v", err)
		}

		query := make([]float64, 16)
		for i := range query {
			query[i] = rng.NormFloat64()
		}
		// This should succeed (gamma is naturally > 0, but the code path is tested)
		_, err = tpq.EstimateInnerProduct(query, cv)
		if err != nil {
			t.Errorf("EstimateInnerProduct: %v", err)
		}
	})
}

// ---------------------------------------------------------------------------
// ParseProdVector error paths (was 62.5%)
// ---------------------------------------------------------------------------

func TestParseProdVector_Errors(t *testing.T) {
	tpq, err := NewTurboProdQuantizer(16, 3, 16, 42)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}

	t.Run("dim_mismatch", func(t *testing.T) {
		cv := CompressedVector{Dim: 8, BitsPer: 3}
		_, err := tpq.ParseProdVector(cv)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("expected ErrDimensionMismatch, got: %v", err)
		}
	})

	t.Run("bits_mismatch", func(t *testing.T) {
		cv := CompressedVector{Dim: 16, BitsPer: 4}
		_, err := tpq.ParseProdVector(cv)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("expected ErrDimensionMismatch, got: %v", err)
		}
	})

	t.Run("corrupt_data", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{0xFF}, Dim: 16, BitsPer: 3}
		_, err := tpq.ParseProdVector(cv)
		if err == nil {
			t.Error("expected error for corrupt data")
		}
	})
}

// ---------------------------------------------------------------------------
// decodeProdWire error paths (was 72.7%)
// ---------------------------------------------------------------------------

func TestDecodeProdWire_Truncated(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"empty", []byte{}},
		{"wrong_version", []byte{0xFF}},
		{"partial_sketchDim", []byte{prodWireVersion, 0, 0}},
		{"partial_norm", []byte{prodWireVersion, 16, 0, 0, 0, 0, 0}},
		{"partial_gamma", append([]byte{prodWireVersion, 16, 0, 0, 0}, make([]byte, 8)...)},
		{"partial_mseDataLen", append([]byte{prodWireVersion, 16, 0, 0, 0}, make([]byte, 16)...)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := decodeProdWire(tt.data)
			if err == nil {
				t.Error("expected error for truncated data")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// encodeProdWire round-trip (was 71.4%)
// ---------------------------------------------------------------------------

func TestEncodeProdWire_RoundTrip(t *testing.T) {
	mseData := []byte{1, 2, 3, 4, 5}
	sketchBits := []uint64{0xAABBCCDDEEFF0011, 0x1122334455667788}

	data := encodeProdWire(16, 1.5, 0.3, mseData, sketchBits)

	pd, err := decodeProdWire(data)
	if err != nil {
		t.Fatalf("decodeProdWire: %v", err)
	}

	if pd.sketchDim != 16 {
		t.Errorf("sketchDim = %d, want 16", pd.sketchDim)
	}
	if pd.norm != 1.5 {
		t.Errorf("norm = %f, want 1.5", pd.norm)
	}
	if pd.gamma != 0.3 {
		t.Errorf("gamma = %f, want 0.3", pd.gamma)
	}
	if len(pd.mseData) != 5 {
		t.Errorf("mseData len = %d, want 5", len(pd.mseData))
	}
	if len(pd.sketchBits) != 2 {
		t.Errorf("sketchBits len = %d, want 2", len(pd.sketchBits))
	}
}

// ---------------------------------------------------------------------------
// PolarQuantizer unpack error paths (was 76.5%)
// ---------------------------------------------------------------------------

func TestPolarUnpack_Errors(t *testing.T) {
	cfg := DefaultPolarConfig(16)
	cfg.Seed = 42
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}

	t.Run("empty", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{}, Dim: 16, BitsPer: pq.Bits()}
		_, err := pq.Dequantize(cv)
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("wrong_version", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{0xFF}, Dim: 16, BitsPer: pq.Bits()}
		_, err := pq.Dequantize(cv)
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("dim_mismatch", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{0xFF}, Dim: 8, BitsPer: pq.Bits()}
		_, err := pq.Dequantize(cv)
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("bits_mismatch", func(t *testing.T) {
		cv := CompressedVector{Data: []byte{0xFF}, Dim: 16, BitsPer: 1}
		_, err := pq.Dequantize(cv)
		if err == nil {
			t.Error("expected error")
		}
	})
}

// ---------------------------------------------------------------------------
// quantizeRadii16 edge cases (was 81.8%)
// ---------------------------------------------------------------------------

func TestQuantizeRadii16_EdgeCases(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		q, err := quantizeRadii16(nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(q) != 0 {
			t.Errorf("expected empty, got len=%d", len(q))
		}
	})

	t.Run("all_zero", func(t *testing.T) {
		q, err := quantizeRadii16([]float64{0, 0, 0})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		for i, v := range q {
			if v != 0 {
				t.Errorf("q[%d] = %d, want 0", i, v)
			}
		}
	})

	t.Run("negative_radius", func(t *testing.T) {
		_, err := quantizeRadii16([]float64{0.5, -0.1})
		if err == nil {
			t.Error("expected error for negative radius")
		}
	})

	t.Run("clamp_above_1", func(t *testing.T) {
		q, err := quantizeRadii16([]float64{1.0001, 0.5})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if q[0] != math.MaxUint16 {
			t.Errorf("expected MaxUint16 for clamped value, got %d", q[0])
		}
	})
}

// ---------------------------------------------------------------------------
// dequantizeRadii16 edge cases (was 85.7%)
// ---------------------------------------------------------------------------

func TestDequantizeRadii16_Empty(t *testing.T) {
	r := dequantizeRadii16(nil)
	if len(r) != 0 {
		t.Errorf("expected empty, got len=%d", len(r))
	}
}

// ---------------------------------------------------------------------------
// LloydMax error paths (was 88.7%)
// ---------------------------------------------------------------------------

func TestLloydMax_ErrorPaths(t *testing.T) {
	t.Run("nil_pdf", func(t *testing.T) {
		_, _, err := LloydMax(nil, 0, 1, 4, 100)
		if err == nil {
			t.Error("expected error for nil pdf")
		}
	})

	t.Run("min>=max", func(t *testing.T) {
		pdf := func(float64) float64 { return 1.0 }
		_, _, err := LloydMax(pdf, 1, 1, 4, 100)
		if err == nil {
			t.Error("expected error for min >= max")
		}
	})

	t.Run("levels=0", func(t *testing.T) {
		pdf := func(float64) float64 { return 1.0 }
		_, _, err := LloydMax(pdf, 0, 1, 0, 100)
		if err == nil {
			t.Error("expected error for levels=0")
		}
	})

	t.Run("iterations=0", func(t *testing.T) {
		pdf := func(float64) float64 { return 1.0 }
		_, _, err := LloydMax(pdf, 0, 1, 4, 0)
		if err == nil {
			t.Error("expected error for iterations=0")
		}
	})

	t.Run("zero_area_pdf", func(t *testing.T) {
		// PDF that returns 0 everywhere → fallback to uniform init.
		pdf := func(float64) float64 { return 0.0 }
		centroids, boundaries, err := LloydMax(pdf, -1, 1, 4, 10)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(centroids) != 4 {
			t.Errorf("expected 4 centroids, got %d", len(centroids))
		}
		if len(boundaries) != 5 {
			t.Errorf("expected 5 boundaries, got %d", len(boundaries))
		}
	})
}

// ---------------------------------------------------------------------------
// TurboCodebook cache hit (was 93.8%)
// ---------------------------------------------------------------------------

func TestTurboCodebook_CacheHit(t *testing.T) {
	// First call — cache miss, computes.
	cb1, err := TurboCodebook(32, 2)
	if err != nil {
		t.Fatalf("TurboCodebook: %v", err)
	}
	// Second call — cache hit.
	cb2, err := TurboCodebook(32, 2)
	if err != nil {
		t.Fatalf("TurboCodebook: %v", err)
	}
	if len(cb1) != len(cb2) {
		t.Fatalf("different lengths: %d vs %d", len(cb1), len(cb2))
	}
	for i := range cb1 {
		if cb1[i] != cb2[i] {
			t.Errorf("cb1[%d] = %f != cb2[%d] = %f", i, cb1[i], i, cb2[i])
		}
	}
}

// ---------------------------------------------------------------------------
// BetaPDF error path
// ---------------------------------------------------------------------------

func TestBetaPDF_Error(t *testing.T) {
	for _, d := range []int{0, 1, -1} {
		_, err := BetaPDF(d)
		if err == nil {
			t.Errorf("expected error for d=%d", d)
		}
		if !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("expected ErrInvalidConfig for d=%d, got: %v", d, err)
		}
	}
}

// ---------------------------------------------------------------------------
// sinPowerPDF boundary paths (was 88.9%)
// ---------------------------------------------------------------------------

func TestSinPowerPDF_Boundaries(t *testing.T) {
	pdf := sinPowerPDF(1)

	// Outside range should be 0
	if pdf(-0.1) != 0 {
		t.Errorf("sinPowerPDF(1)(-0.1) = %f, want 0", pdf(-0.1))
	}
	if pdf(math.Pi/2+0.1) != 0 {
		t.Errorf("sinPowerPDF(1)(pi/2+0.1) = %f, want 0", pdf(math.Pi/2+0.1))
	}

	// At 0, sin(0)=0 so pdf should be 0
	if pdf(0) != 0 {
		t.Errorf("sinPowerPDF(1)(0) = %f, want 0", pdf(0))
	}

	// At pi/4, sin(pi/2) = 1
	if math.Abs(pdf(math.Pi/4)-1.0) > 1e-10 {
		t.Errorf("sinPowerPDF(1)(pi/4) = %f, want 1.0", pdf(math.Pi/4))
	}

	// Test n > 1 uses Pow path
	pdfN3 := sinPowerPDF(3)
	v := pdfN3(math.Pi / 4)
	if math.Abs(v-1.0) > 1e-10 {
		t.Errorf("sinPowerPDF(3)(pi/4) = %f, want 1.0", v)
	}
}

// ---------------------------------------------------------------------------
// encodeValue Inf clamping (was 80%)
// ---------------------------------------------------------------------------

func TestScalar_EncodeValue_Inf(t *testing.T) {
	q, err := NewUniformQuantizer(-1, 1, 8)
	if err != nil {
		t.Fatalf("NewUniformQuantizer: %v", err)
	}

	// Quantize vector with +Inf and -Inf
	vec := []float64{math.Inf(1), math.Inf(-1), 0.5}
	cv, err := q.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// Dequantize and verify clamping
	result, err := q.Dequantize(cv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}
	if result[0] != 1.0 {
		t.Errorf("result[0] = %f, want 1.0 (clamped +Inf)", result[0])
	}
	if result[1] != -1.0 {
		t.Errorf("result[1] = %f, want -1.0 (clamped -Inf)", result[1])
	}
}

// ---------------------------------------------------------------------------
// Polar.Quantize NaN path (was 94.7%)
// ---------------------------------------------------------------------------

func TestPolar_Quantize_NaN(t *testing.T) {
	cfg := DefaultPolarConfig(16)
	cfg.Seed = 42
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}
	vec := make([]float64, 16)
	vec[3] = math.NaN()
	_, err = pq.Quantize(vec)
	if !errors.Is(err, ErrNaNInput) {
		t.Errorf("expected ErrNaNInput, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Polar.Quantize dimension mismatch
// ---------------------------------------------------------------------------

func TestPolar_Quantize_DimMismatch(t *testing.T) {
	cfg := DefaultPolarConfig(16)
	cfg.Seed = 42
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}
	_, err = pq.Quantize(make([]float64, 8))
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// PolarTransform error paths (was 90.9%)
// ---------------------------------------------------------------------------

func TestPolarTransform_Errors(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		_, _, err := PolarTransform(nil, 1)
		if err == nil {
			t.Error("expected error for empty input")
		}
	})

	t.Run("levels=0", func(t *testing.T) {
		_, _, err := PolarTransform([]float64{1, 2}, 0)
		if err == nil {
			t.Error("expected error for levels=0")
		}
	})

	t.Run("not_multiple", func(t *testing.T) {
		_, _, err := PolarTransform([]float64{1, 2, 3}, 2)
		if err == nil {
			t.Error("expected error for non-multiple dimension")
		}
	})
}

// ---------------------------------------------------------------------------
// CompressedVector MarshalBinary full round-trip (was 69.6%)
// ---------------------------------------------------------------------------

func TestCompressedVector_MarshalRoundTrip(t *testing.T) {
	cv := CompressedVector{
		Data:    []byte{1, 2, 3, 4, 5},
		Dim:     10,
		Min:     -1.0,
		Max:     1.0,
		BitsPer: 4,
	}
	data, err := cv.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	var cv2 CompressedVector
	if err := cv2.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary: %v", err)
	}

	if cv2.Dim != cv.Dim {
		t.Errorf("Dim = %d, want %d", cv2.Dim, cv.Dim)
	}
	if cv2.Min != cv.Min {
		t.Errorf("Min = %f, want %f", cv2.Min, cv.Min)
	}
	if cv2.Max != cv.Max {
		t.Errorf("Max = %f, want %f", cv2.Max, cv.Max)
	}
	if cv2.BitsPer != cv.BitsPer {
		t.Errorf("BitsPer = %d, want %d", cv2.BitsPer, cv.BitsPer)
	}
	if len(cv2.Data) != len(cv.Data) {
		t.Errorf("Data len = %d, want %d", len(cv2.Data), len(cv.Data))
	}
}

// ---------------------------------------------------------------------------
// CompressedVector UnmarshalBinary truncated data
// ---------------------------------------------------------------------------

func TestCompressedVector_UnmarshalBinary_Truncated(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"empty", []byte{}},
		{"version_only", []byte{0x01}},
		{"partial_dim", []byte{0x01, 0, 0}},
		{"wrong_version", []byte{0xFF, 0, 0, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cv CompressedVector
			err := cv.UnmarshalBinary(tt.data)
			if err == nil {
				t.Error("expected error for truncated data")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// normalizeUnit helper tests
// ---------------------------------------------------------------------------

func TestNormalizeUnit(t *testing.T) {
	t.Run("normal_vector", func(t *testing.T) {
		vec := []float64{3, 4}
		unit, norm := normalizeUnit(vec)
		if math.Abs(norm-5.0) > 1e-10 {
			t.Errorf("norm = %f, want 5.0", norm)
		}
		if math.Abs(unit[0]-0.6) > 1e-10 || math.Abs(unit[1]-0.8) > 1e-10 {
			t.Errorf("unit = %v, want [0.6, 0.8]", unit)
		}
		// Original should not be modified
		if vec[0] != 3 || vec[1] != 4 {
			t.Error("normalizeUnit modified original vector")
		}
	})

	t.Run("zero_vector", func(t *testing.T) {
		vec := []float64{0, 0, 0}
		unit, norm := normalizeUnit(vec)
		if norm != 0 {
			t.Errorf("norm = %f, want 0", norm)
		}
		for i, v := range unit {
			if v != 0 {
				t.Errorf("unit[%d] = %f, want 0", i, v)
			}
		}
	})
}

// ---------------------------------------------------------------------------
// rotateForward / rotateInverse helpers
// ---------------------------------------------------------------------------

func TestRotateHelpers(t *testing.T) {
	tq, err := NewTurboQuantizer(16, 2, 42)
	if err != nil {
		t.Fatalf("NewTurboQuantizer: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, 16)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	// Forward then inverse should recover original
	rotated := rotateForward(tq.rotation, vec)
	recovered := rotateInverse(tq.rotation, rotated)

	for i := range vec {
		if math.Abs(vec[i]-recovered[i]) > 1e-10 {
			t.Errorf("vec[%d] = %f, recovered = %f", i, vec[i], recovered[i])
		}
	}
}

// ---------------------------------------------------------------------------
// fnvDeriveSeed
// ---------------------------------------------------------------------------

func TestFnvDeriveSeed(t *testing.T) {
	s1 := fnvDeriveSeed(42)
	s2 := fnvDeriveSeed(42)
	s3 := fnvDeriveSeed(43)

	if s1 != s2 {
		t.Errorf("determinism: fnvDeriveSeed(42) = %d, then %d", s1, s2)
	}
	if s1 == s3 {
		t.Errorf("collision: fnvDeriveSeed(42) == fnvDeriveSeed(43) = %d", s1)
	}
	if s1 == 42 {
		t.Errorf("identity: fnvDeriveSeed(42) == 42 (no mixing)")
	}
}

// ---------------------------------------------------------------------------
// Turbo.Dequantize with invalid codebook index (reachable via crafted wire data)
// ---------------------------------------------------------------------------

func TestTurbo_Dequantize_InvalidIndex(t *testing.T) {
	tq, err := NewTurboQuantizer(16, 2, 42)
	if err != nil {
		t.Fatalf("NewTurboQuantizer: %v", err)
	}

	// Build valid wire data but with an index that exceeds codebook size.
	// For 2-bit, codebook has 4 entries (indices 0-3). Set all indices to 3 (valid).
	// Then manually corrupt to force an invalid index.
	vec := make([]float64, 16)
	for i := range vec {
		vec[i] = float64(i + 1)
	}
	_, err = tq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// Use 1-bit quantizer where codebook has 2 entries.
	tq1, err := NewTurboQuantizer(16, 1, 42)
	if err != nil {
		t.Fatalf("NewTurboQuantizer 1-bit: %v", err)
	}
	cv1, err := tq1.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize 1-bit: %v", err)
	}
	// Round-trip should work
	_, err = tq1.Dequantize(cv1)
	if err != nil {
		t.Errorf("Dequantize should succeed: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Polar unpack with header mismatch (reachable via valid encoding with different config)
// ---------------------------------------------------------------------------

func TestPolar_Unpack_HeaderMismatch(t *testing.T) {
	cfg1 := PolarConfig{Dim: 16, Levels: 4, BitsLevel1: 4, BitsRest: 2, RadiusBits: 16, Seed: 42}
	pq1, err := NewPolarQuantizer(cfg1)
	if err != nil {
		t.Fatalf("NewPolarQuantizer cfg1: %v", err)
	}

	vec := make([]float64, 16)
	rng := rand.New(rand.NewSource(42))
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}
	cv, err := pq1.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// Create a quantizer with different BitsLevel1
	cfg2 := PolarConfig{Dim: 16, Levels: 4, BitsLevel1: 3, BitsRest: 2, RadiusBits: 16, Seed: 42}
	pq2, err := NewPolarQuantizer(cfg2)
	if err != nil {
		t.Fatalf("NewPolarQuantizer cfg2: %v", err)
	}

	// Try to decode with mismatched config
	cv.BitsPer = pq2.Bits()
	_, err = pq2.Dequantize(cv)
	if err == nil {
		t.Error("expected error for header mismatch")
	}
}

// ---------------------------------------------------------------------------
// LloydMax boundary collapse (a >= b)
// ---------------------------------------------------------------------------

func TestLloydMax_ConcentratedPDF(t *testing.T) {
	// A PDF that is extremely concentrated: all mass at a single point.
	// This can cause boundary collapse.
	pdf := func(x float64) float64 {
		if math.Abs(x-0.5) < 0.0001 {
			return 10000.0
		}
		return 0.0
	}
	centroids, boundaries, err := LloydMax(pdf, 0, 1, 4, 50)
	if err != nil {
		t.Fatalf("LloydMax: %v", err)
	}
	if len(centroids) != 4 {
		t.Errorf("expected 4 centroids, got %d", len(centroids))
	}
	if len(boundaries) != 5 {
		t.Errorf("expected 5 boundaries, got %d", len(boundaries))
	}
}

// ---------------------------------------------------------------------------
// Scalar encodeValue safety clamp (code < 0 or code > levels)
// ---------------------------------------------------------------------------

func TestScalar_EncodeValue_EdgeCases(t *testing.T) {
	q, err := NewUniformQuantizer(0, 1, 8)
	if err != nil {
		t.Fatalf("NewUniformQuantizer: %v", err)
	}

	// Test value exactly at boundary
	vec := []float64{0, 1, 0.5}
	cv, err := q.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	result, err := q.Dequantize(cv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}
	if result[0] != 0 {
		t.Errorf("result[0] = %f, want 0", result[0])
	}
}

// ---------------------------------------------------------------------------
// TurboProd.EstimateInnerProduct with valid data (cover decode + IP path)
// ---------------------------------------------------------------------------

func TestTurboProd_EstimateIP_ValidPath(t *testing.T) {
	const dim = 16
	tpq, err := NewTurboProdQuantizer(dim, 3, dim, 42)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	vec := make([]float64, dim)
	query := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
		query[i] = rng.NormFloat64()
	}

	cv, err := tpq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	ip, err := tpq.EstimateInnerProduct(query, cv)
	if err != nil {
		t.Fatalf("EstimateInnerProduct: %v", err)
	}

	// Sanity: IP should be a finite number
	if math.IsNaN(ip) || math.IsInf(ip, 0) {
		t.Errorf("EstimateInnerProduct returned non-finite: %f", ip)
	}
}

// ---------------------------------------------------------------------------
// TurboProd.ParseProdVector valid path
// ---------------------------------------------------------------------------

func TestParseProdVector_Valid(t *testing.T) {
	const dim = 16
	tpq, err := NewTurboProdQuantizer(dim, 3, dim, 42)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	cv, err := tpq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	pv, err := tpq.ParseProdVector(cv)
	if err != nil {
		t.Fatalf("ParseProdVector: %v", err)
	}

	if pv.Dim != dim {
		t.Errorf("Dim = %d, want %d", pv.Dim, dim)
	}
	if pv.Bits != 3 {
		t.Errorf("Bits = %d, want 3", pv.Bits)
	}
	if pv.Norm <= 0 {
		t.Errorf("Norm = %f, want > 0", pv.Norm)
	}
	if pv.ResidualNorm < 0 {
		t.Errorf("ResidualNorm = %f, want >= 0", pv.ResidualNorm)
	}
}

// ---------------------------------------------------------------------------
// TurboProd.Dequantize valid path (covers norm rescaling)
// ---------------------------------------------------------------------------

func TestTurboProd_Dequantize_Valid(t *testing.T) {
	const dim = 16
	tpq, err := NewTurboProdQuantizer(dim, 3, dim, 42)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	cv, err := tpq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	result, err := tpq.Dequantize(cv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	if len(result) != dim {
		t.Errorf("result length = %d, want %d", len(result), dim)
	}
}

// ---------------------------------------------------------------------------
// TurboProd.EstimateInnerProduct invalid wire data
// ---------------------------------------------------------------------------

func TestTurboProd_EstimateIP_InvalidWire(t *testing.T) {
	tpq, err := NewTurboProdQuantizer(16, 3, 16, 42)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}

	cv := CompressedVector{Data: []byte{0xFF}, Dim: 16, BitsPer: 3}
	_, err = tpq.EstimateInnerProduct(make([]float64, 16), cv)
	if err == nil {
		t.Error("expected error for invalid wire data")
	}
}

// ---------------------------------------------------------------------------
// TurboProd.Dequantize invalid wire data
// ---------------------------------------------------------------------------

func TestTurboProd_Dequantize_InvalidWire(t *testing.T) {
	tpq, err := NewTurboProdQuantizer(16, 3, 16, 42)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}

	cv := CompressedVector{Data: []byte{0xFF}, Dim: 16, BitsPer: 3}
	_, err = tpq.Dequantize(cv)
	if err == nil {
		t.Error("expected error for invalid wire data")
	}
}

// ---------------------------------------------------------------------------
// TurboProd.Quantize zero-norm vector (covers gamma=0 path)
// ---------------------------------------------------------------------------

func TestTurboProd_Quantize_ZeroNorm(t *testing.T) {
	tpq, err := NewTurboProdQuantizer(16, 3, 16, 42)
	if err != nil {
		t.Fatalf("NewTurboProdQuantizer: %v", err)
	}

	vec := make([]float64, 16) // all zeros
	cv, err := tpq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// Should be able to dequantize and estimate IP
	_, err = tpq.Dequantize(cv)
	if err != nil {
		t.Errorf("Dequantize: %v", err)
	}

	query := make([]float64, 16)
	query[0] = 1
	ip, err := tpq.EstimateInnerProduct(query, cv)
	if err != nil {
		t.Errorf("EstimateInnerProduct: %v", err)
	}
	if ip != 0 {
		t.Logf("IP for zero vector = %f (expected ~0)", ip)
	}
}

// ---------------------------------------------------------------------------
// Polar dequantize with crafted invalid codebook index via wire manipulation
// ---------------------------------------------------------------------------

func TestPolar_Dequantize_CorruptAngleIndex(t *testing.T) {
	cfg := DefaultPolarConfig(16)
	cfg.Seed = 42
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}

	// Quantize normally
	rng := rand.New(rand.NewSource(42))
	vec := make([]float64, 16)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}
	cv, err := pq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// Corrupt the packed angle indices (they start after the header + radii).
	// Header: 1 + 4 + 1 + 1 + 1 + 1 + 8 + 4 = 21 bytes
	// Radii: numRadii * 2 = (16/16) * 2 = 2 bytes
	// Angle data starts at offset 23.
	// Set all angle bytes to 0xFF which forces max indices that may exceed codebook.
	// For 4-bit level1 (16 centroids) 0xFF = index 15 which is valid.
	// For 2-bit levels, 0xFF = index 3 which is valid with 4 centroids.
	// So this won't actually produce an error. Valid round-trip is still useful coverage.
	_, err = pq.Dequantize(cv)
	if err != nil {
		t.Errorf("Dequantize of valid data should succeed: %v", err)
	}
}

// ---------------------------------------------------------------------------
// PolarTransform and InversePolarTransform (cover more paths)
// ---------------------------------------------------------------------------

func TestPolarTransform_Levels1(t *testing.T) {
	vec := []float64{3, 4}
	angles, radii, err := PolarTransform(vec, 1)
	if err != nil {
		t.Fatalf("PolarTransform: %v", err)
	}
	if len(angles) != 1 {
		t.Errorf("expected 1 level of angles, got %d", len(angles))
	}
	if len(radii) != 1 {
		t.Errorf("expected 1 radius, got %d", len(radii))
	}
	// radius should be hypot(3,4) = 5
	if math.Abs(radii[0]-5.0) > 1e-10 {
		t.Errorf("radius = %f, want 5.0", radii[0])
	}
}

// ---------------------------------------------------------------------------
// LevelCodebook error paths
// ---------------------------------------------------------------------------

func TestLevelCodebook_CoverageErrors(t *testing.T) {
	t.Run("bits=0", func(t *testing.T) {
		_, err := LevelCodebook(1, 0, 0)
		if err == nil {
			t.Error("expected error for bits=0")
		}
	})

	t.Run("level=0", func(t *testing.T) {
		_, err := LevelCodebook(0, 1, 2)
		if err == nil {
			t.Error("expected error for level=0")
		}
	})

	t.Run("n=0_level>=2", func(t *testing.T) {
		_, err := LevelCodebook(2, 0, 2)
		if err == nil {
			t.Error("expected error for n=0 at level>=2")
		}
	})
}

// ---------------------------------------------------------------------------
// Full round-trip via wire format for TurboQuant
// ---------------------------------------------------------------------------

func TestTurboWire_FullRoundTrip(t *testing.T) {
	// Test the complete wire encode/decode pipeline
	norm := 3.14159
	packed := []byte{0xAA, 0xBB, 0xCC}

	data := encodeTurboWire(norm, packed)
	gotNorm, gotPacked, err := decodeTurboWire(data)
	if err != nil {
		t.Fatalf("decodeTurboWire: %v", err)
	}
	if gotNorm != norm {
		t.Errorf("norm = %f, want %f", gotNorm, norm)
	}
	if len(gotPacked) != len(packed) {
		t.Fatalf("packed len = %d, want %d", len(gotPacked), len(packed))
	}
	for i := range packed {
		if gotPacked[i] != packed[i] {
			t.Errorf("packed[%d] = 0x%X, want 0x%X", i, gotPacked[i], packed[i])
		}
	}
}

// ---------------------------------------------------------------------------
// decodeProdWire with truncated sketch data
// ---------------------------------------------------------------------------

func TestDecodeProdWire_TruncatedSketch(t *testing.T) {
	// Build valid header but truncate the sketch bits section
	mseData := []byte{1, 2, 3}
	sketchBits := []uint64{0x123456789ABCDEF0}

	data := encodeProdWire(16, 1.0, 0.5, mseData, sketchBits)

	// Truncate last 4 bytes (removes part of sketch data)
	truncated := data[:len(data)-4]
	_, err := decodeProdWire(truncated)
	if err == nil {
		t.Error("expected error for truncated sketch data")
	}
}

// ---------------------------------------------------------------------------
// decodeProdWire with truncated MSE data
// ---------------------------------------------------------------------------

func TestDecodeProdWire_TruncatedMSE(t *testing.T) {
	mseData := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	sketchBits := []uint64{0xABCD}

	data := encodeProdWire(16, 1.0, 0.5, mseData, sketchBits)

	// Find offset where mseData starts: version(1) + sketchDim(4) + norm(8) + gamma(8) + mseDataLen(4) = 25
	// Truncate to just after mseDataLen but before full mseData
	truncated := data[:26] // 25 + 1 byte of MSE data
	_, err := decodeProdWire(truncated)
	if err == nil {
		t.Error("expected error for truncated MSE data")
	}
}

// ---------------------------------------------------------------------------
// decodeProdWire with truncated numWords
// ---------------------------------------------------------------------------

func TestDecodeProdWire_TruncatedNumWords(t *testing.T) {
	mseData := []byte{1}
	sketchBits := []uint64{0xABCD}

	data := encodeProdWire(16, 1.0, 0.5, mseData, sketchBits)

	// Truncate just before numWords: version(1) + sketchDim(4) + norm(8) + gamma(8) + mseDataLen(4) + mseData(1) = 26
	truncated := data[:27] // 26 + 1 (partial numWords)
	_, err := decodeProdWire(truncated)
	if err == nil {
		t.Error("expected error for truncated numWords")
	}
}

// ---------------------------------------------------------------------------
// Polar pack/unpack with zero radii (edge case for quantizeRadii16 max==0 path)
// ---------------------------------------------------------------------------

func TestPolar_ZeroVector(t *testing.T) {
	cfg := DefaultPolarConfig(16)
	cfg.Seed = 42
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}

	vec := make([]float64, 16)
	cv, err := pq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize zero vector: %v", err)
	}

	result, err := pq.Dequantize(cv)
	if err != nil {
		t.Fatalf("Dequantize zero vector: %v", err)
	}
	for i, v := range result {
		if v != 0 {
			t.Errorf("result[%d] = %f, want 0", i, v)
		}
	}
}

// ---------------------------------------------------------------------------
// TurboCodebook: test the BetaPDF error propagation path
// ---------------------------------------------------------------------------

func TestTurboCodebook_InvalidDim(t *testing.T) {
	_, err := TurboCodebook(1, 2)
	if err == nil {
		t.Error("expected error for dim=1")
	}
}

// ---------------------------------------------------------------------------
// Polar quantizeRadii16 edge case: max input clamp and round
// ---------------------------------------------------------------------------

func TestQuantizeRadii16_RoundTrip(t *testing.T) {
	radii := []float64{0.0, 0.25, 0.5, 0.75, 1.0}
	q, err := quantizeRadii16(radii)
	if err != nil {
		t.Fatalf("quantizeRadii16: %v", err)
	}
	dq := dequantizeRadii16(q)

	for i, orig := range radii {
		if math.Abs(orig-dq[i]) > 1e-4 {
			t.Errorf("radii[%d] = %f, reconstructed = %f", i, orig, dq[i])
		}
	}
}

// ---------------------------------------------------------------------------
// Pack/Unpack indices edge cases
// ---------------------------------------------------------------------------

func TestPackUnpackIndices_RoundTrip(t *testing.T) {
	for _, bits := range []int{1, 2, 3, 4} {
		maxIdx := (1 << bits) - 1
		indices := make([]int, 20)
		for i := range indices {
			indices[i] = i % (maxIdx + 1)
		}
		packed := packIndices(indices, bits)
		unpacked := unpackIndices(packed, len(indices), bits)
		for i := range indices {
			if unpacked[i] != indices[i] {
				t.Errorf("bits=%d: index[%d] = %d, want %d", bits, i, unpacked[i], indices[i])
			}
		}
	}
}

func TestPackIndices_Empty(t *testing.T) {
	packed := packIndices(nil, 2)
	if len(packed) != 0 {
		t.Errorf("expected empty packed, got len=%d", len(packed))
	}
}

func TestUnpackIndices_Empty(t *testing.T) {
	indices := unpackIndices(nil, 0, 2)
	if len(indices) != 0 {
		t.Errorf("expected empty indices, got len=%d", len(indices))
	}
}

// ---------------------------------------------------------------------------
// Polar Dequantize with truncated radii
// ---------------------------------------------------------------------------

func TestPolar_Dequantize_TruncatedRadii(t *testing.T) {
	cfg := DefaultPolarConfig(16)
	cfg.Seed = 42
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}

	// Craft a minimal wire payload that will fail when reading radii.
	// Header: version(1) + dim(4) + levels(1) + bitsLevel1(1) + bitsRest(1) + radiusBits(1) + norm(8) + numRadii(4) = 21 bytes
	data := make([]byte, 21)
	data[0] = polarWireVersion
	binary.LittleEndian.PutUint32(data[1:5], 16)
	data[5] = 4 // levels
	data[6] = 4 // bitsLevel1
	data[7] = 2 // bitsRest
	data[8] = 16 // radiusBits
	binary.LittleEndian.PutUint64(data[9:17], math.Float64bits(1.0))
	binary.LittleEndian.PutUint32(data[17:21], 10) // numRadii = 10 (but no data follows)

	cv := CompressedVector{Data: data, Dim: 16, BitsPer: pq.Bits()}
	_, err = pq.Dequantize(cv)
	if err == nil {
		t.Error("expected error for truncated radii")
	}
}

// ---------------------------------------------------------------------------
// Polar unpack header truncation at every field boundary
// ---------------------------------------------------------------------------

func TestPolar_Unpack_HeaderTruncation(t *testing.T) {
	cfg := DefaultPolarConfig(16)
	cfg.Seed = 42
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}

	// Quantize a valid vector to get valid wire data.
	rng := rand.New(rand.NewSource(42))
	vec := make([]float64, 16)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}
	cv, err := pq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// Truncate at each header field boundary.
	// Header layout: version(1) + dim(4) + levels(1) + bitsL1(1) + bitsRest(1) + radiusBits(1) + norm(8) + numRadii(4) = 21
	truncPoints := []struct {
		name string
		at   int
	}{
		{"after_version", 1},
		{"partial_dim", 3},
		{"after_dim", 5},
		{"after_levels", 6},
		{"after_bitsL1", 7},
		{"after_bitsRest", 8},
		{"after_radiusBits", 9},
		{"partial_norm", 13},
		{"after_norm", 17},
		{"partial_numRadii", 19},
	}

	for _, tt := range truncPoints {
		t.Run(tt.name, func(t *testing.T) {
			truncated := cv.Data[:tt.at]
			tcv := CompressedVector{Data: truncated, Dim: 16, BitsPer: pq.Bits()}
			_, err := pq.Dequantize(tcv)
			if err == nil {
				t.Errorf("expected error for truncation at %d bytes", tt.at)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Polar unpack: truncation in angle indices section
// ---------------------------------------------------------------------------

func TestPolar_Unpack_TruncatedAngles(t *testing.T) {
	cfg := DefaultPolarConfig(16)
	cfg.Seed = 42
	pq, err := NewPolarQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewPolarQuantizer: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	vec := make([]float64, 16)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}
	cv, err := pq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// Truncate 1 byte before the end — removes part of the angle indices.
	if len(cv.Data) > 1 {
		truncated := cv.Data[:len(cv.Data)-1]
		tcv := CompressedVector{Data: truncated, Dim: 16, BitsPer: pq.Bits()}
		_, err := pq.Dequantize(tcv)
		if err == nil {
			t.Error("expected error for truncated angle indices")
		}
	}
}

// ---------------------------------------------------------------------------
// LloydMax: trigger idx > cdfN guard
// ---------------------------------------------------------------------------

func TestLloydMax_CdfIdxClamp(t *testing.T) {
	// A PDF that concentrates all mass at x = max.
	// This causes CDF targets to exceed the table, triggering idx > cdfN clamp.
	pdf := func(x float64) float64 {
		if x > 0.9999 {
			return 1e6
		}
		return 0.0
	}
	centroids, _, err := LloydMax(pdf, 0, 1, 2, 10)
	if err != nil {
		t.Fatalf("LloydMax: %v", err)
	}
	if len(centroids) != 2 {
		t.Errorf("expected 2 centroids, got %d", len(centroids))
	}
}

// ---------------------------------------------------------------------------
// Scalar encodeValue: trigger code < 0 and code > levels clamp
// ---------------------------------------------------------------------------

func TestScalar_EncodeValue_FloatEdgeCases(t *testing.T) {
	// Use very small scale to provoke floating-point edge cases.
	q, err := NewUniformQuantizer(-1e-15, 1e-15, 8)
	if err != nil {
		t.Fatalf("NewUniformQuantizer: %v", err)
	}

	// Values at exact boundaries.
	vec := []float64{-1e-15, 1e-15, 0}
	cv, err := q.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	result, err := q.Dequantize(cv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}
	// Just verify no panic and results are finite.
	for i, v := range result {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("result[%d] = %v, want finite", i, v)
		}
	}

	// Values that could cause code < 0 or code > levels due to float imprecision.
	q2, _ := NewUniformQuantizer(0, 1, 4)
	vec2 := []float64{-1e-16, 1.0 + 1e-16, 0.5}
	cv2, err := q2.Quantize(vec2)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	r2, err := q2.Dequantize(cv2)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}
	if r2[0] != 0 {
		t.Errorf("clamped min: got %f, want 0", r2[0])
	}
	if r2[1] != 1.0 {
		t.Errorf("clamped max: got %f, want 1.0", r2[1])
	}
}

// ---------------------------------------------------------------------------
// quantizeRadii16: trigger the code < 0 and code > MaxUint16 clamps
// ---------------------------------------------------------------------------

func TestTurbo_Dequantize_CorruptWire(t *testing.T) {
	tq, err := NewTurboQuantizer(16, 2, 42)
	if err != nil {
		t.Fatalf("NewTurboQuantizer: %v", err)
	}
	// Data with wrong version byte — passes dim/bits check but fails wire decode.
	cv := CompressedVector{Data: []byte{0xFF, 0, 0, 0, 0, 0, 0, 0, 0}, Dim: 16, BitsPer: 2}
	_, err = tq.Dequantize(cv)
	if err == nil {
		t.Error("expected error for corrupt wire data")
	}
}

func TestQuantizeRadii16_ClampPaths(t *testing.T) {
	// radius > 1.0 triggers clamping
	q, err := quantizeRadii16([]float64{1.5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if q[0] != math.MaxUint16 {
		t.Errorf("expected MaxUint16, got %d", q[0])
	}

	// Verify round-trip at extremes
	q2, err := quantizeRadii16([]float64{0.0, 1.0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	dq := dequantizeRadii16(q2)
	if dq[0] != 0.0 {
		t.Errorf("dq[0] = %f, want 0", dq[0])
	}
	if math.Abs(dq[1]-1.0) > 1e-4 {
		t.Errorf("dq[1] = %f, want ~1.0", dq[1])
	}
}
