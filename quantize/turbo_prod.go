package quantize

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/lleontor705/turboquant-go/internal/bits"
	"github.com/lleontor705/turboquant-go/sketch"
)

// prodWireVersion is the wire format version for TurboQuant_prod data.
const prodWireVersion byte = 1

// TurboProdQuantizer implements the TurboQuant_prod algorithm (arXiv:2504.19874):
// two-stage quantization: MSE quantization (b-1 bits) + QJL 1-bit sketch of residual.
// Provides UNBIASED inner product estimation.
//
// Pipeline:
//
//	Encode: x → MSE quantize(b-1 bits) → compute residual → QJL sketch residual(1 bit)
//	IP Estimate: ⟨y, x̂⟩ ≈ ⟨y, x̂_mse⟩ + γ · correction(y, residual_sketch)
//
// The quantizer is safe for concurrent use without external synchronization;
// the rotation matrix, projection matrix, and codebook are immutable after construction.
type TurboProdQuantizer struct {
	dim             int
	bits            int // total bits per coordinate (2, 3, or 4)
	sketchDim       int
	mseQuant        *TurboQuantizer
	projector       sketch.Projector // Gaussian projection for encoding + IP correction
	correctionScale float64          // precomputed: √(π·dim / (2·sketchDim²))
}

// ProdVector holds the decoded TurboQuant_prod compressed representation.
//
// This type is useful for inspecting the internal structure of a compressed
// vector. Use ParseProdVector to extract from a CompressedVector.
type ProdVector struct {
	// MSEData is the MSE-quantized sub-vector (b-1 bits per coordinate).
	MSEData CompressedVector
	// Residual is the 1-bit QJL sketch of the residual (x - x̂_mse).
	Residual sketch.BitVector
	// ResidualNorm is γ = ||x - x̂_mse||₂.
	ResidualNorm float64
	// Dim is the original vector dimension.
	Dim int
	// Bits is the effective bits per coordinate.
	Bits int
}

// NewTurboProdQuantizer creates a TurboQuant_prod quantizer.
//
// dim: vector dimension (must be >= 2).
// bits: total bits per coordinate (2, 3, or 4). Uses (bits-1) for MSE + 1 for QJL.
// sketchDim: dimension of QJL sketch (typically = dim for optimal quality, must be <= dim).
// seed: random seed for reproducibility.
//
// Returns ErrInvalidConfig for invalid parameters.
func NewTurboProdQuantizer(dim int, bits int, sketchDim int, seed int64) (*TurboProdQuantizer, error) {
	if dim < 2 {
		return nil, fmt.Errorf("%w: dim must be >= 2, got %d", ErrInvalidConfig, dim)
	}
	if bits < 2 || bits > 4 {
		return nil, fmt.Errorf("%w: bits must be 2, 3, or 4, got %d", ErrInvalidConfig, bits)
	}
	if sketchDim < 1 || sketchDim > dim {
		return nil, fmt.Errorf("%w: sketchDim must be in [1, dim=%d], got %d",
			ErrInvalidConfig, dim, sketchDim)
	}

	mseBits := bits - 1
	mseSeed := seed
	qjlSeed := seed ^ 0x5A5A5A5A5A5A5A5A // derive independent seed for QJL

	// Create MSE quantizer with (bits-1) bits.
	mseQuant, err := NewTurboQuantizer(dim, mseBits, mseSeed)
	if err != nil {
		return nil, fmt.Errorf("turbo_prod: MSE quantizer creation failed: %w", err)
	}

	// Create Gaussian projection for both encoding and IP correction.
	// The projection matrix S has shape sketchDim × dim with entries N(0, 1/dim).
	projector, err := sketch.NewGaussianProjection(dim, sketchDim, qjlSeed)
	if err != nil {
		return nil, fmt.Errorf("turbo_prod: projection creation failed: %w", err)
	}

	// Precompute correction scale: √(π·dim / (2·sketchDim²)).
	//
	// Derivation: for unbiased IP estimation, the correction term is
	//   γ · scale · dot(signs, S·query)
	// where scale = √(π·dim / (2·k²)) with k = sketchDim.
	//
	// Proof of unbiasedness:
	//   E_S[sign(S·r)_j · (S·y)_j] = √(2/π) · <r,y> / (γ·√dim)
	//   E_S[correction] = γ · scale · k · √(2/π) · <r,y>/(γ·√dim) = <r,y>
	//   E_S[estimate] = <y, x̂_mse> + <y, r> = <y, x>
	correctionScale := math.Sqrt(math.Pi * float64(dim) / (2.0 * float64(sketchDim*sketchDim)))

	return &TurboProdQuantizer{
		dim:             dim,
		bits:            bits,
		sketchDim:       sketchDim,
		mseQuant:        mseQuant,
		projector:       projector,
		correctionScale: correctionScale,
	}, nil
}

// Quantize encodes a vector using TurboQuant_prod.
//
// Steps:
//  1. Normalize input to unit norm
//  2. MSE quantize with (bits-1) bits → packed indices
//  3. MSE dequantize → x̂_mse reconstruction
//  4. Compute residual r = x_unit - x̂_mse, store γ = ||r||
//  5. Project residual through QJL matrix, sign-quantize → 1-bit sketch
//  6. Pack all data into CompressedVector wire format
func (tpq *TurboProdQuantizer) Quantize(vec []float64) (CompressedVector, error) {
	if len(vec) != tpq.dim {
		return CompressedVector{}, fmt.Errorf("%w: expected dim %d, got %d",
			ErrDimensionMismatch, tpq.dim, len(vec))
	}

	// Check for NaN.
	for _, v := range vec {
		if math.IsNaN(v) {
			return CompressedVector{}, ErrNaNInput
		}
	}

	// Normalize to unit norm.
	unit := make([]float64, tpq.dim)
	copy(unit, vec)
	norm := 0.0
	for _, v := range unit {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		invNorm := 1.0 / norm
		for i := range unit {
			unit[i] *= invNorm
		}
	}

	// MSE quantize.
	mseCV, err := tpq.mseQuant.Quantize(unit)
	if err != nil {
		return CompressedVector{}, fmt.Errorf("turbo_prod: MSE quantize failed: %w", err)
	}

	// MSE dequantize to get reconstruction.
	xHat, err := tpq.mseQuant.Dequantize(mseCV)
	if err != nil {
		return CompressedVector{}, fmt.Errorf("turbo_prod: MSE dequantize failed: %w", err)
	}

	// Compute residual: r = unit - x̂, and residual norm γ = ||r||.
	residual := make([]float64, tpq.dim)
	gamma := 0.0
	for i := range unit {
		residual[i] = unit[i] - xHat[i]
		gamma += residual[i] * residual[i]
	}
	gamma = math.Sqrt(gamma)

	// QJL: project residual, sign-quantize, and pack into bits.
	sketchBits, err := tpq.sketchResidual(residual)
	if err != nil {
		return CompressedVector{}, fmt.Errorf("turbo_prod: sketch residual failed: %w", err)
	}

	// Pack into CompressedVector wire format.
	data, err := encodeProdWire(tpq.sketchDim, gamma, mseCV.Data, sketchBits)
	if err != nil {
		return CompressedVector{}, fmt.Errorf("turbo_prod: wire encode failed: %w", err)
	}

	return CompressedVector{
		Data:    data,
		Dim:     tpq.dim,
		Min:     -1.0,
		Max:     1.0,
		BitsPer: tpq.bits,
	}, nil
}

// Dequantize reconstructs the MSE part only.
// The residual is a 1-bit sketch and cannot be inverted, so the reconstruction
// is x̂_mse — the MSE reconstruction of the unit-norm input.
func (tpq *TurboProdQuantizer) Dequantize(cv CompressedVector) ([]float64, error) {
	if cv.Dim != tpq.dim {
		return nil, fmt.Errorf("%w: expected dim %d, got %d",
			ErrDimensionMismatch, tpq.dim, cv.Dim)
	}
	if cv.BitsPer != tpq.bits {
		return nil, fmt.Errorf("%w: expected bits %d, got %d",
			ErrDimensionMismatch, tpq.bits, cv.BitsPer)
	}

	// Decode wire format.
	pd, err := decodeProdWire(cv.Data)
	if err != nil {
		return nil, fmt.Errorf("turbo_prod: decode failed: %w", err)
	}

	// Reconstruct MSE CompressedVector and dequantize.
	mseCV := CompressedVector{
		Data:    pd.mseData,
		Dim:     tpq.dim,
		Min:     -1.0,
		Max:     1.0,
		BitsPer: tpq.bits - 1,
	}
	return tpq.mseQuant.Dequantize(mseCV)
}

// EstimateInnerProduct estimates ⟨query, x⟩ from encoded representations.
//
// This is the key innovation of TurboQuant_prod — UNBIASED IP estimation:
//
//	⟨query, x̂⟩ ≈ ⟨query, x̂_mse⟩ + γ · scale · dot(signs, S·query)
//
// where:
//   - x̂_mse is the MSE reconstruction
//   - γ = ||x - x̂_mse|| is the stored residual norm
//   - signs = sign(S·(x - x̂_mse)) are the stored QJL sketch bits
//   - scale = √(π·dim / (2·sketchDim²)) is the precomputed correction factor
//   - S is the Gaussian projection matrix
//
// The estimator is unbiased: E_S[estimate] = ⟨query, x_unit⟩ where x_unit is
// the unit-norm version of the original input. For unit-norm query vectors,
// this equals the cosine similarity.
//
// The query is NOT normalized internally — pass unit-norm queries for cosine
// similarity estimation, or non-unit queries for raw inner product estimation.
func (tpq *TurboProdQuantizer) EstimateInnerProduct(query []float64, cv CompressedVector) (float64, error) {
	if len(query) != tpq.dim {
		return 0, fmt.Errorf("%w: query dim %d != quantizer dim %d",
			ErrDimensionMismatch, len(query), tpq.dim)
	}
	if cv.Dim != tpq.dim {
		return 0, fmt.Errorf("%w: cv dim %d != quantizer dim %d",
			ErrDimensionMismatch, cv.Dim, tpq.dim)
	}
	if cv.BitsPer != tpq.bits {
		return 0, fmt.Errorf("%w: cv bits %d != quantizer bits %d",
			ErrDimensionMismatch, cv.BitsPer, tpq.bits)
	}

	// Decode wire format.
	pd, err := decodeProdWire(cv.Data)
	if err != nil {
		return 0, fmt.Errorf("turbo_prod: decode failed: %w", err)
	}

	// MSE dequantize.
	mseCV := CompressedVector{
		Data:    pd.mseData,
		Dim:     tpq.dim,
		Min:     -1.0,
		Max:     1.0,
		BitsPer: tpq.bits - 1,
	}
	xHat, err := tpq.mseQuant.Dequantize(mseCV)
	if err != nil {
		return 0, fmt.Errorf("turbo_prod: MSE dequantize failed: %w", err)
	}

	// MSE inner product: dot(query, x̂_mse).
	ipMse := 0.0
	for i := range query {
		ipMse += query[i] * xHat[i]
	}

	// If residual norm is zero (perfect MSE quantization), no correction needed.
	if pd.gamma == 0 {
		return ipMse, nil
	}

	// Unpack sketch signs from packed uint64s.
	signs, err := bits.Unpack(pd.sketchBits, tpq.sketchDim)
	if err != nil {
		return 0, fmt.Errorf("turbo_prod: unpack signs failed: %w", err)
	}

	// Project query through the same QJL matrix: projQuery = S · query.
	projQuery, err := tpq.projector.Project(query)
	if err != nil {
		return 0, fmt.Errorf("turbo_prod: project query failed: %w", err)
	}

	// Compute correction: γ · correctionScale · dot(signs, projQuery).
	signDotProj := 0.0
	for i, s := range signs {
		signDotProj += float64(s) * projQuery[i]
	}
	correction := pd.gamma * tpq.correctionScale * signDotProj

	return ipMse + correction, nil
}

// ParseProdVector extracts the internal ProdVector representation from a
// CompressedVector produced by TurboProdQuantizer.Quantize.
//
// Useful for debugging and inspecting the compressed structure.
func (tpq *TurboProdQuantizer) ParseProdVector(cv CompressedVector) (ProdVector, error) {
	if cv.Dim != tpq.dim {
		return ProdVector{}, fmt.Errorf("%w: expected dim %d, got %d",
			ErrDimensionMismatch, tpq.dim, cv.Dim)
	}
	if cv.BitsPer != tpq.bits {
		return ProdVector{}, fmt.Errorf("%w: expected bits %d, got %d",
			ErrDimensionMismatch, tpq.bits, cv.BitsPer)
	}

	pd, err := decodeProdWire(cv.Data)
	if err != nil {
		return ProdVector{}, fmt.Errorf("turbo_prod: decode failed: %w", err)
	}

	return ProdVector{
		MSEData: CompressedVector{
			Data:    pd.mseData,
			Dim:     tpq.dim,
			Min:     -1.0,
			Max:     1.0,
			BitsPer: tpq.bits - 1,
		},
		Residual: sketch.BitVector{
			Bits: pd.sketchBits,
			Dim:  tpq.sketchDim,
		},
		ResidualNorm: pd.gamma,
		Dim:          tpq.dim,
		Bits:         tpq.bits,
	}, nil
}

// Bits returns total bits per coordinate.
func (tpq *TurboProdQuantizer) Bits() int { return tpq.bits }

// Dim returns the input dimension.
func (tpq *TurboProdQuantizer) Dim() int { return tpq.dim }

// SketchDim returns the QJL sketch dimension.
func (tpq *TurboProdQuantizer) SketchDim() int { return tpq.sketchDim }

// ---------------------------------------------------------------------------
// Internal: residual sketching
// ---------------------------------------------------------------------------

// sketchResidual projects the residual vector through the QJL matrix and
// returns the sign-quantized packed bits.
func (tpq *TurboProdQuantizer) sketchResidual(residual []float64) ([]uint64, error) {
	// Project: projected = S · residual.
	projected, err := tpq.projector.Project(residual)
	if err != nil {
		return nil, err
	}

	// Sign quantize: positive → +1, non-positive → -1.
	signs := make([]int8, tpq.sketchDim)
	for i, v := range projected {
		if v > 0 {
			signs[i] = 1
		} else {
			signs[i] = -1
		}
	}

	// Pack into uint64s.
	packed, err := bits.Pack(signs)
	if err != nil {
		return nil, err
	}
	return packed, nil
}

// ---------------------------------------------------------------------------
// Wire format for CompressedVector.Data
// ---------------------------------------------------------------------------
//
// All fields are little-endian:
//
//	[1:  version = prodWireVersion]
//	[4:  sketchDim as uint32]
//	[8:  residualNorm (gamma) as float64]
//	[4:  mseDataLen as uint32]
//	[mseDataLen: MSE packed index bytes]
//	[4:  numSketchWords as uint32]
//	[numSketchWords * 8: packed sketch uint64s]

// prodWireData holds the decoded wire format fields.
type prodWireData struct {
	sketchDim  int
	gamma      float64
	mseData    []byte
	sketchBits []uint64
}

func encodeProdWire(sketchDim int, gamma float64, mseData []byte, sketchBits []uint64) ([]byte, error) {
	buf := new(bytes.Buffer)
	var tmp4 [4]byte
	var tmp8 [8]byte

	// Version.
	if err := buf.WriteByte(prodWireVersion); err != nil {
		return nil, fmt.Errorf("turbo_prod: encode version: %w", err)
	}

	// SketchDim.
	binary.LittleEndian.PutUint32(tmp4[:], uint32(sketchDim))
	if _, err := buf.Write(tmp4[:]); err != nil {
		return nil, fmt.Errorf("turbo_prod: encode sketchDim: %w", err)
	}

	// Gamma (residual norm).
	binary.LittleEndian.PutUint64(tmp8[:], math.Float64bits(gamma))
	if _, err := buf.Write(tmp8[:]); err != nil {
		return nil, fmt.Errorf("turbo_prod: encode gamma: %w", err)
	}

	// MSE data with length prefix.
	binary.LittleEndian.PutUint32(tmp4[:], uint32(len(mseData)))
	if _, err := buf.Write(tmp4[:]); err != nil {
		return nil, fmt.Errorf("turbo_prod: encode mseDataLen: %w", err)
	}
	if _, err := buf.Write(mseData); err != nil {
		return nil, fmt.Errorf("turbo_prod: encode mseData: %w", err)
	}

	// Sketch bits (packed uint64s).
	numWords := uint32(len(sketchBits))
	binary.LittleEndian.PutUint32(tmp4[:], numWords)
	if _, err := buf.Write(tmp4[:]); err != nil {
		return nil, fmt.Errorf("turbo_prod: encode numSketchWords: %w", err)
	}
	for _, w := range sketchBits {
		binary.LittleEndian.PutUint64(tmp8[:], w)
		if _, err := buf.Write(tmp8[:]); err != nil {
			return nil, fmt.Errorf("turbo_prod: encode sketchBits: %w", err)
		}
	}

	return buf.Bytes(), nil
}

func decodeProdWire(data []byte) (prodWireData, error) {
	r := bytes.NewReader(data)
	var tmp4 [4]byte
	var tmp8 [8]byte

	// Version.
	ver, err := r.ReadByte()
	if err != nil {
		return prodWireData{}, fmt.Errorf("turbo_prod: decode version: %w", err)
	}
	if ver != prodWireVersion {
		return prodWireData{}, fmt.Errorf("turbo_prod: unsupported wire version %d", ver)
	}

	// SketchDim.
	if _, err := io.ReadFull(r, tmp4[:]); err != nil {
		return prodWireData{}, fmt.Errorf("turbo_prod: decode sketchDim: %w", err)
	}
	sketchDim := int(binary.LittleEndian.Uint32(tmp4[:]))

	// Gamma.
	if _, err := io.ReadFull(r, tmp8[:]); err != nil {
		return prodWireData{}, fmt.Errorf("turbo_prod: decode gamma: %w", err)
	}
	gamma := math.Float64frombits(binary.LittleEndian.Uint64(tmp8[:]))

	// MSE data.
	if _, err := io.ReadFull(r, tmp4[:]); err != nil {
		return prodWireData{}, fmt.Errorf("turbo_prod: decode mseDataLen: %w", err)
	}
	mseDataLen := int(binary.LittleEndian.Uint32(tmp4[:]))
	mseData := make([]byte, mseDataLen)
	if mseDataLen > 0 {
		if _, err := io.ReadFull(r, mseData); err != nil {
			return prodWireData{}, fmt.Errorf("turbo_prod: decode mseData: %w", err)
		}
	}

	// Sketch bits.
	if _, err := io.ReadFull(r, tmp4[:]); err != nil {
		return prodWireData{}, fmt.Errorf("turbo_prod: decode numSketchWords: %w", err)
	}
	numWords := int(binary.LittleEndian.Uint32(tmp4[:]))
	sketchBits := make([]uint64, numWords)
	for i := range sketchBits {
		if _, err := io.ReadFull(r, tmp8[:]); err != nil {
			return prodWireData{}, fmt.Errorf("turbo_prod: decode sketchBits: %w", err)
		}
		sketchBits[i] = binary.LittleEndian.Uint64(tmp8[:])
	}

	return prodWireData{
		sketchDim:  sketchDim,
		gamma:      gamma,
		mseData:    mseData,
		sketchBits: sketchBits,
	}, nil
}
