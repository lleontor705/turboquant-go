package quantize

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"

	"github.com/lleontor705/turboquant-go/rotate"
)

// TurboQuantizer implements the TurboQuant_mse algorithm:
// random rotation → nearest-centroid quantization using Beta distribution codebooks.
// Implements the Quantizer interface.
//
// Algorithm:
//
//	Encode: y = Π·x → idx_j = argmin_k |y_j - c_k| → pack indices
//	Decode: ỹ_j = c[idx_j] → x̃ = Π^T · ỹ
//
// Where Π is a random orthogonal matrix and c_k are Lloyd-Max centroids
// for the Beta distribution that arises after random rotation.
//
// The quantizer is safe for concurrent use without external synchronization;
// the rotation matrix is immutable after construction and each Quantize/Dequantize
// call uses local buffers.
type TurboQuantizer struct {
	dim      int
	bits     int        // bits per coordinate (1, 2, 3, or 4)
	rotation *mat.Dense // random orthogonal matrix (dim × dim)
	codebook []float64  // precomputed centroids
}

// turboWireVersion is the wire format version for TurboQuant_mse data.
const turboWireVersion byte = 1

// NewTurboQuantizer creates a TurboQuant_mse quantizer.
//
// dim: vector dimension (must be >= 2).
// bits: bits per coordinate (1, 2, 3, or 4).
// seed: random seed for rotation matrix (deterministic).
//
// Returns ErrInvalidConfig for invalid parameters.
func NewTurboQuantizer(dim int, bits int, seed int64) (*TurboQuantizer, error) {
	if dim < 2 {
		return nil, fmt.Errorf("%w: dim must be >= 2, got %d", ErrInvalidConfig, dim)
	}
	if bits < 1 || bits > 4 {
		return nil, fmt.Errorf("%w: bits must be 1-4, got %d", ErrInvalidConfig, bits)
	}

	// Generate random orthogonal matrix.
	// Cannot fail: dim >= 2 (validated above) and rng is non-nil.
	rng := rand.New(rand.NewSource(seed))
	rot, _ := rotate.RandomOrthogonal(dim, rng)

	// Get precomputed codebook for Beta(d) distribution.
	// Cannot fail: dim >= 2 and bits 1-4 are validated above.
	codebook, _ := TurboCodebook(dim, bits)

	return &TurboQuantizer{
		dim:      dim,
		bits:     bits,
		rotation: rot,
		codebook: codebook,
	}, nil
}

// Quantize encodes a vector using TurboQuant_mse.
//
// Steps: 1) Normalize input to unit norm  2) Apply random rotation
//
//  3. Quantize each coordinate to nearest centroid  4) Pack indices
func (tq *TurboQuantizer) Quantize(vec []float64) (CompressedVector, error) {
	if len(vec) != tq.dim {
		return CompressedVector{}, fmt.Errorf("%w: expected dim %d, got %d",
			ErrDimensionMismatch, tq.dim, len(vec))
	}

	// Check for NaN.
	for _, v := range vec {
		if math.IsNaN(v) {
			return CompressedVector{}, ErrNaNInput
		}
	}

	// Normalize to unit norm (required for Beta distribution assumption).
	unit, norm := normalizeUnit(vec)

	// Apply random rotation: y = Π · x
	rotated := rotateForward(tq.rotation, unit)

	// Quantize each coordinate: idx_j = argmin_k |y_j - c_k|
	indices := QuantizeWithCodebook(rotated, tq.codebook)

	// Pack indices into bytes and store wire format.
	packed := packIndices(indices, tq.bits)
	data := encodeTurboWire(norm, packed)

	return CompressedVector{
		Data:    data,
		Dim:     tq.dim,
		Min:     -1.0,
		Max:     1.0,
		BitsPer: tq.bits,
	}, nil
}

// Dequantize reconstructs a vector from compressed form.
//
// Steps: 1) Unpack indices  2) Look up centroids  3) Apply inverse rotation
func (tq *TurboQuantizer) Dequantize(cv CompressedVector) ([]float64, error) {
	if cv.Dim != tq.dim {
		return nil, fmt.Errorf("%w: expected dim %d, got %d",
			ErrDimensionMismatch, tq.dim, cv.Dim)
	}
	if cv.BitsPer != tq.bits {
		return nil, fmt.Errorf("%w: expected bits %d, got %d",
			ErrDimensionMismatch, tq.bits, cv.BitsPer)
	}

	// Decode wire format and unpack indices.
	norm, packed, err := decodeTurboWire(cv.Data)
	if err != nil {
		return nil, fmt.Errorf("turbo: decode wire format failed: %w", err)
	}
	indices := unpackIndices(packed, tq.dim, tq.bits)

	// Look up centroids: ỹ_j = c[idx_j]
	// Index bounds are guaranteed by unpackIndices (bit-masked to [0, 2^bits-1]).
	centroids := make([]float64, tq.dim)
	for i, idx := range indices {
		centroids[i] = tq.codebook[idx]
	}

	// Apply inverse rotation: x̃ = Π^T · ỹ
	result := rotateInverse(tq.rotation, centroids)

	// Re-apply original norm.
	if norm != 1.0 {
		for i := range result {
			result[i] *= norm
		}
	}

	return result, nil
}

// Bits returns the bits per coordinate.
func (tq *TurboQuantizer) Bits() int {
	return tq.bits
}

// Dim returns the input dimension.
func (tq *TurboQuantizer) Dim() int {
	return tq.dim
}

// RotationMatrix returns a copy of the rotation matrix (for testing/serialization).
func (tq *TurboQuantizer) RotationMatrix() *mat.Dense {
	r, c := tq.rotation.Dims()
	clone := mat.NewDense(r, c, nil)
	clone.Copy(tq.rotation)
	return clone
}

// ---------------------------------------------------------------------------
// Bit packing helpers
// ---------------------------------------------------------------------------

// packIndices packs quantization indices (each using 'bits' bits) into []byte.
// MSB first: first index occupies the highest bits of the first byte.
// The last byte is zero-padded if the total number of bits is not a multiple of 8.
func packIndices(indices []int, bits int) []byte {
	if len(indices) == 0 {
		return []byte{}
	}

	totalBits := len(indices) * bits
	numBytes := (totalBits + 7) / 8
	data := make([]byte, numBytes)

	bitPos := 0 // current bit position (0 = highest bit of byte 0)
	for _, idx := range indices {
		// Place 'bits' bits of idx starting at bitPos (MSB-first order).
		for b := 0; b < bits; b++ {
			// Extract bit (bits-1-b) from idx (MSB first).
			bit := (idx >> (bits - 1 - b)) & 1
			byteIdx := bitPos / 8
			bitIdx := 7 - (bitPos % 8) // MSB = bit 7
			data[byteIdx] |= byte(bit << bitIdx)
			bitPos++
		}
	}

	return data
}

// unpackIndices reverses packIndices: extracts n indices from packed data,
// each using 'bits' bits, MSB first.
func unpackIndices(data []byte, n int, bits int) []int {
	if n == 0 {
		return []int{}
	}

	indices := make([]int, n)
	bitPos := 0

	for i := 0; i < n; i++ {
		val := 0
		for b := 0; b < bits; b++ {
			byteIdx := bitPos / 8
			bitIdx := 7 - (bitPos % 8)
			if byteIdx < len(data) {
				bit := int((data[byteIdx] >> bitIdx) & 1)
				val = (val << 1) | bit
			}
			bitPos++
		}
		indices[i] = val
	}

	return indices
}

// ---------------------------------------------------------------------------
// Wire format for TurboQuant_mse compressed data
// ---------------------------------------------------------------------------
//
// Layout (little-endian):
//
//	[version:1B] [norm:8B float64] [packed indices: bytes]

func encodeTurboWire(norm float64, packed []byte) []byte {
	var buf bytes.Buffer
	var tmp8 [8]byte

	buf.WriteByte(turboWireVersion)
	binary.LittleEndian.PutUint64(tmp8[:], math.Float64bits(norm))
	buf.Write(tmp8[:])
	buf.Write(packed)

	return buf.Bytes()
}

func decodeTurboWire(data []byte) (float64, []byte, error) {
	r := bytes.NewReader(data)
	ver, err := r.ReadByte()
	if err != nil {
		return 0, nil, fmt.Errorf("reading version: %w", err)
	}
	if ver != turboWireVersion {
		return 0, nil, fmt.Errorf("unsupported version %d", ver)
	}
	var tmp8 [8]byte
	if _, err := io.ReadFull(r, tmp8[:]); err != nil {
		return 0, nil, fmt.Errorf("reading norm: %w", err)
	}
	norm := math.Float64frombits(binary.LittleEndian.Uint64(tmp8[:]))
	packed := make([]byte, r.Len())
	// Cannot fail: we read exactly r.Len() bytes from r.
	io.ReadFull(r, packed) //nolint:errcheck
	return norm, packed, nil
}
