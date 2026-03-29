package quantize

import (
	"fmt"
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
	rng := rand.New(rand.NewSource(seed))
	rot, err := rotate.RandomOrthogonal(dim, rng)
	if err != nil {
		return nil, fmt.Errorf("turbo: failed to generate rotation: %w", err)
	}

	// Get precomputed codebook for Beta(d) distribution.
	codebook, err := TurboCodebook(dim, bits)
	if err != nil {
		return nil, fmt.Errorf("turbo: failed to get codebook: %w", err)
	}

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
	unit := make([]float64, tq.dim)
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

	// Apply random rotation: y = Π · x
	rotated := make([]float64, tq.dim)
	for i := 0; i < tq.dim; i++ {
		var sum float64
		for j := 0; j < tq.dim; j++ {
			sum += tq.rotation.At(i, j) * unit[j]
		}
		rotated[i] = sum
	}

	// Quantize each coordinate: idx_j = argmin_k |y_j - c_k|
	indices := QuantizeWithCodebook(rotated, tq.codebook)

	// Pack indices into bytes.
	data := packIndices(indices, tq.bits)

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

	// Unpack indices.
	indices := unpackIndices(cv.Data, tq.dim, tq.bits)

	// Look up centroids: ỹ_j = c[idx_j]
	centroids := make([]float64, tq.dim)
	for i, idx := range indices {
		if idx < 0 || idx >= len(tq.codebook) {
			return nil, fmt.Errorf("turbo: invalid index %d for codebook size %d",
				idx, len(tq.codebook))
		}
		centroids[i] = tq.codebook[idx]
	}

	// Apply inverse rotation: x̃ = Π^T · ỹ
	result := make([]float64, tq.dim)
	for i := 0; i < tq.dim; i++ {
		var sum float64
		for j := 0; j < tq.dim; j++ {
			// Π^T[i][j] = Π[j][i]
			sum += tq.rotation.At(j, i) * centroids[j]
		}
		result[i] = sum
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
