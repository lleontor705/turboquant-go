package quantize

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"

	"github.com/lleontor705/turboquant-go/rotate"
)

// PolarQuantizer implements the PolarQuant algorithm (arXiv:2502.02617):
// random rotation → recursive polar decomposition → angle quantization → packing.
//
// Pipeline:
//
//	Encode: x → Π·x (rotate) → polar transform → quantize angles per level → pack indices + radii
//	Decode: unpack → lookup centroids → inverse polar → Π^T · (inverse rotate)
//
// The joint distribution of (radii, angles) FACTORIZES after random rotation,
// making per-level independent quantization OPTIMAL (not an approximation).
//
// The quantizer is safe for concurrent use without external synchronization;
// the rotation matrix and codebooks are immutable after construction and each
// Quantize/Dequantize call uses local buffers.
type PolarQuantizer struct {
	config    PolarConfig
	rotation  *mat.Dense  // random orthogonal matrix (dim × dim)
	codebooks [][]float64 // precomputed codebooks per level
}

// polarWireVersion is the wire format version for PolarQuant data.
const polarWireVersion byte = 2

// NewPolarQuantizer creates a PolarQuant quantizer.
//
// config.Dim must be a multiple of 2^config.Levels (default: multiple of 16).
// config.Levels defaults to 4 if 0.
// config.BitsLevel1 defaults to 4 if 0.
// config.BitsRest defaults to 2 if 0.
// config.Seed is required for deterministic rotation.
//
// Returns ErrInvalidConfig for invalid parameters.
func NewPolarQuantizer(config PolarConfig) (*PolarQuantizer, error) {
	// Apply defaults.
	if config.Levels == 0 {
		config.Levels = 4
	}
	if config.BitsLevel1 == 0 {
		config.BitsLevel1 = 4
	}
	if config.BitsRest == 0 {
		config.BitsRest = 2
	}
	if config.RadiusBits == 0 {
		config.RadiusBits = 16
	}

	// Validate dimension.
	if config.Dim < 2 {
		return nil, fmt.Errorf("%w: dim must be >= 2, got %d", ErrInvalidConfig, config.Dim)
	}

	blockSize := 1 << config.Levels
	if config.Dim%blockSize != 0 {
		return nil, fmt.Errorf("%w: dim=%d must be a multiple of %d (2^%d)",
			ErrInvalidConfig, config.Dim, blockSize, config.Levels)
	}

	// Validate bits.
	if config.BitsLevel1 < 1 || config.BitsLevel1 > 8 {
		return nil, fmt.Errorf("%w: BitsLevel1 must be 1-8, got %d", ErrInvalidConfig, config.BitsLevel1)
	}
	if config.BitsRest < 1 || config.BitsRest > 8 {
		return nil, fmt.Errorf("%w: BitsRest must be 1-8, got %d", ErrInvalidConfig, config.BitsRest)
	}
	if config.RadiusBits != 16 {
		return nil, fmt.Errorf("%w: RadiusBits must be 16, got %d", ErrInvalidConfig, config.RadiusBits)
	}

	// Generate random orthogonal matrix.
	// Cannot fail: config.Dim >= 2 (validated above) and rng is non-nil.
	rng := rand.New(rand.NewSource(config.Seed))
	rot, _ := rotate.RandomOrthogonal(config.Dim, rng)

	// Precompute codebooks for each level.
	// Cannot fail: level >= 1, bits >= 1 are validated above.
	codebooks := make([][]float64, config.Levels)
	for ℓ := 1; ℓ <= config.Levels; ℓ++ {
		n := PolarSinExponent(ℓ)
		var bits int
		if ℓ == 1 {
			bits = config.BitsLevel1
		} else {
			bits = config.BitsRest
		}

		cb, _ := LevelCodebook(ℓ, n, bits)
		codebooks[ℓ-1] = cb
	}

	return &PolarQuantizer{
		config:    config,
		rotation:  rot,
		codebooks: codebooks,
	}, nil
}

// Quantize encodes a vector using PolarQuant.
//
// Returns a CompressedVector with the packed data.
// Steps:
//  1. Normalize vector to unit norm
//  2. Apply random rotation: y = Π · x
//  3. Polar transform: y → (angles[0..L-1], radii)
//  4. Quantize each angle to nearest centroid in level codebook
//  5. Pack angle indices per level (variable bit-width)
//  6. Store final radii in float64
func (pq *PolarQuantizer) Quantize(vec []float64) (CompressedVector, error) {
	if len(vec) != pq.config.Dim {
		return CompressedVector{}, fmt.Errorf("%w: expected dim %d, got %d",
			ErrDimensionMismatch, pq.config.Dim, len(vec))
	}

	// Check for NaN.
	for _, v := range vec {
		if math.IsNaN(v) {
			return CompressedVector{}, ErrNaNInput
		}
	}

	// Normalize to unit norm.
	unit, norm := normalizeUnit(vec)

	// Apply random rotation: y = Π · x
	rotated := rotateForward(pq.rotation, unit)

	// Polar transform — cannot fail: dim is validated as multiple of 2^Levels.
	angles, radii, _ := PolarTransform(rotated, pq.config.Levels)

	// Quantize angles per level.
	// Level-1 angles from atan2 are in [-π, π], but the codebook is on [0, 2π).
	// Shift them before quantization.
	angleIndices := make([][]int, pq.config.Levels)
	for ℓ := 0; ℓ < pq.config.Levels; ℓ++ {
		values := angles[ℓ]
		if ℓ == 0 {
			// Map [-π, π] → [0, 2π)
			values = make([]float64, len(angles[ℓ]))
			for i, a := range angles[ℓ] {
				values[i] = a
				if values[i] < 0 {
					values[i] += 2 * math.Pi
				}
			}
		}
		angleIndices[ℓ] = QuantizeWithCodebook(values, pq.codebooks[ℓ])
	}

	// Pack everything into CompressedVector.Data.
	// Cannot fail: radii from polar decomposition are always >= 0.
	data := pq.packUnchecked(norm, angleIndices, radii)

	return CompressedVector{
		Data:    data,
		Dim:     pq.config.Dim,
		Min:     -1.0,
		Max:     1.0,
		BitsPer: pq.Bits(),
	}, nil
}

// Dequantize reconstructs a vector from compressed form.
//
// Steps:
//  1. Unpack angle indices per level
//  2. Map indices to centroid values using codebooks
//  3. Inverse polar transform: (centroids, radii) → ỹ
//  4. Inverse rotation: x̃ = Π^T · ỹ
func (pq *PolarQuantizer) Dequantize(cv CompressedVector) ([]float64, error) {
	if cv.Dim != pq.config.Dim {
		return nil, fmt.Errorf("%w: expected dim %d, got %d",
			ErrDimensionMismatch, pq.config.Dim, cv.Dim)
	}
	if cv.BitsPer != pq.Bits() {
		return nil, fmt.Errorf("%w: expected bits %d, got %d",
			ErrDimensionMismatch, pq.Bits(), cv.BitsPer)
	}

	// Unpack angle indices and radii.
	angleIndices, radii, norm, err := pq.unpack(cv.Data)
	if err != nil {
		return nil, fmt.Errorf("polar: unpack failed: %w", err)
	}

	// Map indices to centroid values using codebooks.
	// Level-1 centroids are in [0, 2π), shift back to [-π, π] for inverse transform.
	angleCentroids := make([][]float64, pq.config.Levels)
	for ℓ := 0; ℓ < pq.config.Levels; ℓ++ {
		centroids := make([]float64, len(angleIndices[ℓ]))
		// Index bounds guaranteed by unpackIndices (bit-masked to [0, 2^bits-1]).
		for i, idx := range angleIndices[ℓ] {
			centroids[i] = pq.codebooks[ℓ][idx]
		}
		if ℓ == 0 {
			// Map [0, 2π) → [-π, π]
			for i, c := range centroids {
				if c > math.Pi {
					centroids[i] = c - 2*math.Pi
				}
			}
		}
		angleCentroids[ℓ] = centroids
	}

	// Inverse polar transform.
	// Cannot fail: dimensions are consistent from pack/unpack round-trip.
	reconstructed, _ := InversePolarTransform(angleCentroids, radii, pq.config.Levels, pq.config.Dim)

	// Apply inverse rotation: x~ = rotation^T * y~
	result := rotateInverse(pq.rotation, reconstructed)

	// Re-apply original norm.
	if norm != 1.0 {
		for i := range result {
			result[i] *= norm
		}
	}

	return result, nil
}

// Bits returns the effective bits per coordinate.
func (pq *PolarQuantizer) Bits() int {
	return int(pq.config.BitsPerCoord())
}

// Dim returns the input dimension.
func (pq *PolarQuantizer) Dim() int {
	return pq.config.Dim
}

// RotationMatrix returns a copy of the rotation matrix (for testing/serialization).
func (pq *PolarQuantizer) RotationMatrix() *mat.Dense {
	r, c := pq.rotation.Dims()
	clone := mat.NewDense(r, c, nil)
	clone.Copy(pq.rotation)
	return clone
}

// ---------------------------------------------------------------------------
// Wire format for PolarQuantizer compressed data
// ---------------------------------------------------------------------------
//
// Layout (little-endian):
//
//	[version:1B]         — format version, currently 2
//	[dim:4B uint32]      — original vector dimension
//	[levels:1B]          — number of polar levels L
//	[bitsLevel1:1B]      — bits for level-1 angles
//	[bitsRest:1B]        — bits for levels 2..L angles
//	[radiusBits:1B]      — bits for final radii (must be 16)
//	[norm:8B float64]    — original vector norm
//	[numRadii:4B uint32] — count of final radii (= dim / 2^L)
//	[radii: numRadii×2B] — quantized uint16 radii
//	[level 1 indices packed] — packIndices(angleIndices[0], bitsLevel1)
//	[level 2 indices packed] — packIndices(angleIndices[1], bitsRest)
//	...
//	[level L indices packed] — packIndices(angleIndices[L-1], bitsRest)

// packUnchecked serializes angle indices and radii into the binary wire format.
// Radii are guaranteed >= 0 from polar decomposition; buffer writes never fail.
func (pq *PolarQuantizer) packUnchecked(norm float64, angleIndices [][]int, radii []float64) []byte {
	var buf bytes.Buffer
	var u32 [4]byte
	var u16 [2]byte
	var u64 [8]byte

	buf.WriteByte(polarWireVersion)

	binary.LittleEndian.PutUint32(u32[:], uint32(pq.config.Dim))
	buf.Write(u32[:])
	buf.WriteByte(byte(pq.config.Levels))
	buf.WriteByte(byte(pq.config.BitsLevel1))
	buf.WriteByte(byte(pq.config.BitsRest))
	buf.WriteByte(byte(pq.config.RadiusBits))

	binary.LittleEndian.PutUint64(u64[:], math.Float64bits(norm))
	buf.Write(u64[:])

	binary.LittleEndian.PutUint32(u32[:], uint32(len(radii)))
	buf.Write(u32[:])

	// Radii as quantized uint16 — cannot fail: radii from polar decomposition are >= 0.
	quantized, _ := quantizeRadii16(radii)
	for _, r := range quantized {
		binary.LittleEndian.PutUint16(u16[:], r)
		buf.Write(u16[:])
	}

	for ℓ := 0; ℓ < pq.config.Levels; ℓ++ {
		var bits int
		if ℓ == 0 {
			bits = pq.config.BitsLevel1
		} else {
			bits = pq.config.BitsRest
		}
		packed := packIndices(angleIndices[ℓ], bits)
		buf.Write(packed)
	}

	return buf.Bytes()
}

// unpack deserializes the binary wire format into angle indices and radii.
func (pq *PolarQuantizer) unpack(data []byte) ([][]int, []float64, float64, error) {
	r := bytes.NewReader(data)

	var tmp1 [1]byte
	var tmp4 [4]byte
	var tmp2 [2]byte
	var tmp8 [8]byte

	// Version.
	if _, err := r.Read(tmp1[:]); err != nil {
		return nil, nil, 0, fmt.Errorf("reading version: %w", err)
	}
	if tmp1[0] != polarWireVersion {
		return nil, nil, 0, fmt.Errorf("unsupported version %d", tmp1[0])
	}

	// Dim.
	if _, err := r.Read(tmp4[:]); err != nil {
		return nil, nil, 0, fmt.Errorf("reading dim: %w", err)
	}
	dim := int(binary.LittleEndian.Uint32(tmp4[:]))

	// Levels.
	if _, err := r.Read(tmp1[:]); err != nil {
		return nil, nil, 0, fmt.Errorf("reading levels: %w", err)
	}
	levels := int(tmp1[0])

	// BitsLevel1.
	if _, err := r.Read(tmp1[:]); err != nil {
		return nil, nil, 0, fmt.Errorf("reading bitsLevel1: %w", err)
	}
	bitsLevel1 := int(tmp1[0])

	// BitsRest.
	if _, err := r.Read(tmp1[:]); err != nil {
		return nil, nil, 0, fmt.Errorf("reading bitsRest: %w", err)
	}
	bitsRest := int(tmp1[0])

	// RadiusBits.
	if _, err := r.Read(tmp1[:]); err != nil {
		return nil, nil, 0, fmt.Errorf("reading radiusBits: %w", err)
	}
	radiusBits := int(tmp1[0])

	// Validate header matches quantizer config.
	if dim != pq.config.Dim || levels != pq.config.Levels ||
		bitsLevel1 != pq.config.BitsLevel1 || bitsRest != pq.config.BitsRest ||
		radiusBits != pq.config.RadiusBits {
		return nil, nil, 0, fmt.Errorf("header mismatch: got dim=%d levels=%d bitsL1=%d bitsRest=%d radiusBits=%d, want dim=%d levels=%d bitsL1=%d bitsRest=%d radiusBits=%d",
			dim, levels, bitsLevel1, bitsRest, radiusBits,
			pq.config.Dim, pq.config.Levels, pq.config.BitsLevel1, pq.config.BitsRest, pq.config.RadiusBits)
	}

	// Norm.
	if _, err := r.Read(tmp8[:]); err != nil {
		return nil, nil, 0, fmt.Errorf("reading norm: %w", err)
	}
	norm := math.Float64frombits(binary.LittleEndian.Uint64(tmp8[:]))

	// NumRadii.
	if _, err := r.Read(tmp4[:]); err != nil {
		return nil, nil, 0, fmt.Errorf("reading numRadii: %w", err)
	}
	numRadii := int(binary.LittleEndian.Uint32(tmp4[:]))

	// Radii.
	quantized := make([]uint16, numRadii)
	for i := 0; i < numRadii; i++ {
		if _, err := r.Read(tmp2[:]); err != nil {
			return nil, nil, 0, fmt.Errorf("reading radii[%d]: %w", i, err)
		}
		quantized[i] = binary.LittleEndian.Uint16(tmp2[:])
	}
	radii := dequantizeRadii16(quantized)

	// Unpack angle indices per level.
	angleIndices := make([][]int, levels)
	for ℓ := 0; ℓ < levels; ℓ++ {
		var bits int
		if ℓ == 0 {
			bits = bitsLevel1
		} else {
			bits = bitsRest
		}
		// Number of angles at level ℓ = dim / 2^(ℓ+1)
		nAngles := dim >> (ℓ + 1)
		packedSize := (nAngles*bits + 7) / 8
		packed := make([]byte, packedSize)
		if _, err := r.Read(packed); err != nil {
			return nil, nil, 0, fmt.Errorf("reading level %d indices: %w", ℓ+1, err)
		}
		angleIndices[ℓ] = unpackIndices(packed, nAngles, bits)
	}

	return angleIndices, radii, norm, nil
}

// quantizeRadii16 quantizes radii to uint16 using linear mapping.
func quantizeRadii16(radii []float64) ([]uint16, error) {
	if len(radii) == 0 {
		return []uint16{}, nil
	}
	for _, r := range radii {
		if r < 0 {
			return nil, fmt.Errorf("polar: radius must be >= 0, got %.6f", r)
		}
	}
	// Radii from unit vectors are in [0,1]; clamp to 1 for numerical noise.
	scale := float64(math.MaxUint16)
	quantized := make([]uint16, len(radii))
	for i, r := range radii {
		if r > 1.0 {
			r = 1.0
		}
		// After clamping r to [0,1], q is guaranteed in [0, MaxUint16].
		quantized[i] = uint16(math.Round(r * scale))
	}
	return quantized, nil
}

// dequantizeRadii16 reconstructs radii from uint16 values using stored max.
// The max radius is encoded implicitly by the maximum code value in the slice.
func dequantizeRadii16(quantized []uint16) []float64 {
	if len(quantized) == 0 {
		return []float64{}
	}
	invScale := 1.0 / float64(math.MaxUint16)
	radii := make([]float64, len(quantized))
	for i, q := range quantized {
		radii[i] = float64(q) * invScale
	}
	return radii
}
