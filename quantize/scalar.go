package quantize

import "math"

// UniformQuantizer performs uniform scalar quantization on float64 vectors.
//
// Values are mapped to 2^bits equally-spaced code levels spanning [min, max].
// The quantizer is safe for concurrent use without external synchronization;
// all parameters are immutable after construction.
type UniformQuantizer struct {
	min    float64
	max    float64
	bits   int     // 4 or 8
	levels int     // 2^bits - 1 (maximum code value)
	scale  float64 // (max - min) / levels
}

// NewUniformQuantizer creates a uniform scalar quantizer.
//
// bits must be 4 or 8; min must be strictly less than max.
// Returns ErrInvalidConfig for unsupported bit widths or an inverted/degenerate range.
func NewUniformQuantizer(min, max float64, bits int) (*UniformQuantizer, error) {
	if bits != 4 && bits != 8 {
		return nil, ErrInvalidConfig
	}
	if min >= max {
		return nil, ErrInvalidConfig
	}

	levels := (1 << bits) - 1 // 15 for 4-bit, 255 for 8-bit
	scale := (max - min) / float64(levels)

	return &UniformQuantizer{
		min:    min,
		max:    max,
		bits:   bits,
		levels: levels,
		scale:  scale,
	}, nil
}

// Quantize encodes a float64 vector into a CompressedVector.
//
// Out-of-range finite values are clamped to [min, max].
// ±Inf values are clamped to the nearest boundary.
// NaN values cause an immediate ErrNaNInput return.
//
// For 4-bit quantization two dimensions are packed per byte (high nibble first,
// low nibble second). Odd-length vectors have the final low nibble zero-padded.
// For 8-bit quantization one dimension occupies one byte.
func (q *UniformQuantizer) Quantize(vec []float64) (CompressedVector, error) {
	dim := len(vec)

	// Scan for NaN before doing any work.
	for _, v := range vec {
		if math.IsNaN(v) {
			return CompressedVector{}, ErrNaNInput
		}
	}

	var data []byte
	if q.bits == 4 {
		dataLen := (dim + 1) / 2
		data = make([]byte, dataLen)
		for i := 0; i < dim; i++ {
			code := q.encodeValue(vec[i])
			if i%2 == 0 {
				data[i/2] = code << 4 // high nibble
			} else {
				data[i/2] |= code // low nibble
			}
		}
	} else {
		data = make([]byte, dim)
		for i, v := range vec {
			data[i] = q.encodeValue(v)
		}
	}

	return CompressedVector{
		Data:    data,
		Dim:     dim,
		Min:     q.min,
		Max:     q.max,
		BitsPer: q.bits,
	}, nil
}

// encodeValue maps a single float64 to a quantization code in [0, levels].
// Out-of-range values and ±Inf are clamped; the caller guarantees v is not NaN.
func (q *UniformQuantizer) encodeValue(v float64) byte {
	// Clamp to [min, max] — also handles ±Inf naturally.
	if v < q.min {
		v = q.min
	} else if v > q.max {
		v = q.max
	}

	code := math.Round((v - q.min) / q.scale)

	// Safety clamp against floating-point edge cases.
	if code < 0 {
		code = 0
	} else if code > float64(q.levels) {
		code = float64(q.levels)
	}

	return byte(code)
}

// Dequantize reconstructs a float64 vector from a CompressedVector.
//
// Returns ErrDimensionMismatch when the compressed data length does not match
// the declared dimension for the quantizer's bit width.
func (q *UniformQuantizer) Dequantize(cv CompressedVector) ([]float64, error) {
	// Validate data length against declared dimension.
	var expectedLen int
	if q.bits == 4 {
		expectedLen = (cv.Dim + 1) / 2
	} else {
		expectedLen = cv.Dim
	}
	if len(cv.Data) != expectedLen {
		return nil, ErrDimensionMismatch
	}

	result := make([]float64, cv.Dim)

	if q.bits == 4 {
		for i := 0; i < cv.Dim; i++ {
			var code byte
			if i%2 == 0 {
				code = cv.Data[i/2] >> 4 // high nibble
			} else {
				code = cv.Data[i/2] & 0x0F // low nibble
			}
			result[i] = q.min + float64(code)*q.scale
		}
	} else {
		for i := 0; i < cv.Dim; i++ {
			result[i] = q.min + float64(cv.Data[i])*q.scale
		}
	}

	return result, nil
}

// Bits returns the number of bits per element (4 or 8).
func (q *UniformQuantizer) Bits() int {
	return q.bits
}

// MaxError returns the maximum theoretical quantization error per element:
//
//	(max - min) / (2 * levels)
//
// where levels = 2^bits - 1.
func (q *UniformQuantizer) MaxError() float64 {
	return q.scale / 2.0
}
