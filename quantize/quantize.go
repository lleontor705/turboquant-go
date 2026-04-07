// Package quantize provides vector quantization interfaces and types for
// compressing float64 vectors into compact integer codes.
//
// The core abstraction is the Quantizer interface, which supports encoding
// (Quantize) and decoding (Dequantize) of vectors with configurable bit
// widths. All implementations are safe for concurrent use without external
// synchronization.
package quantize

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/lleontor705/turboquant-go/internal/serial"
)

// Quantizer compresses and decompresses float64 vectors.
//
// Implementations must be safe for concurrent use. Constructor-initialized
// parameters (scale, offset, min, max) must be immutable after construction.
type Quantizer interface {
	// Quantize encodes a float64 vector into a compressed representation.
	//
	// The input vector length must match the dimension the quantizer was
	// constructed for. Out-of-range finite values are clamped to [min, max].
	// NaN values are rejected with ErrNaNInput.
	Quantize(vec []float64) (CompressedVector, error)

	// Dequantize decodes a compressed vector back to float64 values.
	//
	// The returned vector has length equal to cv.Dim. Round-trip error is
	// bounded by the quantizer's MaxError().
	Dequantize(cv CompressedVector) ([]float64, error)

	// Bits returns the number of bits used per element (e.g. 4 or 8).
	Bits() int
}

// CompressedVector holds a quantized vector with metadata.
//
// For 4-bit quantization, two dimensions are packed per byte (high nibble =
// first dimension, low nibble = second dimension). For odd-length vectors,
// the final nibble of the last byte is zero-padded.
//
// CompressedVector implements encoding.BinaryMarshaler and
// encoding.BinaryUnmarshaler for cross-platform serialization using a
// little-endian wire format with a version byte prefix.
type CompressedVector struct {
	// Data contains the packed quantization codes.
	Data []byte
	// Dim is the original vector dimension.
	Dim int
	// Min is the lower bound of the quantization range.
	Min float64
	// Max is the upper bound of the quantization range.
	Max float64
	// BitsPer is the number of bits per element (4 or 8).
	BitsPer int
}

// MarshalBinary implements encoding.BinaryMarshaler.
//
// Wire format (little-endian):
//
//	[version:1B] [dim:uint32] [min:float64] [max:float64] [bitsPer:uint32] [data_len:uint32] [data:bytes]
func (cv CompressedVector) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	var u32 [4]byte
	var u64 [8]byte

	// bytes.Buffer.Write/WriteByte never return errors, so we skip error checks.
	buf.WriteByte(serial.Version)

	binary.LittleEndian.PutUint32(u32[:], uint32(cv.Dim))
	buf.Write(u32[:])

	binary.LittleEndian.PutUint64(u64[:], math.Float64bits(cv.Min))
	buf.Write(u64[:])

	binary.LittleEndian.PutUint64(u64[:], math.Float64bits(cv.Max))
	buf.Write(u64[:])

	binary.LittleEndian.PutUint32(u32[:], uint32(cv.BitsPer))
	buf.Write(u32[:])

	binary.LittleEndian.PutUint32(u32[:], uint32(len(cv.Data)))
	buf.Write(u32[:])
	buf.Write(cv.Data)

	return buf.Bytes(), nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler.
//
// It decodes data produced by MarshalBinary, fully replacing all fields of cv.
func (cv *CompressedVector) UnmarshalBinary(data []byte) error {
	r := bytes.NewReader(data)

	// Version byte.
	if err := serial.ReadVersion(r); err != nil {
		return fmt.Errorf("quantize: unmarshal version: %w", err)
	}

	var tmp4 [4]byte
	var tmp8 [8]byte

	// Dim.
	if _, err := io.ReadFull(r, tmp4[:]); err != nil {
		return fmt.Errorf("quantize: unmarshal dim: %w", err)
	}
	cv.Dim = int(binary.LittleEndian.Uint32(tmp4[:]))

	// Min.
	if _, err := io.ReadFull(r, tmp8[:]); err != nil {
		return fmt.Errorf("quantize: unmarshal min: %w", err)
	}
	cv.Min = math.Float64frombits(binary.LittleEndian.Uint64(tmp8[:]))

	// Max.
	if _, err := io.ReadFull(r, tmp8[:]); err != nil {
		return fmt.Errorf("quantize: unmarshal max: %w", err)
	}
	cv.Max = math.Float64frombits(binary.LittleEndian.Uint64(tmp8[:]))

	// BitsPer.
	if _, err := io.ReadFull(r, tmp4[:]); err != nil {
		return fmt.Errorf("quantize: unmarshal bitsPer: %w", err)
	}
	cv.BitsPer = int(binary.LittleEndian.Uint32(tmp4[:]))

	// Data length.
	if _, err := io.ReadFull(r, tmp4[:]); err != nil {
		return fmt.Errorf("quantize: unmarshal data length: %w", err)
	}
	dataLen := binary.LittleEndian.Uint32(tmp4[:])

	// Data bytes.
	cv.Data = make([]byte, dataLen)
	if _, err := io.ReadFull(r, cv.Data); err != nil {
		return fmt.Errorf("quantize: unmarshal data: %w", err)
	}

	return nil
}

// Sentinel errors for the quantize package.
// All errors are defined as var values so callers can use errors.Is for
// matching, including through wrapped errors created with fmt.Errorf("%w", err).
var (
	// ErrInvalidConfig indicates that the quantizer construction parameters
	// are invalid (e.g. min >= max, unsupported bit width, or zero/negative dimensions).
	ErrInvalidConfig = errors.New("quantize: invalid configuration")

	// ErrDimensionMismatch indicates that the input vector length does not
	// match the expected dimension.
	ErrDimensionMismatch = errors.New("quantize: dimension mismatch")

	// ErrNaNInput indicates that the input vector contains one or more NaN
	// values, which cannot be quantized.
	ErrNaNInput = errors.New("quantize: NaN input")
)
