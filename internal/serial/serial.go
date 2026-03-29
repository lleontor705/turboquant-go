// Package serial provides versioned, little-endian binary serialization helpers
// for the TurboQuant wire format. All multi-byte values use binary.LittleEndian
// for cross-platform compatibility (amd64 ↔ arm64).
//
// Wire format overview:
//
//	[version byte] [field1] [field2] ...
//
// Variable-length fields use a uint32 length prefix followed by the raw
// little-endian encoded elements.
package serial

import (
	"encoding/binary"
	"errors"
	"io"
	"math"
)

// Version is the current wire-format protocol version.
const Version byte = 0x01

// ErrUnsupportedVersion indicates that the serialized data was written with
// a protocol version that this library does not recognize.
var ErrUnsupportedVersion = errors.New("serial: unsupported version")

// WriteVersion writes the protocol version byte to w.
func WriteVersion(w io.Writer) error {
	_, err := w.Write([]byte{Version})
	return err
}

// ReadVersion reads a single byte from r and validates that it matches the
// supported protocol version. Returns ErrUnsupportedVersion for unknown
// versions, or io.ErrUnexpectedEOF if no byte could be read.
func ReadVersion(r io.Reader) error {
	var buf [1]byte
	_, err := io.ReadFull(r, buf[:])
	if err != nil {
		if err == io.ErrUnexpectedEOF {
			return io.ErrUnexpectedEOF
		}
		return err
	}
	if buf[0] != Version {
		return ErrUnsupportedVersion
	}
	return nil
}

// WriteUint64Slice writes a length-prefixed []uint64 to w in little-endian
// byte order.
//
// Wire format: [uint32 length] [uint64 values...]
func WriteUint64Slice(w io.Writer, data []uint64) error {
	// Write length prefix as uint32 little-endian.
	var lenBuf [4]byte
	binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(data)))
	if _, err := w.Write(lenBuf[:]); err != nil {
		return err
	}

	// Write each uint64 value in little-endian order.
	var valBuf [8]byte
	for _, v := range data {
		binary.LittleEndian.PutUint64(valBuf[:], v)
		if _, err := w.Write(valBuf[:]); err != nil {
			return err
		}
	}
	return nil
}

// ReadUint64Slice reads a length-prefixed []uint64 from r in little-endian
// byte order.
//
// Returns io.ErrUnexpectedEOF if the data is truncated.
func ReadUint64Slice(r io.Reader) ([]uint64, error) {
	// Read length prefix.
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return nil, err
	}
	n := binary.LittleEndian.Uint32(lenBuf[:])

	if n == 0 {
		return nil, nil
	}

	data := make([]uint64, n)
	var valBuf [8]byte
	for i := uint32(0); i < n; i++ {
		if _, err := io.ReadFull(r, valBuf[:]); err != nil {
			return nil, err
		}
		data[i] = binary.LittleEndian.Uint64(valBuf[:])
	}
	return data, nil
}

// WriteFloat64Slice writes a length-prefixed []float64 to w in little-endian
// byte order.
//
// Wire format: [uint32 length] [float64 values...]
func WriteFloat64Slice(w io.Writer, data []float64) error {
	// Write length prefix as uint32 little-endian.
	var lenBuf [4]byte
	binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(data)))
	if _, err := w.Write(lenBuf[:]); err != nil {
		return err
	}

	// Write each float64 value in little-endian order.
	var valBuf [8]byte
	for _, v := range data {
		binary.LittleEndian.PutUint64(valBuf[:], math.Float64bits(v))
		if _, err := w.Write(valBuf[:]); err != nil {
			return err
		}
	}
	return nil
}

// ReadFloat64Slice reads a length-prefixed []float64 from r in little-endian
// byte order.
//
// Returns io.ErrUnexpectedEOF if the data is truncated.
func ReadFloat64Slice(r io.Reader) ([]float64, error) {
	// Read length prefix.
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return nil, err
	}
	n := binary.LittleEndian.Uint32(lenBuf[:])

	if n == 0 {
		return nil, nil
	}

	data := make([]float64, n)
	var valBuf [8]byte
	for i := uint32(0); i < n; i++ {
		if _, err := io.ReadFull(r, valBuf[:]); err != nil {
			return nil, err
		}
		data[i] = math.Float64frombits(binary.LittleEndian.Uint64(valBuf[:]))
	}
	return data, nil
}
