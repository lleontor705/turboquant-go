package serial

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"testing"
)

// TestVersionRoundTrip verifies that writing then reading a version byte succeeds.
func TestVersionRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteVersion(&buf); err != nil {
		t.Fatalf("WriteVersion: %v", err)
	}
	if err := ReadVersion(&buf); err != nil {
		t.Fatalf("ReadVersion: %v", err)
	}
}

// TestUnknownVersion verifies that reading an unknown version byte returns
// ErrUnsupportedVersion.
func TestUnknownVersion(t *testing.T) {
	buf := bytes.NewBuffer([]byte{0x99})
	err := ReadVersion(buf)
	if !errors.Is(err, ErrUnsupportedVersion) {
		t.Fatalf("expected ErrUnsupportedVersion, got %v", err)
	}
}

// TestUint64SliceRoundTrip verifies that a []uint64 survives serialization.
func TestUint64SliceRoundTrip(t *testing.T) {
	original := []uint64{0xDEADBEEF, 0xCAFEBABE, 0x123456789ABCDEF0, 0, 0xFFFFFFFFFFFFFFFF}
	var buf bytes.Buffer
	if err := WriteUint64Slice(&buf, original); err != nil {
		t.Fatalf("WriteUint64Slice: %v", err)
	}
	got, err := ReadUint64Slice(&buf)
	if err != nil {
		t.Fatalf("ReadUint64Slice: %v", err)
	}
	if len(got) != len(original) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(original))
	}
	for i := range original {
		if got[i] != original[i] {
			t.Errorf("element %d: got 0x%X, want 0x%X", i, got[i], original[i])
		}
	}
}

// TestFloat64SliceRoundTrip verifies that a []float64 survives serialization.
func TestFloat64SliceRoundTrip(t *testing.T) {
	original := []float64{3.14, -2.71, 0.0, 1e308, -1e308, math.Inf(1), math.Inf(-1)}
	var buf bytes.Buffer
	if err := WriteFloat64Slice(&buf, original); err != nil {
		t.Fatalf("WriteFloat64Slice: %v", err)
	}
	got, err := ReadFloat64Slice(&buf)
	if err != nil {
		t.Fatalf("ReadFloat64Slice: %v", err)
	}
	if len(got) != len(original) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(original))
	}
	for i := range original {
		if got[i] != original[i] {
			t.Errorf("element %d: got %v, want %v", i, got[i], original[i])
		}
	}
}

// TestTruncatedData verifies that reading from truncated input returns
// an error (io.ErrUnexpectedEOF or io.EOF).
func TestTruncatedData(t *testing.T) {
	// Write a uint64 slice then truncate the output.
	original := []uint64{1, 2, 3, 4}
	var buf bytes.Buffer
	if err := WriteUint64Slice(&buf, original); err != nil {
		t.Fatalf("WriteUint64Slice: %v", err)
	}
	fullData := buf.Bytes()

	// Truncate: keep only the length prefix (4 bytes) + 5 bytes (partial first uint64).
	truncated := fullData[:9]
	_, err := ReadUint64Slice(bytes.NewReader(truncated))
	if err == nil {
		t.Fatal("expected error reading truncated uint64 data, got nil")
	}

	// Also test truncated float64 slice.
	buf.Reset()
	floatOriginal := []float64{1.0, 2.0}
	if err := WriteFloat64Slice(&buf, floatOriginal); err != nil {
		t.Fatalf("WriteFloat64Slice: %v", err)
	}
	floatFullData := buf.Bytes()
	floatTruncated := floatFullData[:9] // length prefix + partial first float64
	_, err = ReadFloat64Slice(bytes.NewReader(floatTruncated))
	if err == nil {
		t.Fatal("expected error reading truncated float64 data, got nil")
	}

	// Test truncated version read (empty reader).
	err = ReadVersion(bytes.NewReader(nil))
	if err == nil {
		t.Fatal("expected error reading version from empty reader, got nil")
	}
}

// TestEmptySlices verifies that empty []uint64 and []float64 round-trip correctly.
func TestEmptySlices(t *testing.T) {
	// Empty uint64 slice.
	var buf bytes.Buffer
	if err := WriteUint64Slice(&buf, nil); err != nil {
		t.Fatalf("WriteUint64Slice(nil): %v", err)
	}
	got, err := ReadUint64Slice(&buf)
	if err != nil {
		t.Fatalf("ReadUint64Slice: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected empty uint64 slice, got %d elements", len(got))
	}

	// Empty float64 slice.
	buf.Reset()
	if err := WriteFloat64Slice(&buf, nil); err != nil {
		t.Fatalf("WriteFloat64Slice(nil): %v", err)
	}
	gotF, err := ReadFloat64Slice(&buf)
	if err != nil {
		t.Fatalf("ReadFloat64Slice: %v", err)
	}
	if len(gotF) != 0 {
		t.Fatalf("expected empty float64 slice, got %d elements", len(gotF))
	}

	// Also test with explicitly empty (non-nil) slices.
	buf.Reset()
	if err := WriteUint64Slice(&buf, []uint64{}); err != nil {
		t.Fatalf("WriteUint64Slice(empty): %v", err)
	}
	got, err = ReadUint64Slice(&buf)
	if err != nil {
		t.Fatalf("ReadUint64Slice: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected empty uint64 slice, got %d elements", len(got))
	}
}

// TestLargeSlice verifies that serialization works correctly with large slices
// (10000+ elements).
func TestLargeSlice(t *testing.T) {
	const n = 12345

	// Large uint64 slice.
	uintData := make([]uint64, n)
	for i := range uintData {
		uintData[i] = uint64(i) * 0x0102030405060708
	}
	var buf bytes.Buffer
	if err := WriteUint64Slice(&buf, uintData); err != nil {
		t.Fatalf("WriteUint64Slice large: %v", err)
	}
	gotUint, err := ReadUint64Slice(&buf)
	if err != nil {
		t.Fatalf("ReadUint64Slice large: %v", err)
	}
	if len(gotUint) != n {
		t.Fatalf("uint64 length: got %d, want %d", len(gotUint), n)
	}
	for i := range uintData {
		if gotUint[i] != uintData[i] {
			t.Fatalf("uint64[%d]: got 0x%X, want 0x%X", i, gotUint[i], uintData[i])
		}
	}

	// Large float64 slice.
	floatData := make([]float64, n)
	for i := range floatData {
		floatData[i] = float64(i) * 3.141592653589793
	}
	buf.Reset()
	if err := WriteFloat64Slice(&buf, floatData); err != nil {
		t.Fatalf("WriteFloat64Slice large: %v", err)
	}
	gotFloat, err := ReadFloat64Slice(&buf)
	if err != nil {
		t.Fatalf("ReadFloat64Slice large: %v", err)
	}
	if len(gotFloat) != n {
		t.Fatalf("float64 length: got %d, want %d", len(gotFloat), n)
	}
	for i := range floatData {
		if gotFloat[i] != floatData[i] {
			t.Fatalf("float64[%d]: got %v, want %v", i, gotFloat[i], floatData[i])
		}
	}
}

// TestLittleEndianExplicit verifies that the wire format is explicitly
// little-endian by checking the raw bytes.
func TestLittleEndianExplicit(t *testing.T) {
	// Write a single uint64 value and check the byte order.
	var buf bytes.Buffer
	if err := WriteUint64Slice(&buf, []uint64{0x0102030405060708}); err != nil {
		t.Fatalf("WriteUint64Slice: %v", err)
	}
	data := buf.Bytes()
	// data[0:4] = length prefix (1 as uint32 LE) = {1, 0, 0, 0}
	// data[4:12] = uint64 in LE = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01}

	// Verify length prefix.
	if got := binary.LittleEndian.Uint32(data[0:4]); got != 1 {
		t.Fatalf("length prefix: got %d, want 1", got)
	}
	// Verify uint64 bytes are little-endian.
	want := []byte{0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01}
	for i, b := range want {
		if data[4+i] != b {
			t.Fatalf("byte %d: got 0x%02X, want 0x%02X", i, data[4+i], b)
		}
	}

	// Verify float64 byte order too.
	buf.Reset()
	pi := math.Float64bits(math.Pi)
	if err := WriteFloat64Slice(&buf, []float64{math.Pi}); err != nil {
		t.Fatalf("WriteFloat64Slice: %v", err)
	}
	fdata := buf.Bytes()
	gotBits := binary.LittleEndian.Uint64(fdata[4:12])
	if gotBits != pi {
		t.Fatalf("float64 bits: got 0x%X, want 0x%X", gotBits, pi)
	}
}

// TestWriteReadVersionThenSlices simulates the full wire format:
// version + uint64 slice + float64 slice.
func TestWriteReadVersionThenSlices(t *testing.T) {
	var buf bytes.Buffer

	// Write: version, uint64 slice, float64 slice.
	if err := WriteVersion(&buf); err != nil {
		t.Fatalf("WriteVersion: %v", err)
	}
	uints := []uint64{42, 0xFF}
	if err := WriteUint64Slice(&buf, uints); err != nil {
		t.Fatalf("WriteUint64Slice: %v", err)
	}
	floats := []float64{1.5, -3.14}
	if err := WriteFloat64Slice(&buf, floats); err != nil {
		t.Fatalf("WriteFloat64Slice: %v", err)
	}

	// Read back.
	if err := ReadVersion(&buf); err != nil {
		t.Fatalf("ReadVersion: %v", err)
	}
	gotUints, err := ReadUint64Slice(&buf)
	if err != nil {
		t.Fatalf("ReadUint64Slice: %v", err)
	}
	gotFloats, err := ReadFloat64Slice(&buf)
	if err != nil {
		t.Fatalf("ReadFloat64Slice: %v", err)
	}

	// Verify.
	if len(gotUints) != len(uints) {
		t.Fatalf("uints length: got %d, want %d", len(gotUints), len(uints))
	}
	for i := range uints {
		if gotUints[i] != uints[i] {
			t.Errorf("uints[%d]: got %d, want %d", i, gotUints[i], uints[i])
		}
	}
	if len(gotFloats) != len(floats) {
		t.Fatalf("floats length: got %d, want %d", len(gotFloats), len(floats))
	}
	for i := range floats {
		if gotFloats[i] != floats[i] {
			t.Errorf("floats[%d]: got %v, want %v", i, gotFloats[i], floats[i])
		}
	}
}

// TestTruncatedLengthPrefix tests the case where even the length prefix is
// truncated.
func TestTruncatedLengthPrefix(t *testing.T) {
	// Only 2 bytes of length prefix (need 4).
	_, err := ReadUint64Slice(bytes.NewReader([]byte{0x01, 0x00}))
	if err == nil {
		t.Fatal("expected error for truncated length prefix")
	}
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("expected io.ErrUnexpectedEOF, got %v", err)
	}

	_, err = ReadFloat64Slice(bytes.NewReader([]byte{0x01, 0x00}))
	if err == nil {
		t.Fatal("expected error for truncated length prefix")
	}
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("expected io.ErrUnexpectedEOF, got %v", err)
	}
}

// failWriter is an io.Writer that succeeds for the first N bytes and then
// returns an error. This is used to exercise write-error paths inside the
// value-writing loops of WriteUint64Slice and WriteFloat64Slice.
type failWriter struct {
	remaining int
}

func (w *failWriter) Write(p []byte) (int, error) {
	if w.remaining <= 0 {
		return 0, fmt.Errorf("failWriter: intentional write error")
	}
	if len(p) <= w.remaining {
		w.remaining -= len(p)
		return len(p), nil
	}
	n := w.remaining
	w.remaining = 0
	return n, fmt.Errorf("failWriter: intentional write error")
}

// TestWriteUint64Slice_LengthPrefixError verifies that WriteUint64Slice
// propagates errors that occur while writing the length prefix.
func TestWriteUint64Slice_LengthPrefixError(t *testing.T) {
	data := []uint64{1, 2, 3}
	w := &failWriter{remaining: 0}
	err := WriteUint64Slice(w, data)
	if err == nil {
		t.Fatal("expected error from failWriter during length prefix write, got nil")
	}
}

// TestWriteUint64Slice_ValueError verifies that WriteUint64Slice propagates
// errors that occur while writing individual uint64 values (not the length prefix).
func TestWriteUint64Slice_ValueError(t *testing.T) {
	data := []uint64{1, 2, 3}
	// Allow the 4-byte length prefix to succeed, then fail on the first value write.
	w := &failWriter{remaining: 4}
	err := WriteUint64Slice(w, data)
	if err == nil {
		t.Fatal("expected error from failWriter during value writes, got nil")
	}
}

// TestWriteFloat64Slice_LengthPrefixError verifies that WriteFloat64Slice
// propagates errors that occur while writing the length prefix.
func TestWriteFloat64Slice_LengthPrefixError(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0}
	w := &failWriter{remaining: 0}
	err := WriteFloat64Slice(w, data)
	if err == nil {
		t.Fatal("expected error from failWriter during length prefix write, got nil")
	}
}

// TestWriteFloat64Slice_ValueError verifies that WriteFloat64Slice propagates
// errors that occur while writing individual float64 values (not the length prefix).
func TestWriteFloat64Slice_ValueError(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0}
	// Allow the 4-byte length prefix to succeed, then fail on the first value write.
	w := &failWriter{remaining: 4}
	err := WriteFloat64Slice(w, data)
	if err == nil {
		t.Fatal("expected error from failWriter during value writes, got nil")
	}
}

// TestReadVersion_UnexpectedEOF verifies that ReadVersion returns
// io.ErrUnexpectedEOF when io.ReadFull itself returns io.ErrUnexpectedEOF.
// This is different from a plain io.EOF (empty reader) — it occurs when the
// reader provides fewer bytes than requested but more than zero.
// For a 1-byte read, io.ReadFull returns io.EOF (not ErrUnexpectedEOF) on an
// empty reader, so we use a reader that returns ErrUnexpectedEOF directly.
func TestReadVersion_UnexpectedEOF(t *testing.T) {
	r := &unexpectedEOFReader{}
	err := ReadVersion(r)
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("expected io.ErrUnexpectedEOF, got %v", err)
	}
}

// unexpectedEOFReader is a reader that always returns io.ErrUnexpectedEOF.
type unexpectedEOFReader struct{}

func (r *unexpectedEOFReader) Read(p []byte) (int, error) {
	return 0, io.ErrUnexpectedEOF
}
