// Program genfixtures generates golden test fixtures for the turboquant library.
// Run with: go run ./cmd/genfixtures
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/lleontor705/turboquant-go/internal/bits"
	"github.com/lleontor705/turboquant-go/quantize"
	"github.com/lleontor705/turboquant-go/rotate"
	"github.com/lleontor705/turboquant-go/sketch"
)

// ---- Bit Packing Fixtures ----

type BitPackCase struct {
	Name           string  `json:"name"`
	Input          []int8  `json:"input"`
	ExpectedPacked []uint6 `json:"expected_packed"`
	ExpectedDim    int     `json:"expected_dim"`
}

type BitPackFixture struct {
	Version int           `json:"version"`
	Cases   []BitPackCase `json:"cases"`
}

type uint6 uint64

func (u uint6) MarshalJSON() ([]byte, error) {
	return json.Marshal(uint64(u))
}

func genBitPacking() error {
	cases := []struct {
		name  string
		input []int8
	}{
		{"standard_8", []int8{1, -1, 1, 1, -1, -1, 1, -1}},
		{"all_positive_8", []int8{1, 1, 1, 1, 1, 1, 1, 1}},
		{"all_negative_8", []int8{-1, -1, -1, -1, -1, -1, -1, -1}},
		{"single_positive", []int8{1}},
		{"single_negative", []int8{-1}},
		{"alternating_16", []int8{1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1}},
		{"first_only_8", []int8{1, -1, -1, -1, -1, -1, -1, -1}},
		{"last_only_8", []int8{-1, -1, -1, -1, -1, -1, -1, 1}},
		{"65_elements", make65Elems()},
		{"64_all_positive", makeAllOnes(64)},
		{"128_alternating", makeAlternating(128)},
	}

	fixture := BitPackFixture{Version: 1}
	for _, c := range cases {
		packed, err := bits.Pack(c.input)
		if err != nil {
			return fmt.Errorf("Pack(%s): %w", c.name, err)
		}
		// Convert []uint64 to []uint6
		packed6 := make([]uint6, len(packed))
		for i, v := range packed {
			packed6[i] = uint6(v)
		}
		fixture.Cases = append(fixture.Cases, BitPackCase{
			Name:           c.name,
			Input:          c.input,
			ExpectedPacked: packed6,
			ExpectedDim:    len(c.input),
		})
	}

	return writeJSON("testdata/bitpacking.json", fixture)
}

func make65Elems() []int8 {
	v := make([]int8, 65)
	for i := range v {
		if i%2 == 0 {
			v[i] = 1
		} else {
			v[i] = -1
		}
	}
	return v
}

func makeAllOnes(n int) []int8 {
	v := make([]int8, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

func makeAlternating(n int) []int8 {
	v := make([]int8, n)
	for i := range v {
		if i%2 == 0 {
			v[i] = 1
		} else {
			v[i] = -1
		}
	}
	return v
}

// ---- FWHT Fixtures ----

type FWHTECase struct {
	Name     string    `json:"name"`
	Input    []float64 `json:"input"`
	Expected []float64 `json:"expected"`
}

type FWHTFixture struct {
	Version int         `json:"version"`
	Cases   []FWHTECase `json:"cases"`
}

func genFWHT() error {
	inputCases := []struct {
		name  string
		input []float64
	}{
		{"dim4", []float64{1, 2, 3, 4}},
		{"dim1", []float64{7.5}},
		{"dim2", []float64{1, 1}},
		{"dim2_neg", []float64{1, -1}},
		{"dim8", []float64{1, 0, 0, 0, 0, 0, 0, 0}},
		{"dim4_zeros", []float64{0, 0, 0, 0}},
		{"dim4_all_ones", []float64{1, 1, 1, 1}},
		{"dim8_mixed", []float64{1, -1, 2, -2, 3, -3, 4, -4}},
	}

	fixture := FWHTFixture{Version: 1}
	for _, c := range inputCases {
		x := make([]float64, len(c.input))
		copy(x, c.input)
		if err := rotate.FWHT(x); err != nil {
			return fmt.Errorf("FWHT(%s): %w", c.name, err)
		}
		fixture.Cases = append(fixture.Cases, FWHTECase{
			Name:     c.name,
			Input:    c.input,
			Expected: x,
		})
	}

	return writeJSON("testdata/fwht.json", fixture)
}

// ---- Scalar Quantize Fixtures ----

type ScalarQCase struct {
	Name           string    `json:"name"`
	Min            float64   `json:"min"`
	Max            float64   `json:"max"`
	Bits           int       `json:"bits"`
	Input          []float64 `json:"input"`
	ExpectedApprox []float64 `json:"expected_approx"`
}

type ScalarQFixture struct {
	Version int           `json:"version"`
	Cases   []ScalarQCase `json:"cases"`
}

func genScalarQuantize() error {
	configs := []struct {
		name  string
		min   float64
		max   float64
		bits  int
		input []float64
	}{
		{"8bit_basic", 0, 10, 8, []float64{0, 5, 10}},
		{"4bit_basic", -5, 5, 4, []float64{-5, 0, 5}},
		{"8bit_negative_range", -100, 0, 8, []float64{-100, -50, 0}},
		{"4bit_small_range", 0, 1, 4, []float64{0, 0.5, 1}},
		{"8bit_clamp_low", 0, 10, 8, []float64{-5, 0, 5}},
		{"8bit_clamp_high", 0, 10, 8, []float64{5, 10, 15}},
		{"8bit_single_value", 0, 100, 8, []float64{50}},
		{"4bit_all_levels", 0, 15, 4, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
		{"8bit_boundary_values", 0, 255, 8, []float64{0, 127.5, 255}},
		{"4bit_negative_values", -10, -5, 4, []float64{-10, -7.5, -5}},
		{"8bit_fine_range", 0, 1, 8, []float64{0, 0.25, 0.5, 0.75, 1}},
		{"4bit_odd_length", 0, 15, 4, []float64{3, 7, 11, 15, 0}},
	}

	fixture := ScalarQFixture{Version: 1}
	for _, c := range configs {
		q, err := quantize.NewUniformQuantizer(c.min, c.max, c.bits)
		if err != nil {
			return fmt.Errorf("NewUniformQuantizer(%s): %w", c.name, err)
		}
		cv, err := q.Quantize(c.input)
		if err != nil {
			return fmt.Errorf("Quantize(%s): %w", c.name, err)
		}
		deq, err := q.Dequantize(cv)
		if err != nil {
			return fmt.Errorf("Dequantize(%s): %w", c.name, err)
		}

		// Round to avoid floating-point noise in JSON
		expected := make([]float64, len(deq))
		for i, v := range deq {
			expected[i] = math.Round(v*1e10) / 1e10
		}

		fixture.Cases = append(fixture.Cases, ScalarQCase{
			Name:           c.name,
			Min:            c.min,
			Max:            c.max,
			Bits:           c.bits,
			Input:          c.input,
			ExpectedApprox: expected,
		})
	}

	return writeJSON("testdata/scalar_quantize.json", fixture)
}

// ---- QJL Sketch Fixtures ----

type QJLSketchCase struct {
	Name             string    `json:"name"`
	Dim              int       `json:"dim"`
	SketchDim        int       `json:"sketch_dim"`
	Seed             int64     `json:"seed"`
	OutlierK         int       `json:"outlier_k"`
	Input            []float64 `json:"input"`
	ExpectedBitCount int       `json:"expected_bit_count"` // number of +1 signs
	ExpectedDim      int       `json:"expected_dim"`
}

type QJLSketchFixture struct {
	Version int             `json:"version"`
	Cases   []QJLSketchCase `json:"cases"`
}

func genQJLSketch() error {
	configs := []struct {
		name      string
		dim       int
		sketchDim int
		seed      int64
		outlierK  int
		input     []float64
	}{
		{"zero_vector", 8, 4, 42, 0, []float64{0, 0, 0, 0, 0, 0, 0, 0}},
		{"unit_basis_0", 8, 4, 42, 0, []float64{1, 0, 0, 0, 0, 0, 0, 0}},
		{"all_ones", 8, 8, 42, 0, []float64{1, 1, 1, 1, 1, 1, 1, 1}},
		{"mixed_vector", 8, 4, 123, 0, []float64{1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.0, -1.0}},
		{"with_outliers", 8, 4, 42, 2, []float64{1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.0, -1.0}},
		{"large_vector", 16, 8, 99, 0, []float64{1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8}},
	}

	fixture := QJLSketchFixture{Version: 1}
	for _, c := range configs {
		sketcher, err := sketch.NewQJLSketcher(sketch.QJLOptions{
			Dim:       c.dim,
			SketchDim: c.sketchDim,
			Seed:      c.seed,
			OutlierK:  c.outlierK,
		})
		if err != nil {
			return fmt.Errorf("NewQJLSketcher(%s): %w", c.name, err)
		}

		bv, err := sketcher.Sketch(c.input)
		if err != nil {
			return fmt.Errorf("Sketch(%s): %w", c.name, err)
		}

		// Count positive bits
		bitCount := bits.PopCount(bv.Bits)

		fixture.Cases = append(fixture.Cases, QJLSketchCase{
			Name:             c.name,
			Dim:              c.dim,
			SketchDim:        c.sketchDim,
			Seed:             c.seed,
			OutlierK:         c.outlierK,
			Input:            c.input,
			ExpectedBitCount: bitCount,
			ExpectedDim:      bv.Dim,
		})
	}

	return writeJSON("testdata/qjl_sketch.json", fixture)
}

// ---- Helpers ----

func writeJSON(path string, v interface{}) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0644)
}

func main() {
	if err := genBitPacking(); err != nil {
		fmt.Fprintf(os.Stderr, "bitpacking: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Generated testdata/bitpacking.json")

	if err := genFWHT(); err != nil {
		fmt.Fprintf(os.Stderr, "fwht: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Generated testdata/fwht.json")

	if err := genScalarQuantize(); err != nil {
		fmt.Fprintf(os.Stderr, "scalar_quantize: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Generated testdata/scalar_quantize.json")

	if err := genQJLSketch(); err != nil {
		fmt.Fprintf(os.Stderr, "qjl_sketch: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Generated testdata/qjl_sketch.json")

	fmt.Println("All fixtures generated successfully.")
}
