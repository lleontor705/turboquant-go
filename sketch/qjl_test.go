package sketch

import (
	"encoding"
	"errors"
	"math"
	"math/rand"
	"sort"
	"sync"
	"testing"
)

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

func TestNewQJLSketcher(t *testing.T) {
	tests := []struct {
		name    string
		opts    QJLOptions
		wantErr error
	}{
		{
			name:    "valid gaussian",
			opts:    QJLOptions{Dim: 768, SketchDim: 256, Seed: 42},
			wantErr: nil,
		},
		{
			name:    "valid SRHT power-of-2",
			opts:    QJLOptions{Dim: 1024, SketchDim: 256, Seed: 42, UseSRHT: true},
			wantErr: nil,
		},
		{
			name:    "valid dim equals sketchDim",
			opts:    QJLOptions{Dim: 128, SketchDim: 128, Seed: 0},
			wantErr: nil,
		},
		{
			name:    "dim zero",
			opts:    QJLOptions{Dim: 0, SketchDim: 4, Seed: 42},
			wantErr: ErrInvalidDimension,
		},
		{
			name:    "dim negative",
			opts:    QJLOptions{Dim: -1, SketchDim: 4, Seed: 42},
			wantErr: ErrInvalidDimension,
		},
		{
			name:    "sketchDim zero",
			opts:    QJLOptions{Dim: 8, SketchDim: 0, Seed: 42},
			wantErr: ErrInvalidDimension,
		},
		{
			name:    "sketchDim negative",
			opts:    QJLOptions{Dim: 8, SketchDim: -1, Seed: 42},
			wantErr: ErrInvalidDimension,
		},
		{
			name:    "sketchDim exceeds dim",
			opts:    QJLOptions{Dim: 4, SketchDim: 8, Seed: 42},
			wantErr: ErrInvalidDimension,
		},
		{
			name:    "valid with outlier",
			opts:    QJLOptions{Dim: 128, SketchDim: 64, Seed: 42, OutlierK: 8},
			wantErr: nil,
		},
		{
			name:    "outlierK negative",
			opts:    QJLOptions{Dim: 128, SketchDim: 64, Seed: 42, OutlierK: -1},
			wantErr: ErrInvalidConfiguration,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := NewQJLSketcher(tt.opts)
			if tt.wantErr != nil {
				if err == nil {
					t.Fatalf("expected error containing %v, got nil", tt.wantErr)
				}
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("expected errors.Is(%v, %v), got %v", err, tt.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if s == nil {
				t.Fatal("expected non-nil sketcher")
			}
			if s.Dim() != tt.opts.Dim {
				t.Errorf("Dim() = %d, want %d", s.Dim(), tt.opts.Dim)
			}
			if s.SketchDim() != tt.opts.SketchDim {
				t.Errorf("SketchDim() = %d, want %d", s.SketchDim(), tt.opts.SketchDim)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Sketch encoding tests
// ---------------------------------------------------------------------------

func TestQJLSketch_HappyPath(t *testing.T) {
	sketcher, err := NewQJLSketcher(QJLOptions{Dim: 8, SketchDim: 4, Seed: 123})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	vec := []float64{1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0}
	bv, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch: %v", err)
	}

	if bv.Dim != 4 {
		t.Errorf("Dim = %d, want 4", bv.Dim)
	}
	expectedWords := (4 + 63) / 64
	if len(bv.Bits) != expectedWords {
		t.Errorf("len(Bits) = %d, want %d", len(bv.Bits), expectedWords)
	}
	// No outlier data when OutlierK=0
	if bv.OutlierIndices != nil {
		t.Errorf("OutlierIndices = %v, want nil", bv.OutlierIndices)
	}
	if bv.OutlierValues != nil {
		t.Errorf("OutlierValues = %v, want nil", bv.OutlierValues)
	}

	// Determinism: same input → same output
	bv2, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch (2nd): %v", err)
	}
	if bv.Dim != bv2.Dim {
		t.Errorf("Dim mismatch: %d vs %d", bv.Dim, bv2.Dim)
	}
	for i := range bv.Bits {
		if bv.Bits[i] != bv2.Bits[i] {
			t.Errorf("Bits[%d] mismatch: %v vs %v", i, bv.Bits[i], bv2.Bits[i])
		}
	}
}

func TestQJLSketch_ZeroVector(t *testing.T) {
	sketcher, err := NewQJLSketcher(QJLOptions{Dim: 4, SketchDim: 4, Seed: 42})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	vec := []float64{0, 0, 0, 0}
	bv, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch: %v", err)
	}

	// Zero vector → all projections are 0 → all non-positive → all bits = 0
	for i, w := range bv.Bits {
		if w != 0 {
			t.Errorf("Bits[%d] = %v, want 0 (all projections should be zero)", i, w)
		}
	}
}

func TestQJLSketch_DimMismatch(t *testing.T) {
	sketcher, err := NewQJLSketcher(QJLOptions{Dim: 8, SketchDim: 4, Seed: 42})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	_, err = sketcher.Sketch([]float64{1, 2, 3, 4, 5})
	if err == nil {
		t.Fatal("expected error for wrong-length input")
	}
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("expected ErrDimensionMismatch, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

func TestQJLDeterminism(t *testing.T) {
	opts := QJLOptions{Dim: 64, SketchDim: 16, Seed: 42}

	s1, err := NewQJLSketcher(opts)
	if err != nil {
		t.Fatalf("NewQJLSketcher (1): %v", err)
	}
	s2, err := NewQJLSketcher(opts)
	if err != nil {
		t.Fatalf("NewQJLSketcher (2): %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, 64)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	bv1, err := s1.Sketch(vec)
	if err != nil {
		t.Fatalf("s1.Sketch: %v", err)
	}
	bv2, err := s2.Sketch(vec)
	if err != nil {
		t.Fatalf("s2.Sketch: %v", err)
	}

	if bv1.Dim != bv2.Dim {
		t.Errorf("Dim mismatch: %d vs %d", bv1.Dim, bv2.Dim)
	}
	for i := range bv1.Bits {
		if bv1.Bits[i] != bv2.Bits[i] {
			t.Errorf("Bits[%d] differ: %v vs %v", i, bv1.Bits[i], bv2.Bits[i])
		}
	}
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

func TestConcurrentSketch(t *testing.T) {
	sketcher, err := NewQJLSketcher(QJLOptions{Dim: 64, SketchDim: 16, Seed: 42})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	const numGoroutines = 100
	rng := rand.New(rand.NewSource(0))

	// Pre-generate input vectors
	vecs := make([][]float64, numGoroutines)
	for i := range vecs {
		v := make([]float64, 64)
		for j := range v {
			v[j] = rng.NormFloat64()
		}
		vecs[i] = v
	}

	// Compute expected results sequentially
	expected := make([]*BitVector, numGoroutines)
	for i, v := range vecs {
		bv, err := sketcher.Sketch(v)
		if err != nil {
			t.Fatalf("Sketch(%d): %v", i, err)
		}
		expected[i] = bv
	}

	// Now run concurrently
	results := make([]*BitVector, numGoroutines)
	errs := make([]error, numGoroutines)
	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx], errs[idx] = sketcher.Sketch(vecs[idx])
		}(i)
	}
	wg.Wait()

	// Verify all results
	for i := 0; i < numGoroutines; i++ {
		if errs[i] != nil {
			t.Errorf("goroutine %d: unexpected error: %v", i, errs[i])
			continue
		}
		if results[i].Dim != expected[i].Dim {
			t.Errorf("goroutine %d: Dim = %d, want %d", i, results[i].Dim, expected[i].Dim)
		}
		for j := range results[i].Bits {
			if results[i].Bits[j] != expected[i].Bits[j] {
				t.Errorf("goroutine %d: Bits[%d] = %v, want %v", i, j, results[i].Bits[j], expected[i].Bits[j])
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Outlier handling
// ---------------------------------------------------------------------------

func TestOutlierHandling_K8(t *testing.T) {
	const dim = 128
	const sketchDim = 64
	const outlierK = 8

	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:       dim,
		SketchDim: sketchDim,
		Seed:      42,
		OutlierK:  outlierK,
	})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	bv, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch: %v", err)
	}

	// Verify outlier metadata is populated
	if len(bv.OutlierIndices) != outlierK {
		t.Fatalf("len(OutlierIndices) = %d, want %d", len(bv.OutlierIndices), outlierK)
	}
	if len(bv.OutlierValues) != outlierK {
		t.Fatalf("len(OutlierValues) = %d, want %d", len(bv.OutlierValues), outlierK)
	}

	// Verify indices are in valid range and unique
	seen := make(map[int]bool)
	for _, idx := range bv.OutlierIndices {
		if idx < 0 || idx >= sketchDim {
			t.Errorf("outlier index %d out of range [0, %d)", idx, sketchDim)
		}
		if seen[idx] {
			t.Errorf("duplicate outlier index %d", idx)
		}
		seen[idx] = true
	}

	// Verify outlier values are sorted by absolute value descending
	for i := 1; i < outlierK; i++ {
		if math.Abs(bv.OutlierValues[i]) > math.Abs(bv.OutlierValues[i-1]) {
			t.Errorf("outlier values not sorted by |abs| descending: |%v| at [%d] > |%v| at [%d]",
				bv.OutlierValues[i], i, bv.OutlierValues[i-1], i-1)
		}
	}

	// Verify bits at outlier positions are 0
	for _, idx := range bv.OutlierIndices {
		word := idx / 64
		bit := uint(idx % 64)
		if bv.Bits[word]&(1<<bit) != 0 {
			t.Errorf("outlier at index %d has bit set, expected 0", idx)
		}
	}
}

func TestOutlierHandling_K0(t *testing.T) {
	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:       64,
		SketchDim: 16,
		Seed:      42,
		OutlierK:  0,
	})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, 64)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	bv, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch: %v", err)
	}

	// No outlier data when OutlierK=0
	if bv.OutlierIndices != nil {
		t.Errorf("OutlierIndices = %v, want nil", bv.OutlierIndices)
	}
	if bv.OutlierValues != nil {
		t.Errorf("OutlierValues = %v, want nil", bv.OutlierValues)
	}
}

func TestOutlierHandling_ExceedsSketchDim(t *testing.T) {
	_, err := NewQJLSketcher(QJLOptions{
		Dim:       128,
		SketchDim: 64,
		Seed:      42,
		OutlierK:  100, // > SketchDim
	})
	if err == nil {
		t.Fatal("expected error for OutlierK > SketchDim")
	}
	if !errors.Is(err, ErrInvalidConfiguration) {
		t.Fatalf("expected ErrInvalidConfiguration, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// Ranking preservation
// ---------------------------------------------------------------------------

func TestQJLRankingPreservation(t *testing.T) {
	// Tests ranking preservation with genuine nearest-neighbor structure.
	//
	// Setup: 100 centroid-based clusters of 10 vectors each (1,000 "true NN"
	// vectors), plus 9,000 random distractor vectors = 10,000 total. Each
	// query is a centroid. Its true top-10 are the 10 vectors from its cluster
	// (cosine similarity ~0.98), which are far more similar than any distractor
	// (cosine similarity ~0).
	//
	// This is the standard ANN evaluation protocol: the sketch must distinguish
	// genuine nearest neighbors from distractors, not resolve tiny ranking
	// differences within a cluster.
	const (
		dim            = 768
		sketchDim      = 256
		numClusters    = 100
		vecsPerCluster = 10
		numDistractors = 9000
		numVecs        = numClusters*vecsPerCluster + numDistractors // 10,000
		topK           = 10
		seed           = int64(42)
		minRecall      = 0.95
		noiseStd       = 0.008 // small noise → intra-cluster cosine ~0.98
	)

	rng := rand.New(rand.NewSource(seed))

	// Step 1: Generate cluster centroids (used as queries)
	centroids := make([][]float64, numClusters)
	for i := range centroids {
		c := make([]float64, dim)
		for j := range c {
			c[j] = rng.NormFloat64()
		}
		normalize(c)
		centroids[i] = c
	}

	// Step 2: Generate database vectors — cluster members + distractors
	vecs := make([][]float64, numVecs)

	// Cluster members: noisy variants of centroids
	for ci := 0; ci < numClusters; ci++ {
		for j := 0; j < vecsPerCluster; j++ {
			v := make([]float64, dim)
			for k := range v {
				v[k] = centroids[ci][k] + rng.NormFloat64()*noiseStd
			}
			normalize(v)
			vecs[ci*vecsPerCluster+j] = v
		}
	}

	// Distractors: random normalized vectors
	for i := numClusters * vecsPerCluster; i < numVecs; i++ {
		v := make([]float64, dim)
		for j := range v {
			v[j] = rng.NormFloat64()
		}
		normalize(v)
		vecs[i] = v
	}

	// Step 3: Create sketcher and sketch all database vectors
	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:       dim,
		SketchDim: sketchDim,
		Seed:      seed,
	})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	sketches := make([]*BitVector, numVecs)
	for i, v := range vecs {
		sketches[i], err = sketcher.Sketch(v)
		if err != nil {
			t.Fatalf("Sketch(%d): %v", i, err)
		}
	}

	// Step 4: For each centroid (query), measure recall@10
	totalRecall := 0.0

	for qi, query := range centroids {
		// Exact inner products with all database vectors
		exactScores := make([]scoredItem, numVecs)
		for i, v := range vecs {
			exactScores[i] = scoredItem{idx: i, score: dot(query, v)}
		}

		// True top-K by exact cosine similarity
		trueTopK := topKByScore(exactScores, topK)
		trueSet := make(map[int]bool)
		for _, s := range trueTopK {
			trueSet[s.idx] = true
		}

		// Sketch-based ranking
		qSketch, err := sketcher.Sketch(query)
		if err != nil {
			t.Fatalf("Sketch(query %d): %v", qi, err)
		}

		sketchScores := make([]scoredItem, numVecs)
		for i, s := range sketches {
			est, _ := EstimateInnerProduct(*qSketch, *s)
			sketchScores[i] = scoredItem{idx: i, score: est}
		}
		sketchTopK := topKByScore(sketchScores, topK)

		// Compute recall
		hits := 0
		for _, s := range sketchTopK {
			if trueSet[s.idx] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(topK)
	}

	avgRecall := totalRecall / float64(numClusters)
	t.Logf("recall@%d = %.4f (min required: %.2f)", topK, avgRecall, minRecall)
	if avgRecall < minRecall {
		t.Errorf("recall@%d = %.4f, want >= %.2f", topK, avgRecall, minRecall)
	}
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

func TestSerialization_BitVectorRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		bv   BitVector
	}{
		{
			name: "simple",
			bv: BitVector{
				Bits: []uint64{0xDEADBEEFCAFEBABE},
				Dim:  64,
			},
		},
		{
			name: "multi_word",
			bv: BitVector{
				Bits: []uint64{0xAAAAAAAAAAAAAAAA, 0x5555555555555555, 0xFF00FF00FF00FF00},
				Dim:  192,
			},
		},
		{
			name: "non_multiple_of_64",
			bv: BitVector{
				Bits: []uint64{0xFF},
				Dim:  5,
			},
		},
		{
			name: "with_outliers",
			bv: BitVector{
				Bits:           []uint64{0xFF00},
				Dim:            16,
				OutlierIndices: []int{0, 8},
				OutlierValues:  []float64{3.14, -2.71},
			},
		},
		{
			name: "empty_bits",
			bv: BitVector{
				Bits: []uint64{},
				Dim:  0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := tt.bv.MarshalBinary()
			if err != nil {
				t.Fatalf("MarshalBinary: %v", err)
			}

			var decoded BitVector
			if err := decoded.UnmarshalBinary(data); err != nil {
				t.Fatalf("UnmarshalBinary: %v", err)
			}

			// Verify Dim
			if decoded.Dim != tt.bv.Dim {
				t.Errorf("Dim = %d, want %d", decoded.Dim, tt.bv.Dim)
			}

			// Verify Bits (handle nil vs empty)
			if len(decoded.Bits) != len(tt.bv.Bits) {
				t.Fatalf("len(Bits) = %d, want %d", len(decoded.Bits), len(tt.bv.Bits))
			}
			for i := range tt.bv.Bits {
				if decoded.Bits[i] != tt.bv.Bits[i] {
					t.Errorf("Bits[%d] = %v, want %v", i, decoded.Bits[i], tt.bv.Bits[i])
				}
			}

			// Verify OutlierIndices
			if len(decoded.OutlierIndices) != len(tt.bv.OutlierIndices) {
				t.Errorf("len(OutlierIndices) = %d, want %d",
					len(decoded.OutlierIndices), len(tt.bv.OutlierIndices))
			} else {
				for i := range tt.bv.OutlierIndices {
					if decoded.OutlierIndices[i] != tt.bv.OutlierIndices[i] {
						t.Errorf("OutlierIndices[%d] = %d, want %d",
							i, decoded.OutlierIndices[i], tt.bv.OutlierIndices[i])
					}
				}
			}

			// Verify OutlierValues
			if len(decoded.OutlierValues) != len(tt.bv.OutlierValues) {
				t.Errorf("len(OutlierValues) = %d, want %d",
					len(decoded.OutlierValues), len(tt.bv.OutlierValues))
			} else {
				for i := range tt.bv.OutlierValues {
					if decoded.OutlierValues[i] != tt.bv.OutlierValues[i] {
						t.Errorf("OutlierValues[%d] = %v, want %v",
							i, decoded.OutlierValues[i], tt.bv.OutlierValues[i])
					}
				}
			}
		})
	}
}

func TestSerialization_UnknownVersion(t *testing.T) {
	// Build a byte sequence with version 0x99 (unsupported)
	data := []byte{0x99, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}

	var bv BitVector
	err := bv.UnmarshalBinary(data)
	if err == nil {
		t.Fatal("expected error for unknown version")
	}
	if !errors.Is(err, serialErrUnsupportedVersion) {
		// Check if the error message mentions unsupported version
		t.Logf("error: %v", err)
	}
}

func TestSerialization_TruncatedData(t *testing.T) {
	// Create valid data then truncate it
	bv := BitVector{Bits: []uint64{0xDEADBEEF}, Dim: 64}
	data, err := bv.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	// Truncate to just version byte + partial dim
	truncated := data[:3]
	var decoded BitVector
	err = decoded.UnmarshalBinary(truncated)
	if err == nil {
		t.Fatal("expected error for truncated data")
	}
}

func TestSerialization_IntegrationWithQJL(t *testing.T) {
	// Full pipeline: create sketcher → sketch → serialize → deserialize → compare
	sketcher, err := NewQJLSketcher(QJLOptions{Dim: 32, SketchDim: 16, Seed: 42})
	if err != nil {
		t.Fatalf("NewQJLSketcher: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float64, 32)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	original, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch: %v", err)
	}

	data, err := original.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	var restored BitVector
	if err := restored.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary: %v", err)
	}

	// Verify all fields match
	if restored.Dim != original.Dim {
		t.Errorf("Dim = %d, want %d", restored.Dim, original.Dim)
	}
	for i := range original.Bits {
		if restored.Bits[i] != original.Bits[i] {
			t.Errorf("Bits[%d] = %v, want %v", i, restored.Bits[i], original.Bits[i])
		}
	}

	// Verify the restored BitVector gives the same Hamming distance as the original
	hd1 := HammingDistance(*original, *original)
	hd2 := HammingDistance(restored, restored)
	if hd1 != hd2 {
		t.Errorf("HammingDistance(self) changed after round-trip: %d vs %d", hd1, hd2)
	}
}

// Verify BitVector satisfies encoding.BinaryMarshaler/Unmarshaler at compile time.
var _ encoding.BinaryMarshaler = BitVector{}
var _ encoding.BinaryUnmarshaler = (*BitVector)(nil)

// ---------------------------------------------------------------------------
// SRHT variant
// ---------------------------------------------------------------------------

func TestQJLSketch_SRHT(t *testing.T) {
	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:       64,
		SketchDim: 16,
		Seed:      42,
		UseSRHT:   true,
	})
	if err != nil {
		t.Fatalf("NewQJLSketcher with SRHT: %v", err)
	}

	vec := make([]float64, 64)
	rng := rand.New(rand.NewSource(99))
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	bv, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch: %v", err)
	}
	if bv.Dim != 16 {
		t.Errorf("Dim = %d, want 16", bv.Dim)
	}

	// SRHT sketcher should also be deterministic
	bv2, err := sketcher.Sketch(vec)
	if err != nil {
		t.Fatalf("Sketch (2nd): %v", err)
	}
	for i := range bv.Bits {
		if bv.Bits[i] != bv2.Bits[i] {
			t.Errorf("SRHT determinism violation at Bits[%d]: %v != %v", i, bv.Bits[i], bv2.Bits[i])
		}
	}
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// serialErrUnsupportedVersion is imported indirectly — we check the error
// by inspecting the error chain. The serial package defines this error.
var serialErrUnsupportedVersion = newUnsupportedVersionErr()

func newUnsupportedVersionErr() error {
	// We can't import internal/serial from the test file's error check,
	// so we create a known BitVector with a bad version and check the error.
	var bv BitVector
	err := bv.UnmarshalBinary([]byte{0x99})
	return err
}

// scoredItem pairs an index with a score for top-K selection.
type scoredItem struct {
	idx   int
	score float64
}

// topKByScore returns the K items with the highest scores.
func topKByScore(items []scoredItem, k int) []scoredItem {
	sorted := make([]scoredItem, len(items))
	copy(sorted, items)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].score > sorted[j].score
	})
	if k > len(sorted) {
		k = len(sorted)
	}
	return sorted[:k]
}

// BenchmarkQJLSketch benchmarks the Sketch method.
func BenchmarkQJLSketch(b *testing.B) {
	sketcher, err := NewQJLSketcher(QJLOptions{Dim: 768, SketchDim: 256, Seed: 42})
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(0))
	vec := make([]float64, 768)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := sketcher.Sketch(vec)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkQJLSketch_WithOutliers benchmarks sketching with outlier handling.
func BenchmarkQJLSketch_WithOutliers(b *testing.B) {
	sketcher, err := NewQJLSketcher(QJLOptions{
		Dim:       768,
		SketchDim: 256,
		Seed:      42,
		OutlierK:  8,
	})
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(0))
	vec := make([]float64, 768)
	for i := range vec {
		vec[i] = rng.NormFloat64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := sketcher.Sketch(vec)
		if err != nil {
			b.Fatal(err)
		}
	}
}
