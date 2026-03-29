// ann_search demonstrates approximate nearest neighbor (ANN) search using
// QJL 1-bit sketches from the turboquant/sketch package.
//
// The example simulates a realistic embedding search scenario:
//  1. Generate cluster centers on the unit hypersphere (simulating semantic
//     categories in embedding space).
//  2. Create database vectors as noisy perturbations of cluster centers
//     (vectors in the same cluster are highly similar).
//  3. Encode every database vector into a compact 1-bit sketch using QJL.
//  4. For each query vector, find the top-10 nearest neighbors using:
//     a) Exact dot-product ranking (ground truth).
//     b) Sketch-based ranking via Hamming distance.
//  5. Measure recall@10: what fraction of the exact top-10 are also found
//     by the sketch approach.
//
// QJL sketching reduces each vector from dim*8 bytes (float64) to
// sketchDim/8 bytes (1 bit per dimension), yielding massive memory savings
// while preserving approximate neighbor ordering through the Johnson-
// Lindenstrauss property of 1-bit quantization.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/lleontor705/turboquant-go/sketch"
)

func main() {
	const (
		// numVectors is the database size.
		numVectors = 10000
		// dim is the original vector dimensionality (e.g. 768 for BERT-base).
		dim = 768
		// sketchDim is the number of random projections. Each projection
		// becomes 1 bit in the sketch, so the sketch occupies sketchDim/8 bytes.
		sketchDim = 256
		// numQueries is the number of query vectors to evaluate.
		numQueries = 10
		// vectorsPerCluster is the number of database vectors per cluster.
		// The exact top-10 for a query from cluster C are the 10 vectors
		// from C (since same-cluster similarity >> cross-cluster similarity).
		vectorsPerCluster = 10
		// numClusters = numVectors / vectorsPerCluster.
		numClusters = numVectors / vectorsPerCluster
		// clusterNoise controls how tightly vectors cluster around centers.
		// With dim=768, noise must be small so that within-cluster dot
		// products stay high (noise^2 * dim << 1). Value 0.005 gives
		// within-cluster similarity ~0.98 while between-cluster ~0.
		clusterNoise = 0.005
		// seed ensures reproducible results.
		seed = int64(42)
	)

	fmt.Println("=== QJL Sketch-based Approximate Nearest Neighbor Search ===")
	fmt.Printf("Database:   %d vectors, dim=%d, clusters=%d\n", numVectors, dim, numClusters)
	fmt.Printf("Sketch:     %d bits (%d bytes per vector)\n", sketchDim, sketchDim/8)
	fmt.Printf("Compression: %.1fx\n\n",
		float64(dim*8)/float64(sketchDim/8))

	rng := rand.New(rand.NewSource(seed))

	// Step 1: Generate cluster centers on the unit hypersphere.
	// These represent the "semantic directions" in embedding space —
	// e.g., one cluster might correspond to "sports articles" and another
	// to "scientific papers." Vectors in the same cluster share a strong
	// directional component.
	centers := make([][]float64, numClusters)
	for i := range centers {
		centers[i] = randomUnitVector(rng, dim)
	}
	fmt.Printf("Generated %d cluster centers\n", numClusters)

	// Step 2: Generate database vectors near cluster centers.
	// Each vector is created by mixing a cluster center with small Gaussian
	// noise, then normalizing to unit length. Vectors from the same cluster
	// have high dot products (~0.98), while vectors from different clusters
	// have near-zero dot products.
	//
	// Deterministic assignment: exactly vectorsPerCluster vectors per cluster.
	// This ensures the exact top-10 for any query is well-defined: it's the
	// 10 vectors from the query's own cluster.
	vectors := make([][]float64, numVectors)
	for i := range vectors {
		clusterID := i / vectorsPerCluster
		vectors[i] = perturbedVector(rng, centers[clusterID], clusterNoise, dim)
	}
	fmt.Printf("Generated %d database vectors (%d per cluster)\n", numVectors, vectorsPerCluster)

	// Step 3: Create a QJL sketcher and encode all database vectors.
	//
	// The sketcher performs:
	//   1. Johnson-Lindenstrauss random projection: dim → sketchDim
	//   2. Sign quantization: positive projection → 1, non-positive → 0
	//   3. Bit packing into a compact BitVector
	//
	// The same sketcher instance is reused for all vectors (and queries)
	// because it holds the shared random projection matrix. It is safe
	// for concurrent use.
	sketcher, err := sketch.NewQJLSketcher(sketch.QJLOptions{
		Dim:       dim,
		SketchDim: sketchDim,
		Seed:      seed,
	})
	if err != nil {
		panic(fmt.Sprintf("failed to create sketcher: %v", err))
	}

	sketches := make([]*sketch.BitVector, numVectors)
	for i, v := range vectors {
		sketches[i], err = sketcher.Sketch(v)
		if err != nil {
			panic(fmt.Sprintf("failed to sketch vector %d: %v", i, err))
		}
	}
	fmt.Printf("Encoded all %d vectors into %d-bit sketches\n\n", numVectors, sketchDim)

	// Step 4: Generate query vectors and evaluate recall.
	// Each query is a perturbation of a different cluster center, so
	// its nearest neighbors are the 10 database vectors from that cluster.
	queries := make([][]float64, numQueries)
	for i := range queries {
		clusterID := i * (numClusters / numQueries) // spread queries across clusters
		queries[i] = perturbedVector(rng, centers[clusterID], clusterNoise, dim)
	}

	fmt.Println("Evaluating recall@10 for each query:")
	fmt.Println("-------------------------------------")

	totalRecall := 0.0
	for qi, query := range queries {
		// Encode the query vector with the same sketcher.
		querySketch, err := sketcher.Sketch(query)
		if err != nil {
			panic(fmt.Sprintf("failed to sketch query %d: %v", qi, err))
		}

		// Exact ranking: compute dot product with every database vector
		// and return the indices of the top-K highest similarity vectors.
		exactTop10 := exactRanking(query, vectors, 10)

		// Sketch ranking: compute Hamming distance between the query sketch
		// and every database sketch, then return the indices of the top-K
		// lowest-distance (most similar) sketches. Lower Hamming distance
		// corresponds to higher estimated inner product.
		sketchTop10 := sketchRanking(*querySketch, sketches, 10)

		// Compute recall: what fraction of the exact top-10 are also
		// in the sketch top-10?
		recall := computeRecall(exactTop10, sketchTop10)
		totalRecall += recall

		fmt.Printf("  Query %2d: recall@10 = %.2f\n", qi, recall)
	}

	// Step 5: Report aggregate results.
	avgRecall := totalRecall / float64(numQueries)
	fmt.Println("-------------------------------------")
	fmt.Printf("\nAverage recall@10: %.4f\n", avgRecall)
	fmt.Printf("Target:           >= 0.95\n\n")

	if avgRecall >= 0.95 {
		fmt.Println("PASSED: Recall target met!")
		fmt.Println("The 1-bit sketches preserve neighbor ordering well enough")
		fmt.Println("for practical approximate nearest neighbor search.")
	} else {
		fmt.Println("WARNING: Below target - may need higher sketchDim.")
		fmt.Println("Try increasing sketchDim (e.g. 512 or 768) for better recall.")
	}
}

// randomUnitVector generates a random vector on the unit hypersphere.
// Uses the standard approach: sample each component from N(0,1) then normalize.
func randomUnitVector(rng *rand.Rand, dim int) []float64 {
	v := make([]float64, dim)
	var norm float64
	for i := range v {
		v[i] = rng.NormFloat64()
		norm += v[i] * v[i]
	}
	norm = math.Sqrt(norm)
	for i := range v {
		v[i] /= norm
	}
	return v
}

// perturbedVector creates a new unit vector near the given center by mixing
// the center with Gaussian noise. The noise parameter controls the spread:
//
//	noise=0 → identical to center, noise=1 → fully random direction.
func perturbedVector(rng *rand.Rand, center []float64, noise float64, dim int) []float64 {
	v := make([]float64, dim)
	var norm float64
	for i := range v {
		v[i] = center[i] + noise*rng.NormFloat64()
		norm += v[i] * v[i]
	}
	norm = math.Sqrt(norm)
	for i := range v {
		v[i] /= norm
	}
	return v
}

// rankedItem holds an index and its similarity score for sorting.
type rankedItem struct {
	Index int
	Score float64
}

// exactRanking computes exact dot-product similarity between the query and
// every database vector, returning the indices of the top-K most similar
// vectors (highest dot product first).
func exactRanking(query []float64, database [][]float64, topK int) []int {
	items := make([]rankedItem, len(database))
	for i, vec := range database {
		items[i] = rankedItem{
			Index: i,
			Score: dotProduct(query, vec),
		}
	}

	// Sort descending by score.
	sort.Slice(items, func(i, j int) bool {
		return items[i].Score > items[j].Score
	})

	result := make([]int, topK)
	for i := 0; i < topK; i++ {
		result[i] = items[i].Index
	}
	return result
}

// sketchRanking ranks database sketches by Hamming distance to the query
// sketch, returning the indices of the top-K lowest-distance (most similar)
// entries. Lower Hamming distance ≈ higher inner product.
//
// The relationship: estimated_IP = 1 - 2 * (hamming / sketchDim).
// Sorting by ascending Hamming distance is equivalent to sorting by
// descending estimated inner product.
func sketchRanking(query sketch.BitVector, database []*sketch.BitVector, topK int) []int {
	items := make([]rankedItem, len(database))
	for i, bv := range database {
		items[i] = rankedItem{
			Index: i,
			// Use negative Hamming distance so that sorting descending
			// gives us the lowest (most similar) distances first.
			Score: -float64(sketch.HammingDistance(query, *bv)),
		}
	}

	// Sort descending by score (most negative = lowest Hamming distance).
	sort.Slice(items, func(i, j int) bool {
		return items[i].Score > items[j].Score
	})

	result := make([]int, topK)
	for i := 0; i < topK; i++ {
		result[i] = items[i].Index
	}
	return result
}

// computeRecall measures the fraction of exactTop items that also appear in
// sketchTop. Returns a value in [0, 1] where 1.0 means perfect recall.
func computeRecall(exactTop, sketchTop []int) float64 {
	sketchSet := make(map[int]struct{}, len(sketchTop))
	for _, idx := range sketchTop {
		sketchSet[idx] = struct{}{}
	}

	hits := 0
	for _, idx := range exactTop {
		if _, ok := sketchSet[idx]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(exactTop))
}

// dotProduct computes the dot product of two vectors.
func dotProduct(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
