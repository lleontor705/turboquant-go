// turbo_search demonstrates TurboQuant_mse vector compression for approximate
// nearest neighbor (ANN) search.
//
// The example simulates a realistic embedding search scenario:
//  1. Generate cluster centers on the unit hypersphere (simulating semantic
//     categories in embedding space).
//  2. Create database vectors as noisy perturbations of cluster centers.
//  3. Encode every database vector using TurboQuant_mse (random rotation +
//     optimal Lloyd-Max quantization).
//  4. For each query: encode → decode → compute dot product with decoded vectors.
//  5. Compare ranking against exact dot product → measure recall@10.
//  6. Report compression ratio, MSE, recall, and throughput.
//
// TurboQuant_mse reduces each vector from dim*8 bytes to dim*bits/8 bytes
// while preserving neighbor ordering through the orthogonal rotation's
// norm-preserving property and optimal centroid placement.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/lleontor705/turboquant-go/quantize"
)

func main() {
	const (
		// numVectors is the database size.
		numVectors = 10000
		// dim is the vector dimensionality (768 for BERT-base embeddings).
		dim = 768
		// bits is the number of bits per coordinate for TurboQuant_mse.
		bits = 3
		// numQueries is the number of query vectors to evaluate.
		numQueries = 10
		// vectorsPerCluster is the database vectors per cluster.
		vectorsPerCluster = 10
		// numClusters = numVectors / vectorsPerCluster.
		numClusters = numVectors / vectorsPerCluster
		// clusterNoise controls how tightly vectors cluster around centers.
		clusterNoise = 0.005
		// topK is the number of nearest neighbors to retrieve.
		topK = 10
		// seed ensures reproducible results.
		seed = int64(42)
	)

	fmt.Println("=== TurboQuant_mse Vector Compression for ANN Search ===")
	fmt.Printf("Database:    %d vectors, dim=%d, clusters=%d\n", numVectors, dim, numClusters)
	fmt.Printf("Quantization: %d bits/coordinate\n", bits)
	fmt.Printf("Original size:    %d bytes/vector (float64)\n", dim*8)
	fmt.Printf("Compressed size:  %d bytes/vector (%d-bit)\n\n", (dim*bits+7)/8, bits)

	rng := rand.New(rand.NewSource(seed))

	// Step 1: Generate cluster centers on the unit hypersphere.
	// These represent semantic directions in embedding space.
	centers := make([][]float64, numClusters)
	for i := range centers {
		centers[i] = randomUnitVector(rng, dim)
	}
	fmt.Printf("Generated %d cluster centers\n", numClusters)

	// Step 2: Generate database vectors near cluster centers.
	vectors := make([][]float64, numVectors)
	for i := range vectors {
		clusterID := i / vectorsPerCluster
		vectors[i] = perturbedVector(rng, centers[clusterID], clusterNoise, dim)
	}
	fmt.Printf("Generated %d database vectors (%d per cluster)\n", numVectors, vectorsPerCluster)

	// Step 3: Create TurboQuantizer and encode all database vectors.
	tq, err := quantize.NewTurboQuantizer(dim, bits, seed)
	if err != nil {
		panic(fmt.Sprintf("failed to create TurboQuantizer: %v", err))
	}

	compressed := make([]quantize.CompressedVector, numVectors)
	encodeStart := time.Now()
	for i, v := range vectors {
		compressed[i], err = tq.Quantize(v)
		if err != nil {
			panic(fmt.Sprintf("failed to quantize vector %d: %v", i, err))
		}
	}
	encodeDuration := time.Since(encodeStart)

	// Dequantize all vectors for MSE measurement and ANN search.
	decoded := make([][]float64, numVectors)
	decodeStart := time.Now()
	for i, cv := range compressed {
		decoded[i], err = tq.Dequantize(cv)
		if err != nil {
			panic(fmt.Sprintf("failed to dequantize vector %d: %v", i, err))
		}
	}
	decodeDuration := time.Since(decodeStart)

	// Compute MSE.
	mse := computeMSE(vectors, decoded)

	// Report compression statistics.
	originalBytes := dim * 8
	compressedBytes := len(compressed[0].Data)
	ratio := float64(originalBytes) / float64(compressedBytes)

	fmt.Printf("Encoded all %d vectors into %d-bit compressed form\n", numVectors, bits)

	fmt.Println()
	fmt.Println("--- Compression Results ---")
	fmt.Printf("  Original:      %d bytes/vector\n", originalBytes)
	fmt.Printf("  Compressed:    %d bytes/vector\n", compressedBytes)
	fmt.Printf("  Compression:   %.1fx\n", ratio)
	fmt.Printf("  MSE:           %.6f\n", mse)
	fmt.Printf("  Encode time:   %s (%.0f vectors/sec)\n",
		encodeDuration, float64(numVectors)/encodeDuration.Seconds())
	fmt.Printf("  Decode time:   %s (%.0f vectors/sec)\n",
		decodeDuration, float64(numVectors)/decodeDuration.Seconds())

	// Step 4: Generate query vectors and evaluate recall.
	queries := make([][]float64, numQueries)
	for i := range queries {
		clusterID := i * (numClusters / numQueries)
		queries[i] = perturbedVector(rng, centers[clusterID], clusterNoise, dim)
	}

	fmt.Println()
	fmt.Println("--- Recall Evaluation ---")

	totalRecall := 0.0
	for qi, query := range queries {
		// Exact ranking: dot product with original vectors.
		exactTopK := exactRanking(query, vectors, topK)

		// Approximate ranking: dot product with decoded vectors.
		approxTopK := exactRanking(query, decoded, topK)

		recall := computeRecall(exactTopK, approxTopK)
		totalRecall += recall

		fmt.Printf("  Query %2d: recall@%d = %.2f\n", qi, topK, recall)
	}

	avgRecall := totalRecall / float64(numQueries)
	fmt.Printf("\n  Average recall@%d: %.4f\n", topK, avgRecall)

	// Step 5: Summary.
	fmt.Println()
	fmt.Println("=== Summary ===")
	fmt.Printf("  TurboQuant_mse (%d-bit, %d-dim):\n", bits, dim)
	fmt.Printf("  Compression: %d bytes → %d bytes (%.1fx)\n",
		originalBytes, compressedBytes, ratio)
	fmt.Printf("  MSE:         %.6f\n", mse)
	fmt.Printf("  Recall@%d:    %.2f\n", topK, avgRecall)
	fmt.Printf("  Encode:      %.0f vectors/sec\n", float64(numVectors)/encodeDuration.Seconds())
	fmt.Printf("  Decode:      %.0f vectors/sec\n", float64(numVectors)/decodeDuration.Seconds())

	if avgRecall >= 0.90 {
		fmt.Println("\n  PASSED: Recall target met!")
		fmt.Println("  TurboQuant_mse preserves neighbor ordering well for ANN search.")
	} else {
		fmt.Println("\n  INFO: Recall is moderate. Try increasing bits (e.g. 4) for higher quality.")
	}
}

// randomUnitVector generates a random vector on the unit hypersphere.
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

// perturbedVector creates a unit vector near center with Gaussian noise.
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

// exactRanking returns indices of top-K highest dot-product vectors.
func exactRanking(query []float64, database [][]float64, topK int) []int {
	items := make([]struct {
		Index int
		Score float64
	}, len(database))
	for i, vec := range database {
		items[i].Index = i
		items[i].Score = dotProduct(query, vec)
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].Score > items[j].Score
	})

	result := make([]int, topK)
	for i := 0; i < topK; i++ {
		result[i] = items[i].Index
	}
	return result
}

// computeRecall measures the fraction of exactTop items in approxTop.
func computeRecall(exactTop, approxTop []int) float64 {
	set := make(map[int]struct{}, len(approxTop))
	for _, idx := range approxTop {
		set[idx] = struct{}{}
	}
	hits := 0
	for _, idx := range exactTop {
		if _, ok := set[idx]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(exactTop))
}

// computeMSE computes mean squared error between original and reconstructed vectors.
func computeMSE(original, reconstructed [][]float64) float64 {
	var totalSqErr float64
	var count float64
	for i := range original {
		for j := range original[i] {
			d := original[i][j] - reconstructed[i][j]
			totalSqErr += d * d
			count++
		}
	}
	return totalSqErr / count
}

// dotProduct computes the dot product of two vectors.
func dotProduct(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
