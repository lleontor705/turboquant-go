// prod_search demonstrates TurboQuant_prod vector compression for approximate
// nearest neighbor (ANN) search using unbiased inner product estimation.
//
// TurboQuant_prod combines MSE quantization with a 1-bit QJL residual sketch
// to provide UNBIASED inner product estimation — a key advantage over plain
// TurboQuant_mse for ranking/retrieval tasks.
//
// The example:
//  1. Create TurboProdQuantizer (3-bit total = 2-bit MSE + 1-bit QJL, d=768)
//  2. Encode 10000 random clustered unit vectors
//  3. For queries: encode → EstimateInnerProduct for ranking
//  4. Compare against exact dot product ranking → measure recall@10
//  5. Report: compression ratio, IP estimation accuracy, recall@10, throughput
//
// Also compares TurboQuant_prod vs plain TurboQuant_mse (no IP correction).
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
		numVectors        = 1000
		dim               = 768
		prodBits          = 3 // 2-bit MSE + 1-bit QJL
		mseBits           = 3 // plain TurboQuant_mse for comparison
		numQueries        = 10
		vectorsPerCluster = 10
		numClusters       = numVectors / vectorsPerCluster
		clusterNoise      = 0.005
		topK              = 10
		ipSampleSize      = 100 // subset for IP accuracy measurement
		seed              = int64(42)
	)

	fmt.Println("=== TurboQuant_prod vs TurboQuant_mse: ANN Search Comparison ===")
	fmt.Printf("Database:    %d vectors, dim=%d, clusters=%d\n", numVectors, dim, numClusters)
	fmt.Printf("TurboQuant_prod: %d bits/coord (%d-bit MSE + 1-bit QJL)\n", prodBits, prodBits-1)
	fmt.Printf("TurboQuant_mse:  %d bits/coord\n\n", mseBits)

	rng := rand.New(rand.NewSource(seed))

	// Step 1: Generate cluster centers on the unit hypersphere.
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

	// Step 3: Create quantizers and encode all database vectors.
	tpq, err := quantize.NewTurboProdQuantizer(dim, prodBits, dim, seed)
	if err != nil {
		panic(fmt.Sprintf("failed to create TurboProdQuantizer: %v", err))
	}

	tq, err := quantize.NewTurboQuantizer(dim, mseBits, seed)
	if err != nil {
		panic(fmt.Sprintf("failed to create TurboQuantizer: %v", err))
	}

	// Encode with TurboQuant_prod.
	prodCompressed := make([]quantize.CompressedVector, numVectors)
	prodEncodeStart := time.Now()
	for i, v := range vectors {
		prodCompressed[i], err = tpq.Quantize(v)
		if err != nil {
			panic(fmt.Sprintf("failed to quantize vector %d: %v", i, err))
		}
	}
	prodEncodeDuration := time.Since(prodEncodeStart)

	// Encode with TurboQuant_mse (for comparison).
	mseCompressed := make([]quantize.CompressedVector, numVectors)
	mseEncodeStart := time.Now()
	for i, v := range vectors {
		mseCompressed[i], err = tq.Quantize(v)
		if err != nil {
			panic(fmt.Sprintf("failed to quantize vector %d: %v", i, err))
		}
	}
	mseEncodeDuration := time.Since(mseEncodeStart)

	// Dequantize MSE vectors for comparison search.
	mseDecoded := make([][]float64, numVectors)
	for i, cv := range mseCompressed {
		mseDecoded[i], err = tq.Dequantize(cv)
		if err != nil {
			panic(fmt.Sprintf("failed to dequantize vector %d: %v", i, err))
		}
	}

	// Report compression statistics.
	originalBytes := dim * 8
	prodCompressedBytes := len(prodCompressed[0].Data)
	mseCompressedBytes := len(mseCompressed[0].Data)
	prodRatio := float64(originalBytes) / float64(prodCompressedBytes)
	mseRatio := float64(originalBytes) / float64(mseCompressedBytes)

	fmt.Println()
	fmt.Println("--- Compression Results ---")
	fmt.Printf("  Original:          %d bytes/vector\n", originalBytes)
	fmt.Printf("  TurboQuant_prod:   %d bytes/vector (%.1fx)\n", prodCompressedBytes, prodRatio)
	fmt.Printf("  TurboQuant_mse:    %d bytes/vector (%.1fx)\n", mseCompressedBytes, mseRatio)
	fmt.Printf("  Prod encode time:  %s (%.0f vec/sec)\n",
		prodEncodeDuration, float64(numVectors)/prodEncodeDuration.Seconds())
	fmt.Printf("  MSE encode time:   %s (%.0f vec/sec)\n",
		mseEncodeDuration, float64(numVectors)/mseEncodeDuration.Seconds())

	// Step 4: Generate query vectors and evaluate recall.
	queries := make([][]float64, numQueries)
	for i := range queries {
		clusterID := i * (numClusters / numQueries)
		queries[i] = perturbedVector(rng, centers[clusterID], clusterNoise, dim)
	}

	fmt.Println()
	fmt.Println("--- Recall Evaluation ---")

	var prodTotalRecall, mseTotalRecall float64
	var prodTotalIPErr, mseTotalIPErr float64
	var totalIPEstimations int

	for qi, query := range queries {
		// Exact ranking: dot product with original vectors.
		exactTopK := exactRanking(query, vectors, topK)

		// TurboQuant_prod ranking: use EstimateInnerProduct.
		prodTopK := prodRanking(tpq, query, prodCompressed, topK)

		// TurboQuant_mse ranking: dequantize → dot product.
		mseTopK := exactRanking(query, mseDecoded, topK)

		prodRecall := computeRecall(exactTopK, prodTopK)
		mseRecall := computeRecall(exactTopK, mseTopK)
		prodTotalRecall += prodRecall
		mseTotalRecall += mseRecall

		// Measure IP estimation accuracy on a sample.
		sampleSize := ipSampleSize
		if sampleSize > len(vectors) {
			sampleSize = len(vectors)
		}
		for i := 0; i < sampleSize; i++ {
			trueIP := dotProduct(query, vectors[i])
			estIP, _ := tpq.EstimateInnerProduct(query, prodCompressed[i])
			prodTotalIPErr += math.Abs(estIP - trueIP)

			mseIP := dotProduct(query, mseDecoded[i])
			mseTotalIPErr += math.Abs(mseIP - trueIP)
			totalIPEstimations++
		}

		if qi < 5 {
			fmt.Printf("  Query %2d: prod recall@%d = %.2f, mse recall@%d = %.2f\n",
				qi, topK, prodRecall, topK, mseRecall)
		}
	}

	avgProdRecall := prodTotalRecall / float64(numQueries)
	avgMseRecall := mseTotalRecall / float64(numQueries)
	avgProdIPErr := prodTotalIPErr / float64(totalIPEstimations)
	avgMseIPErr := mseTotalIPErr / float64(totalIPEstimations)

	fmt.Printf("\n  Average recall@%d:\n", topK)
	fmt.Printf("    TurboQuant_prod: %.4f\n", avgProdRecall)
	fmt.Printf("    TurboQuant_mse:  %.4f\n", avgMseRecall)

	fmt.Printf("\n  Average |IP error| (over %d estimates):\n", totalIPEstimations)
	fmt.Printf("    TurboQuant_prod: %.6f\n", avgProdIPErr)
	fmt.Printf("    TurboQuant_mse:  %.6f\n", avgMseIPErr)

	// Step 5: Measure query throughput on a subset.
	throughputSubset := 100
	if throughputSubset > len(prodCompressed) {
		throughputSubset = len(prodCompressed)
	}
	throughputQueries := 5
	if throughputQueries > len(queries) {
		throughputQueries = len(queries)
	}
	queryStart := time.Now()
	for qi := 0; qi < throughputQueries; qi++ {
		for i := 0; i < throughputSubset; i++ {
			_, _ = tpq.EstimateInnerProduct(queries[qi], prodCompressed[i])
		}
	}
	queryDuration := time.Since(queryStart)
	totalEstimates := throughputQueries * throughputSubset
	throughput := float64(totalEstimates) / queryDuration.Seconds()

	fmt.Println()
	fmt.Println("--- Throughput ---")
	fmt.Printf("  IP estimation: %.0f estimates/sec (%d queries × %d vectors in %s)\n",
		throughput, throughputQueries, throughputSubset, queryDuration.Round(time.Millisecond))

	// Step 6: Summary.
	fmt.Println()
	fmt.Println("=== Summary ===")
	fmt.Printf("  TurboQuant_prod (%d-bit, %d-dim):\n", prodBits, dim)
	fmt.Printf("  Compression: %d bytes → %d bytes (%.1fx)\n",
		originalBytes, prodCompressedBytes, prodRatio)
	fmt.Printf("  Recall@%d:           %.4f\n", topK, avgProdRecall)
	fmt.Printf("  IP accuracy:        %.6f avg |error|\n", avgProdIPErr)
	fmt.Printf("  IP throughput:      %.0f estimates/sec\n", throughput)

	fmt.Println()
	fmt.Printf("  TurboQuant_mse (%d-bit, %d-dim):\n", mseBits, dim)
	fmt.Printf("  Recall@%d:           %.4f\n", topK, avgMseRecall)
	fmt.Printf("  IP accuracy:        %.6f avg |error|\n", avgMseIPErr)

	if avgProdIPErr < avgMseIPErr {
		fmt.Printf("\n  ✓ TurboQuant_prod IP estimation is %.2fx more accurate than MSE-only\n",
			avgMseIPErr/avgProdIPErr)
	}

	if avgProdRecall >= 0.90 {
		fmt.Println("  ✓ TurboQuant_prod recall target (≥0.90) met!")
	} else {
		fmt.Println("  ℹ Recall is moderate. Try increasing bits (e.g. 4) for higher quality.")
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

// prodRanking uses EstimateInnerProduct for ranking.
func prodRanking(tpq *quantize.TurboProdQuantizer, query []float64, compressed []quantize.CompressedVector, topK int) []int {
	items := make([]struct {
		Index int
		Score float64
	}, len(compressed))
	for i, cv := range compressed {
		items[i].Index = i
		items[i].Score, _ = tpq.EstimateInnerProduct(query, cv)
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

// dotProduct computes the dot product of two vectors.
func dotProduct(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
