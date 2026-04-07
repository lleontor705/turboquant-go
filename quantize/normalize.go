package quantize

import (
	"encoding/binary"
	"hash/fnv"
	"math"

	"gonum.org/v1/gonum/mat"
)

// normalizeUnit copies vec and normalizes it to unit norm.
// Returns the normalized vector and original norm.
// A zero-norm vector is returned as-is with norm 0.
func normalizeUnit(vec []float64) (unit []float64, norm float64) {
	unit = make([]float64, len(vec))
	copy(unit, vec)
	for _, v := range unit {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		invNorm := 1.0 / norm
		for i := range unit {
			unit[i] *= invNorm
		}
	}
	return unit, norm
}

// rotateForward computes y = rotation * vec.
func rotateForward(rotation *mat.Dense, vec []float64) []float64 {
	dim := len(vec)
	x := mat.NewVecDense(dim, vec)
	y := mat.NewVecDense(dim, nil)
	y.MulVec(rotation, x)
	return y.RawVector().Data
}

// rotateInverse computes y = rotation^T * vec.
func rotateInverse(rotation *mat.Dense, vec []float64) []float64 {
	dim := len(vec)
	x := mat.NewVecDense(dim, vec)
	y := mat.NewVecDense(dim, nil)
	y.MulVec(rotation.T(), x)
	return y.RawVector().Data
}

// fnvDeriveSeed derives an independent seed from a parent seed using FNV-1a hash.
func fnvDeriveSeed(seed int64) int64 {
	h := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(seed))
	h.Write(buf[:])
	h.Write([]byte("qjl")) // domain separation
	return int64(h.Sum64())
}
