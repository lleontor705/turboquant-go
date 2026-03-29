package quantize

// PolarVector holds the compressed representation from PolarQuant.
//
// The vector is decomposed into angle indices per level (quantized using
// precomputed codebooks) and final-level radii stored in full precision.
//
// For L=4 levels and dim=d:
//   - AngleIndices[0] has d/2 entries (level 1)
//   - AngleIndices[1] has d/4 entries (level 2)
//   - AngleIndices[2] has d/8 entries (level 3)
//   - AngleIndices[3] has d/16 entries (level 4)
//   - Radii has d/16 entries (final radii)
type PolarVector struct {
	AngleIndices [][]int  // angle indices per level: level 0..L-1
	Radii        []uint16 // quantized radii (d/2^L values)
	Dim          int      // original dimension
	BitsPerLevel []int    // bits per level [4, 2, 2, 2]
}

// PolarConfig holds PolarQuant configuration.
//
// The dimension must be a multiple of 2^Levels (typically 16 for L=4).
// Default bit allocation matches the PolarQuant paper (arXiv:2502.02617):
// level 1: 4 bits, levels 2-4: 2 bits each, radii: 16 bits (FP16-like).
type PolarConfig struct {
	Dim        int   // must be multiple of 2^Levels (typically multiple of 16)
	Levels     int   // number of polar decomposition levels (default 4)
	BitsLevel1 int   // bits for level-1 angles (default 4)
	BitsRest   int   // bits for levels 2..Levels angles (default 2)
	RadiusBits int   // bits for final radii (default 16, FP16-like)
	Seed       int64 // random seed for reproducibility
}

// DefaultPolarConfig returns the standard PolarQuant configuration.
func DefaultPolarConfig(dim int) PolarConfig {
	return PolarConfig{
		Dim:        dim,
		Levels:     4,
		BitsLevel1: 4,
		BitsRest:   2,
		RadiusBits: 16,
		Seed:       42,
	}
}

// BitsPerCoord returns the average bits per coordinate for this configuration.
// For the default config: (4*d/2 + 2*d/4 + 2*d/8 + 2*d/16 + 16*d/16) / d = 3.875
func (c PolarConfig) BitsPerCoord() float64 {
	d := float64(c.Dim)
	totalBits := float64(c.BitsLevel1)*d/2 +
		float64(c.BitsRest)*d/4 +
		float64(c.BitsRest)*d/8 +
		float64(c.BitsRest)*d/16 +
		float64(c.RadiusBits)*d/16
	return totalBits / d
}
