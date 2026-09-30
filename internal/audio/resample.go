package audio

import (
	"math"

	"nisqa-go/internal/dsp"
)

// Resampler parameters: a Kaiser-windowed sinc. librosa resamples with soxr_hq,
// whose transition band no Kaiser window reproduces exactly, so these were swept
// against librosa's output and chosen for the best worst-case agreement —
// roughly 53 dB SNR converting 8 or 16 kHz up to 48 kHz, and 100 dB from
// 44.1 kHz. See TestResampleMatchesLibrosa and the README.
const (
	resampleZeros   = 48
	resampleBeta    = 8.6
	resampleRolloff = 0.96
)

// Resample converts a clip to the target sample rate, returning it unchanged if
// it is already there.
func Resample(c Clip, targetRate int) Clip {
	if c.SampleRate == targetRate || len(c.Samples) == 0 {
		return Clip{Samples: c.Samples, SampleRate: targetRate}
	}

	ratio := float64(targetRate) / float64(c.SampleRate)
	// Downsampling has to band-limit to the new Nyquist; upsampling does not.
	cutoff := resampleRolloff * math.Min(1, ratio)
	// Filter half-width measured in input samples.
	halfWidth := float64(resampleZeros) / cutoff

	in := c.Samples
	n := int(float64(len(in)) * ratio)
	out := make([]float64, n)

	i0 := dsp.BesselI0(resampleBeta)
	for t := range out {
		centre := float64(t) / ratio
		lo := int(math.Ceil(centre - halfWidth))
		hi := int(math.Floor(centre + halfWidth))
		var sum, norm float64
		for i := lo; i <= hi; i++ {
			offset := centre - float64(i)
			// Kaiser window, evaluated on the filter's own support.
			r := offset / halfWidth
			if r < -1 || r > 1 {
				continue
			}
			w := dsp.BesselI0(resampleBeta*math.Sqrt(1-r*r)) / i0
			tap := w * cutoff * sinc(cutoff*offset)
			norm += tap
			if i >= 0 && i < len(in) {
				sum += tap * in[i]
			}
		}
		if norm != 0 {
			out[t] = sum / norm
		}
	}
	return Clip{Samples: out, SampleRate: targetRate}
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}
