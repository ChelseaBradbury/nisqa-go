// Package melspec reproduces the mel-spectrogram front-end NISQA computes with
// librosa. The network was trained on these exact features, so anything that
// drifts here shows up as a wrong score — the numbers below are chosen to match
// librosa's defaults rather than to be independently reasonable.
package melspec

import (
	"fmt"
	"math"

	"nisqa-go/internal/dsp"
)

// Config mirrors the arguments NISQA passes to librosa.feature.melspectrogram.
type Config struct {
	SampleRate int
	NFFT       int
	HopLength  int // samples
	WinLength  int // samples
	NMels      int
	FMin       float64
	FMax       float64
}

// AmplitudeToDB parameters, fixed by NISQA's call to librosa.core.amplitude_to_db.
const (
	dbAmin  = 1e-4 // amplitude floor; squared to 1e-8 in the power domain
	dbTopDB = 80.0
)

type Extractor struct {
	cfg     Config
	fft     *dsp.FFT
	window  []float64   // NFFT long: the Hann window zero-padded and centred
	filters [][]float64 // [NMels][NFFT/2+1]
}

func New(cfg Config) (*Extractor, error) {
	if cfg.WinLength > cfg.NFFT {
		return nil, fmt.Errorf("melspec: win_length %d exceeds n_fft %d", cfg.WinLength, cfg.NFFT)
	}
	e := &Extractor{cfg: cfg, fft: dsp.NewFFT(cfg.NFFT)}

	// librosa centres the analysis window inside an n_fft-long frame.
	e.window = make([]float64, cfg.NFFT)
	offset := (cfg.NFFT - cfg.WinLength) / 2
	copy(e.window[offset:], dsp.HannPeriodic(cfg.WinLength))

	e.filters = melFilterBank(cfg)
	return e, nil
}

// Frames reports how many spectrogram columns a signal of the given length
// produces. librosa's centred STFT yields 1 + len(y)/hop frames.
func (e *Extractor) Frames(n int) int { return 1 + n/e.cfg.HopLength }

// Compute returns the log-mel spectrogram in dB, indexed [mel][frame].
func (e *Extractor) Compute(y []float64) ([][]float64, error) {
	pad := e.cfg.NFFT / 2
	if len(y) <= pad {
		return nil, fmt.Errorf("melspec: signal of %d samples is shorter than the %d-sample analysis window", len(y), e.cfg.NFFT)
	}
	padded := reflectPad(y, pad)

	nFrames := e.Frames(len(y))
	spec := make([][]float64, e.cfg.NMels)
	for m := range spec {
		spec[m] = make([]float64, nFrames)
	}

	frame := make([]float64, e.cfg.NFFT)
	mag := make([]float64, e.cfg.NFFT/2+1)
	for t := 0; t < nFrames; t++ {
		start := t * e.cfg.HopLength
		for i := range frame {
			frame[i] = padded[start+i] * e.window[i]
		}
		e.fft.MagnitudeReal(frame, mag)
		for m, filter := range e.filters {
			var sum float64
			for k, w := range filter {
				if w != 0 {
					sum += w * mag[k]
				}
			}
			spec[m][t] = sum
		}
	}

	amplitudeToDB(spec)
	return spec, nil
}

// reflectPad mirrors numpy.pad(y, pad, mode="reflect"): the edge sample itself
// is not repeated.
func reflectPad(y []float64, pad int) []float64 {
	out := make([]float64, len(y)+2*pad)
	copy(out[pad:], y)
	for j := 0; j < pad; j++ {
		out[j] = y[pad-j]
		out[pad+len(y)+j] = y[len(y)-2-j]
	}
	return out
}

// amplitudeToDB is librosa.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80)
// in place: convert to dB, then clamp everything to within 80 dB of the loudest
// bin in the whole spectrogram.
func amplitudeToDB(spec [][]float64) {
	const floor = dbAmin * dbAmin // power-domain amin
	peak := math.Inf(-1)
	for _, row := range spec {
		for t, v := range row {
			p := v * v
			if p < floor {
				p = floor
			}
			db := 10 * math.Log10(p)
			row[t] = db
			if db > peak {
				peak = db
			}
		}
	}
	limit := peak - dbTopDB
	for _, row := range spec {
		for t, db := range row {
			if db < limit {
				row[t] = limit
			}
		}
	}
}

// melFilterBank is librosa.filters.mel(..., htk=False, norm="slaney").
func melFilterBank(cfg Config) [][]float64 {
	nBins := cfg.NFFT/2 + 1
	fftFreqs := make([]float64, nBins)
	for k := range fftFreqs {
		fftFreqs[k] = float64(k) * float64(cfg.SampleRate) / float64(cfg.NFFT)
	}

	// n_mels+2 band edges, evenly spaced on the Slaney mel scale.
	edges := make([]float64, cfg.NMels+2)
	lo, hi := hzToMel(cfg.FMin), hzToMel(cfg.FMax)
	for i := range edges {
		edges[i] = melToHz(lo + (hi-lo)*float64(i)/float64(len(edges)-1))
	}

	filters := make([][]float64, cfg.NMels)
	for m := range filters {
		filters[m] = make([]float64, nBins)
		lower, centre, upper := edges[m], edges[m+1], edges[m+2]
		// Slaney normalisation: equal area per filter rather than equal peak.
		norm := 2.0 / (upper - lower)
		for k, f := range fftFreqs {
			ramp := math.Min((f-lower)/(centre-lower), (upper-f)/(upper-centre))
			if ramp > 0 {
				filters[m][k] = ramp * norm
			}
		}
	}
	return filters
}

// Slaney's auditory mel scale: linear below 1 kHz, logarithmic above.
const (
	melFSp       = 200.0 / 3.0
	melMinLogHz  = 1000.0
	melMinLogMel = melMinLogHz / melFSp
)

var melLogStep = math.Log(6.4) / 27.0

func hzToMel(hz float64) float64 {
	if hz >= melMinLogHz {
		return melMinLogMel + math.Log(hz/melMinLogHz)/melLogStep
	}
	return hz / melFSp
}

func melToHz(mel float64) float64 {
	if mel >= melMinLogMel {
		return melMinLogHz * math.Exp(melLogStep*(mel-melMinLogMel))
	}
	return mel * melFSp
}
