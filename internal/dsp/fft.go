// Package dsp provides the small numeric primitives the mel front-end needs.
package dsp

import "math"

// FFT computes an in-place radix-2 decimation-in-time FFT. len(x) must be a
// power of two. NISQA only ever asks for n_fft = 4096, so radix-2 is enough.
type FFT struct {
	n        int
	twiddles []complex128 // w[k] = exp(-2i*pi*k/n) for k < n/2
	reverse  []int        // bit-reversal permutation
	buf      []complex128
}

func NewFFT(n int) *FFT {
	if n < 2 || n&(n-1) != 0 {
		panic("dsp: FFT size must be a power of two")
	}
	f := &FFT{n: n, twiddles: make([]complex128, n/2), reverse: make([]int, n), buf: make([]complex128, n)}
	for k := range f.twiddles {
		angle := -2 * math.Pi * float64(k) / float64(n)
		f.twiddles[k] = complex(math.Cos(angle), math.Sin(angle))
	}
	bits := 0
	for 1<<bits < n {
		bits++
	}
	for i := range f.reverse {
		r, v := 0, i
		for b := 0; b < bits; b++ {
			r = r<<1 | v&1
			v >>= 1
		}
		f.reverse[i] = r
	}
	return f
}

// MagnitudeReal takes n real samples and writes the magnitudes of the first
// n/2+1 bins into out — the same bins numpy's rfft returns.
func (f *FFT) MagnitudeReal(in []float64, out []float64) {
	if len(in) != f.n {
		panic("dsp: input length must equal FFT size")
	}
	if len(out) != f.n/2+1 {
		panic("dsp: output length must equal n/2+1")
	}
	for i, r := range f.reverse {
		f.buf[i] = complex(in[r], 0)
	}
	for size := 2; size <= f.n; size <<= 1 {
		step := f.n / size
		half := size / 2
		for start := 0; start < f.n; start += size {
			for k := 0; k < half; k++ {
				even := f.buf[start+k]
				odd := f.buf[start+k+half] * f.twiddles[k*step]
				f.buf[start+k] = even + odd
				f.buf[start+k+half] = even - odd
			}
		}
	}
	for k := range out {
		out[k] = math.Hypot(real(f.buf[k]), imag(f.buf[k]))
	}
}

// HannPeriodic returns a periodic (fftbins=True) Hann window, matching
// scipy.signal.get_window("hann", n) as librosa calls it.
func HannPeriodic(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n))
	}
	return w
}

// BesselI0 is the zeroth-order modified Bessel function of the first kind,
// used to build Kaiser windows.
func BesselI0(x float64) float64 {
	sum, term := 1.0, 1.0
	half := x / 2
	for k := 1; k < 64; k++ {
		term *= (half / float64(k)) * (half / float64(k))
		sum += term
		if term < 1e-18*sum {
			break
		}
	}
	return sum
}
