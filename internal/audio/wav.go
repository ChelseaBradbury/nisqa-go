// Package audio decodes WAV files and resamples them to the rate NISQA expects.
package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
)

// Clip is mono audio in [-1, 1].
type Clip struct {
	Samples    []float64
	SampleRate int
}

// Duration in seconds.
func (c Clip) Duration() float64 { return float64(len(c.Samples)) / float64(c.SampleRate) }

const (
	formatPCM        = 1
	formatFloat      = 3
	formatExtensible = 0xFFFE
)

// DecodeWAV reads a RIFF/WAVE file and downmixes it to mono, averaging
// channels the way librosa.to_mono does.
func DecodeWAV(path string) (Clip, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Clip{}, err
	}
	if len(raw) < 12 || string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return Clip{}, fmt.Errorf("%s: not a RIFF/WAVE file", path)
	}

	var (
		format, channels, bits int
		sampleRate             int
		data                   []byte
		haveFmt                bool
	)
	for off := 12; off+8 <= len(raw); {
		id := string(raw[off : off+4])
		size := int(binary.LittleEndian.Uint32(raw[off+4 : off+8]))
		body := off + 8
		if body+size > len(raw) {
			size = len(raw) - body // tolerate a truncated final chunk
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return Clip{}, fmt.Errorf("%s: fmt chunk too short", path)
			}
			f := raw[body : body+size]
			format = int(binary.LittleEndian.Uint16(f[0:2]))
			channels = int(binary.LittleEndian.Uint16(f[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(f[4:8]))
			bits = int(binary.LittleEndian.Uint16(f[14:16]))
			if format == formatExtensible && size >= 26 {
				// The real format is the first two bytes of the subformat GUID.
				format = int(binary.LittleEndian.Uint16(f[24:26]))
			}
			haveFmt = true
		case "data":
			data = raw[body : body+size]
		}
		off = body + size
		if size%2 == 1 {
			off++ // chunks are word-aligned
		}
	}
	if !haveFmt {
		return Clip{}, fmt.Errorf("%s: no fmt chunk", path)
	}
	if data == nil {
		return Clip{}, fmt.Errorf("%s: no data chunk", path)
	}
	if channels < 1 {
		return Clip{}, fmt.Errorf("%s: %d channels", path, channels)
	}

	frames, err := decodeSamples(data, format, bits, channels)
	if err != nil {
		return Clip{}, fmt.Errorf("%s: %w", path, err)
	}
	return Clip{Samples: frames, SampleRate: sampleRate}, nil
}

// ErrUnsupported reports a WAV encoding this decoder does not handle.
var ErrUnsupported = errors.New("unsupported WAV encoding")

// decodeSamples converts interleaved PCM/float samples to mono float64.
func decodeSamples(data []byte, format, bits, channels int) ([]float64, error) {
	bytesPerSample := bits / 8
	if bytesPerSample == 0 {
		return nil, fmt.Errorf("%w: %d-bit", ErrUnsupported, bits)
	}
	stride := bytesPerSample * channels
	n := len(data) / stride

	read := func(b []byte) (float64, error) {
		switch {
		case format == formatFloat && bits == 32:
			return float64(math.Float32frombits(binary.LittleEndian.Uint32(b))), nil
		case format == formatFloat && bits == 64:
			return math.Float64frombits(binary.LittleEndian.Uint64(b)), nil
		case format == formatPCM && bits == 8:
			return (float64(b[0]) - 128) / 128, nil // 8-bit WAV is unsigned
		case format == formatPCM && bits == 16:
			return float64(int16(binary.LittleEndian.Uint16(b))) / 32768, nil
		case format == formatPCM && bits == 24:
			v := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16
			if v&0x800000 != 0 {
				v -= 0x1000000 // sign-extend
			}
			return float64(v) / 8388608, nil
		case format == formatPCM && bits == 32:
			return float64(int32(binary.LittleEndian.Uint32(b))) / 2147483648, nil
		}
		return 0, fmt.Errorf("%w: format %d, %d-bit", ErrUnsupported, format, bits)
	}

	out := make([]float64, n)
	for i := 0; i < n; i++ {
		base := i * stride
		var sum float64
		for c := 0; c < channels; c++ {
			v, err := read(data[base+c*bytesPerSample:])
			if err != nil {
				return nil, err
			}
			sum += v
		}
		out[i] = sum / float64(channels)
	}
	return out, nil
}
