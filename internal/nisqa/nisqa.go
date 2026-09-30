// Package nisqa turns an audio clip into NISQA quality scores: mel front-end,
// segmentation, then the exported ONNX network.
package nisqa

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"nisqa-go/internal/audio"
	"nisqa-go/internal/melspec"
)

// Meta is written alongside the .onnx file by tools/export_onnx.py. It pins the
// front-end parameters to whatever the checkpoint was trained with, so the Go
// and Python pipelines cannot drift apart silently.
type Meta struct {
	Model        string   `json:"model"`
	SampleRate   int      `json:"sr"`
	NFFT         int      `json:"n_fft"`
	HopLengthS   float64  `json:"hop_length_s"`
	WinLengthS   float64  `json:"win_length_s"`
	NMels        int      `json:"n_mels"`
	FMax         float64  `json:"fmax"`
	SegLength    int      `json:"seg_length"`
	SegHopLength int      `json:"seg_hop_length"`
	Outputs      []string `json:"outputs"`
}

func ParseMeta(data []byte) (Meta, error) {
	var m Meta
	if err := json.Unmarshal(data, &m); err != nil {
		return Meta{}, fmt.Errorf("nisqa: bad model metadata: %w", err)
	}
	return m, nil
}

type Predictor struct {
	meta    Meta
	mel     *melspec.Extractor
	session *ort.DynamicAdvancedSession
}

func New(onnxData []byte, meta Meta) (*Predictor, error) {
	if err := initRuntime(); err != nil {
		return nil, err
	}
	mel, err := melspec.New(melspec.Config{
		SampleRate: meta.SampleRate,
		NFFT:       meta.NFFT,
		HopLength:  int(math.Round(meta.HopLengthS * float64(meta.SampleRate))),
		WinLength:  int(math.Round(meta.WinLengthS * float64(meta.SampleRate))),
		NMels:      meta.NMels,
		FMin:       0,
		FMax:       meta.FMax,
	})
	if err != nil {
		return nil, err
	}
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		onnxData, []string{"segments"}, []string{"scores"}, nil)
	if err != nil {
		return nil, fmt.Errorf("nisqa: could not create ONNX session: %w", err)
	}
	return &Predictor{meta: meta, mel: mel, session: session}, nil
}

func (p *Predictor) Close() error { return p.session.Destroy() }

func (p *Predictor) Meta() Meta { return p.meta }

// Score resamples the clip if needed, builds the segmented mel-spectrogram and
// runs the network. The returned values line up with Meta().Outputs.
func (p *Predictor) Score(clip audio.Clip) ([]float32, error) {
	clip = audio.Resample(clip, p.meta.SampleRate)

	spec, err := p.mel.Compute(clip.Samples)
	if err != nil {
		return nil, err
	}
	segments, count, err := p.segment(spec)
	if err != nil {
		return nil, err
	}

	shape := ort.NewShape(int64(count), 1, int64(p.meta.NMels), int64(p.meta.SegLength))
	input, err := ort.NewTensor(shape, segments)
	if err != nil {
		return nil, fmt.Errorf("nisqa: input tensor: %w", err)
	}
	defer input.Destroy()

	output, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(len(p.meta.Outputs))))
	if err != nil {
		return nil, fmt.Errorf("nisqa: output tensor: %w", err)
	}
	defer output.Destroy()

	if err := p.session.Run([]ort.Value{input}, []ort.Value{output}); err != nil {
		return nil, fmt.Errorf("nisqa: inference failed: %w", err)
	}
	scores := make([]float32, len(p.meta.Outputs))
	copy(scores, output.GetData())
	return scores, nil
}

// segment slides a SegLength-wide window over the spectrogram, stepping by
// SegHopLength, and flattens the result to the (T, 1, n_mels, seg_length)
// tensor the network expects.
func (p *Predictor) segment(spec [][]float64) ([]float32, int, error) {
	nFrames := len(spec[0])
	available := nFrames - (p.meta.SegLength - 1)
	if available < 1 {
		return nil, 0, fmt.Errorf(
			"nisqa: clip too short — %d spectrogram frames, need at least %d (about %.2f s)",
			nFrames, p.meta.SegLength, float64(p.meta.SegLength)*p.meta.HopLengthS)
	}
	count := (available + p.meta.SegHopLength - 1) / p.meta.SegHopLength // ceil

	out := make([]float32, count*p.meta.NMels*p.meta.SegLength)
	i := 0
	for t := 0; t < count; t++ {
		start := t * p.meta.SegHopLength
		for m := 0; m < p.meta.NMels; m++ {
			row := spec[m]
			for j := 0; j < p.meta.SegLength; j++ {
				out[i] = float32(row[start+j])
				i++
			}
		}
	}
	return out, count, nil
}

// --- ONNX Runtime bootstrap --------------------------------------------------

var (
	runtimeOnce sync.Once
	runtimeErr  error
)

// initRuntime locates libonnxruntime and initialises the shared environment.
// ONNXRUNTIME_LIB overrides the search.
func initRuntime() error {
	runtimeOnce.Do(func() {
		path := os.Getenv("ONNXRUNTIME_LIB")
		if path == "" {
			path = findLibrary()
		}
		if path == "" {
			runtimeErr = fmt.Errorf(
				"nisqa: could not find the ONNX Runtime shared library — install it " +
					"(macOS: brew install onnxruntime) or set ONNXRUNTIME_LIB to its path")
			return
		}
		ort.SetSharedLibraryPath(path)
		if err := ort.InitializeEnvironment(); err != nil {
			runtimeErr = fmt.Errorf("nisqa: could not initialise ONNX Runtime from %s: %w", path, err)
		}
	})
	return runtimeErr
}

func findLibrary() string {
	var candidates []string
	switch runtime.GOOS {
	case "darwin":
		candidates = []string{
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
		}
	case "windows":
		candidates = []string{"onnxruntime.dll"}
	default:
		candidates = []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
		}
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
		// Homebrew and apt both ship versioned names alongside the symlink.
		if matches, _ := filepath.Glob(c + ".*"); len(matches) > 0 {
			return matches[0]
		}
	}
	return ""
}
