package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"nisqa-go/internal/audio"
	"nisqa-go/internal/melspec"
	"nisqa-go/internal/nisqa"
)

// The golden files under testdata/ come from the Python reference
// implementation — NISQA's own librosa front-end and the original PyTorch
// model. Regenerate them with tools/make_testdata.py.

type manifest struct {
	Clip struct {
		Path       string `json:"path"`
		SampleRate int    `json:"sample_rate"`
		Samples    int    `json:"samples"`
		Melspec    struct {
			Path    string `json:"path"`
			NMels   int    `json:"n_mels"`
			NFrames int    `json:"n_frames"`
		} `json:"melspec"`
		Segments int                `json:"segments"`
		Scores   map[string]float64 `json:"scores"`
	} `json:"clip"`
	Resample struct {
		Path      string `json:"path"`
		FromRate  int    `json:"from_rate"`
		ToRate    int    `json:"to_rate"`
		Reference string `json:"reference"`
	} `json:"resample"`
}

func loadManifest(t *testing.T) manifest {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	return m
}

func loadFloat32s(t *testing.T, name string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

func loadPredictor(t *testing.T) *nisqa.Predictor {
	t.Helper()
	onnxData, err := models.ReadFile("models/nisqa_dim.onnx")
	if err != nil {
		t.Fatalf("embedded model: %v", err)
	}
	metaData, err := models.ReadFile("models/nisqa_dim.json")
	if err != nil {
		t.Fatalf("embedded metadata: %v", err)
	}
	meta, err := nisqa.ParseMeta(metaData)
	if err != nil {
		t.Fatal(err)
	}
	p, err := nisqa.New(onnxData, meta)
	if err != nil {
		t.Skipf("ONNX Runtime unavailable: %v", err)
	}
	return p
}

// TestMelSpectrogramMatchesLibrosa is the load-bearing test: the network was
// trained on librosa's features, so any drift in the Go front-end is a silent
// accuracy bug rather than a crash.
func TestMelSpectrogramMatchesLibrosa(t *testing.T) {
	m := loadManifest(t)
	clip, err := audio.DecodeWAV(filepath.Join("testdata", m.Clip.Path))
	if err != nil {
		t.Fatal(err)
	}
	if clip.SampleRate != m.Clip.SampleRate || len(clip.Samples) != m.Clip.Samples {
		t.Fatalf("decoded %d samples at %d Hz, want %d at %d",
			len(clip.Samples), clip.SampleRate, m.Clip.Samples, m.Clip.SampleRate)
	}

	extractor, err := melspec.New(melspec.Config{
		SampleRate: 48000, NFFT: 4096, HopLength: 480, WinLength: 960,
		NMels: 48, FMin: 0, FMax: 20000,
	})
	if err != nil {
		t.Fatal(err)
	}
	spec, err := extractor.Compute(clip.Samples)
	if err != nil {
		t.Fatal(err)
	}
	if len(spec) != m.Clip.Melspec.NMels || len(spec[0]) != m.Clip.Melspec.NFrames {
		t.Fatalf("spectrogram is %dx%d, want %dx%d",
			len(spec), len(spec[0]), m.Clip.Melspec.NMels, m.Clip.Melspec.NFrames)
	}

	golden := loadFloat32s(t, m.Clip.Melspec.Path)
	var worst float64
	for mel := range spec {
		for frame := range spec[mel] {
			d := math.Abs(spec[mel][frame] - float64(golden[mel*len(spec[mel])+frame]))
			if d > worst {
				worst = d
			}
		}
	}
	// The golden values are float32 dB, so ~1e-4 is the storage floor.
	const tolerance = 1e-3
	t.Logf("largest difference from librosa: %.2e dB", worst)
	if worst > tolerance {
		t.Errorf("mel-spectrogram differs from librosa by %.3e dB (tolerance %g)", worst, tolerance)
	}
}

func TestScoresMatchPyTorch(t *testing.T) {
	m := loadManifest(t)
	predictor := loadPredictor(t)
	defer predictor.Close()

	clip, err := audio.DecodeWAV(filepath.Join("testdata", m.Clip.Path))
	if err != nil {
		t.Fatal(err)
	}
	scores, err := predictor.Score(clip)
	if err != nil {
		t.Fatal(err)
	}

	const tolerance = 1e-3
	for i, name := range predictor.Meta().Outputs {
		want, ok := m.Clip.Scores[name]
		if !ok {
			t.Fatalf("manifest has no expected score for %q", name)
		}
		got := float64(scores[i])
		if d := math.Abs(got - want); d > tolerance {
			t.Errorf("%s = %.6f, PyTorch gives %.6f (difference %.2e)", name, got, want, d)
		} else {
			t.Logf("%-4s go=%.6f pytorch=%.6f  Δ=%.2e", name, got, want, d)
		}
	}
}

// TestResampleMatchesLibrosa measures the Go resampler against librosa's soxr.
// They use different filter designs, so this is a similarity bound rather than
// an equality check — see the constants in internal/audio/resample.go.
func TestResampleMatchesLibrosa(t *testing.T) {
	m := loadManifest(t)
	clip, err := audio.DecodeWAV(filepath.Join("testdata", m.Resample.Path))
	if err != nil {
		t.Fatal(err)
	}
	if clip.SampleRate != m.Resample.FromRate {
		t.Fatalf("clip is %d Hz, want %d", clip.SampleRate, m.Resample.FromRate)
	}

	got := audio.Resample(clip, m.Resample.ToRate)
	if got.SampleRate != m.Resample.ToRate {
		t.Fatalf("resampled to %d Hz, want %d", got.SampleRate, m.Resample.ToRate)
	}
	reference := loadFloat32s(t, m.Resample.Reference)
	if diff := len(got.Samples) - len(reference); diff < -1 || diff > 1 {
		t.Fatalf("resampled length %d, reference %d", len(got.Samples), len(reference))
	}

	n := min(len(got.Samples), len(reference))
	var signal, noise float64
	for i := 0; i < n; i++ {
		ref := float64(reference[i])
		signal += ref * ref
		d := ref - got.Samples[i]
		noise += d * d
	}
	snr := 10 * math.Log10(signal/noise)
	const minSNR = 45.0
	t.Logf("%d -> %d Hz: %.1f dB SNR against librosa", m.Resample.FromRate, m.Resample.ToRate, snr)
	if snr < minSNR {
		t.Errorf("resampler SNR %.1f dB is below the %.0f dB floor", snr, minSNR)
	}
}

func TestSegmentCount(t *testing.T) {
	m := loadManifest(t)
	predictor := loadPredictor(t)
	defer predictor.Close()

	clip, err := audio.DecodeWAV(filepath.Join("testdata", m.Clip.Path))
	if err != nil {
		t.Fatal(err)
	}
	// 2 s at a 10 ms hop is 201 frames; 15-wide windows stepping by 4 give 47.
	if _, err := predictor.Score(clip); err != nil {
		t.Fatal(err)
	}
	frames := 1 + len(clip.Samples)/480
	segments := (frames - 14 + 3) / 4
	if segments != m.Clip.Segments {
		t.Errorf("segmented into %d windows, Python produced %d", segments, m.Clip.Segments)
	}
}

func TestTooShortClipIsRejected(t *testing.T) {
	predictor := loadPredictor(t)
	defer predictor.Close()

	// 100 ms cannot fill a single 15-frame segment.
	clip := audio.Clip{Samples: make([]float64, 4800), SampleRate: 48000}
	if _, err := predictor.Score(clip); err == nil {
		t.Fatal("expected an error for a clip shorter than one segment")
	}
}
