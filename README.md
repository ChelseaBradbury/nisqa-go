# NISQA POC — Go

A Go port of [`../nisqa-poc`](../nisqa-poc): same NISQA speech-quality model, same
scores, no Python at runtime.

The network is exported to ONNX once (offline, with PyTorch) and runs through ONNX
Runtime. Everything in front of the network — WAV decoding, resampling, the mel
spectrogram, segmentation — is reimplemented in Go against librosa's behaviour.
That front-end is the interesting half of the port: NISQA was trained on librosa's
exact features, so a front-end that is merely *reasonable* produces quietly wrong
scores rather than an error.

## Quickstart

```bash
brew install onnxruntime          # the one runtime dependency
cd nisqa-go
go build -o nisqa .
./nisqa ../nisqa-poc/samples/clean.wav
```

```bash
./nisqa /path/to/folder/              # every .wav in it
./nisqa a.wav b.wav --csv out.csv
./nisqa clip.wav --model mos_only     # or --model tts
```

The exported models are embedded in the binary with `go:embed`, so the 7 MB
executable plus `libonnxruntime` is the whole deployment. Set `ONNXRUNTIME_LIB` if
the library isn't in a standard location.

## Does it agree with Python?

Yes — to floating-point noise, on the same clips:

```
$ uv run --project ../nisqa-poc python tools/check_parity.py

=== dim (mos, noi, dis, col, loud) ===
  clean.wav          mos=4.72  noi=4.43  dis=4.71  col=4.45  loud=4.62   max|Δ|=5.00e-07
  clipped.wav        mos=1.42  noi=3.13  dis=3.17  col=1.77  loud=1.41   max|Δ|=1.30e-06
  ...
  worst difference: 1.400e-06  [OK]

OK: Go and Python agree within 0.001 on every model
```

Three separate checks back that up, each against the reference implementation
rather than against the port itself:

| Check | Measured |
|---|---|
| `tools/check_onnx.py` — exported graph vs upstream PyTorch | ≤ 2.1e-6 |
| `go test` — Go mel-spectrogram vs librosa | 8.6e-6 dB |
| `tools/check_parity.py` — Go binary vs Python POC, all 3 models | ≤ 1.4e-6 |

The remaining difference is the CSV's own precision, not the pipeline.

## Speed

| | one 15 s clip | seven 15 s clips |
|---|---|---|
| Go | 0.20 s | 1.48 s |
| Python | 2.9 s | 2.2 s |

Python's fixed cost is importing torch; its per-clip work is faster, because
librosa's FFT is vectorised across frames while this port does one frame at a
time. For a one-shot CLI or a service that scores a clip per request, the Go
startup cost (a few ms) is the number that matters. If per-clip throughput ever
did, the mel loop is the place to look — it's ~90% of the runtime and trivially
parallel across files.

## How it fits together

```
tools/export_onnx.py   PyTorch -> ONNX, offline, run once
models/*.onnx          the exported graphs (+ .json front-end config)
internal/audio         WAV decode, Kaiser-sinc resampling to 48 kHz
internal/melspec       librosa-compatible mel spectrogram
internal/dsp           radix-2 FFT, Hann and Kaiser windows
internal/nisqa         segmentation + ONNX Runtime session
main.go                CLI
```

The `.json` beside each model carries the front-end parameters (`n_fft`, hop,
mel count, segment width…) read out of the checkpoint at export time, and the Go
side builds its front-end from that rather than from hardcoded constants — so a
model trained with different features can't be scored with the wrong ones.

### Getting the model out of PyTorch

Two things had to be rewritten for NISQA to export at all. `tools/export_onnx.py`
checks both against the untouched upstream model before writing anything, and
refuses to export if they disagree (both currently match to 0.0):

- **Padding masks.** Upstream threads an `n_wins` tensor through every layer so a
  padded batch can be masked. Scoring one clip at a time means there is no
  padding: every mask is all-true, `pack_padded_sequence` is an identity reshape,
  and the masked softmax is a plain softmax. Dropping that makes the graph
  traceable.
- **Adaptive max pooling.** `AdaptCNN` pools 48×15 feature maps to 24×7, 12×5 and
  6×3, and neither ONNX exporter handles `adaptive_max_pool2d` when the output
  doesn't divide the input (15 → 7 doesn't). The spatial dimensions are fixed, so
  the same windows are spelled out as explicit slices.

## Regenerating things

```bash
uv sync --project ../nisqa-poc --group onnx                        # one-time
uv run --project ../nisqa-poc python tools/export_onnx.py --model dim
uv run --project ../nisqa-poc python tools/make_testdata.py        # golden files for go test
uv run --project ../nisqa-poc python tools/check_onnx.py           # export vs PyTorch
uv run --project ../nisqa-poc python tools/check_parity.py         # Go vs Python
```

`testdata/` (445 KB) holds a 2 s clip, its librosa mel-spectrogram, the scores
PyTorch gives it, and a librosa-resampled reference — all produced by the Python
side so `go test` compares against the real implementation.

## Limitations

- **WAV only.** PCM 8/16/24/32-bit and 32/64-bit float, any channel count
  (downmixed by averaging, as librosa does). The Python POC also reads FLAC, OGG
  and MP3 via libsndfile; adding those here means a decoder dependency, and the
  point of this port was to keep the runtime surface small.
- **Resampling is close, not identical.** librosa uses soxr; this uses a
  Kaiser-windowed sinc, whose transition band no parameter choice reproduces
  exactly. The constants in `internal/audio/resample.go` were swept against
  librosa's output for the best worst-case agreement. Effect on scores:

  | input | difference vs Python |
  |---|---|
  | 48 kHz (no resampling) | exact |
  | 44.1 kHz | ≤ 0.002 |
  | 16 kHz | ≤ 0.013 |
  | 8 kHz | ≤ 0.073 |

  Feed it 48 kHz and the question doesn't arise. The 8 kHz case is the hard one —
  six-fold upsampling of telephone-band speech, where the model is sensitive to
  exactly where the band edge falls.
- **ONNX Runtime is cgo.** Not a pure-Go binary, and it needs the shared library
  at runtime. There's no pure-Go runtime that would handle this graph.
- Training and fine-tuning are out of scope, same as the Python POC.
- Model weights are under the upstream licence — see
  `../nisqa-poc/vendor/NISQA/weights/LICENSE_model_weights` (non-commercial /
  research use).
