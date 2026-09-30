#!/usr/bin/env python3
"""
Generate the golden files the Go tests check against.

Everything here comes from the Python side — NISQA's own librosa front-end and
the original PyTorch model — so the Go tests are comparing against the reference
implementation rather than against themselves.

    uv run --project ../nisqa-poc python tools/make_testdata.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
from export_onnx import POC, VENDOR, load_model  # noqa: E402

OUT = HERE / "testdata"
CLIP_SECONDS = 2.0
RESAMPLE_SECONDS = 1.0
LABELS = ["mos", "noi", "dis", "col", "loud"]


def main() -> None:
    sys.path.insert(0, str(VENDOR))
    import librosa
    from nisqa.NISQA_lib import get_librosa_melspec, segment_specs

    OUT.mkdir(exist_ok=True)
    meta = json.loads((HERE / "models" / "nisqa_dim.json").read_text())

    source = POC / "samples" / "clean.wav"
    x, sr = sf.read(source, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)

    # 1. The clip the Go tests score.
    clip = x[: int(CLIP_SECONDS * sr)]
    clip_path = OUT / "clip.wav"
    sf.write(clip_path, clip, sr, subtype="PCM_16")

    # 2. Its mel-spectrogram, straight out of NISQA's own front-end.
    spec = get_librosa_melspec(
        str(clip_path),
        sr=meta["sr"],
        n_fft=meta["n_fft"],
        hop_length=meta["hop_length_s"],
        win_length=meta["win_length_s"],
        n_mels=meta["n_mels"],
        fmax=meta["fmax"],
    )
    spec.astype("<f4").tofile(OUT / "clip_melspec.f32")

    # 3. The scores the original PyTorch model gives that clip.
    model, _ = load_model("dim")
    segments, n_wins = segment_specs(str(clip_path), spec, meta["seg_length"], meta["seg_hop_length"], None)
    with torch.no_grad():
        scores = model(segments.to(torch.float32).unsqueeze(0), torch.tensor([int(n_wins)])).numpy()[0]

    # 4. A 16 kHz clip plus librosa's own 48 kHz resampling of it, so the Go
    #    resampler can be measured against soxr rather than against itself.
    short = x[: int(RESAMPLE_SECONDS * sr)]
    down = librosa.resample(short, orig_sr=sr, target_sr=16000)
    sf.write(OUT / "clip_16k.wav", down, 16000, subtype="PCM_16")
    reloaded, _ = sf.read(OUT / "clip_16k.wav", dtype="float64")
    librosa.resample(reloaded, orig_sr=16000, target_sr=48000).astype("<f4").tofile(
        OUT / "clip_16k_upsampled.f32"
    )

    manifest = {
        "clip": {
            "path": "clip.wav",
            "sample_rate": sr,
            "samples": len(clip),
            "melspec": {"path": "clip_melspec.f32", "n_mels": spec.shape[0], "n_frames": spec.shape[1]},
            "segments": int(n_wins),
            "scores": {k: float(v) for k, v in zip(LABELS, scores)},
        },
        "resample": {
            "path": "clip_16k.wav",
            "from_rate": 16000,
            "to_rate": 48000,
            "reference": "clip_16k_upsampled.f32",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"\n{len(list(OUT.iterdir()))} files, {total / 1024:.0f} KB in {OUT}")


if __name__ == "__main__":
    main()
