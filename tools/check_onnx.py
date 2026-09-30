#!/usr/bin/env python3
"""
Check the exported ONNX graph against upstream PyTorch on real audio.

Builds the mel segments with NISQA's own dataset code, runs them through both
the original model and the exported graph, and reports the largest difference.
The segment counts here differ from the one used to trace the export, which is
the point: it checks the dynamic axis as well as the arithmetic.

    uv run --project ../nisqa-poc python tools/check_onnx.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
from export_onnx import POC, VENDOR, load_model  # noqa: E402

LABELS = ["mos", "noi", "dis", "col", "loud"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dim")
    parser.add_argument("--audio", nargs="*", default=None, help="wav files (default: the POC samples)")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    opts = parser.parse_args()

    import onnxruntime as ort
    sys.path.insert(0, str(VENDOR))
    from nisqa.NISQA_lib import get_librosa_melspec, segment_specs

    onnx_path = HERE / "models" / f"nisqa_{opts.model}.onnx"
    meta = json.loads(onnx_path.with_suffix(".json").read_text())
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model, _ = load_model(opts.model)

    files = [Path(f) for f in opts.audio] if opts.audio else sorted((POC / "samples").glob("*.wav"))
    if not files:
        raise SystemExit("no audio files to check")

    worst = 0.0
    for path in files:
        spec = get_librosa_melspec(
            str(path),
            sr=meta["sr"],
            n_fft=meta["n_fft"],
            hop_length=meta["hop_length_s"],
            win_length=meta["win_length_s"],
            n_mels=meta["n_mels"],
            fmax=meta["fmax"],
        )
        segments, n_wins = segment_specs(
            str(path), spec, meta["seg_length"], meta["seg_hop_length"], None
        )
        x = segments.to(torch.float32)

        with torch.no_grad():
            torch_out = model(x.unsqueeze(0), torch.tensor([int(n_wins)])).numpy()[0]
        onnx_out = session.run(["scores"], {"segments": x.numpy()})[0][0]

        delta = float(np.abs(torch_out - onnx_out).max())
        worst = max(worst, delta)
        scores = "  ".join(f"{k}={v:.3f}" for k, v in zip(LABELS, onnx_out))
        print(f"{path.name:<18} segments={int(n_wins):<4} {scores}   max|Δ|={delta:.2e}")

    print(f"\nworst difference across {len(files)} files: {worst:.3e}")
    if worst > opts.tolerance:
        raise SystemExit(f"FAIL: exceeds tolerance {opts.tolerance}")
    print("OK: exported graph matches upstream PyTorch")


if __name__ == "__main__":
    main()
