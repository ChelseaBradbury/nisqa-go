#!/usr/bin/env python3
"""
Export a pretrained NISQA model to ONNX so the Go port can run it.

The exported graph starts at the segmented mel-spectrogram:

    input   segments  float32[T, 1, n_mels, seg_length]
    output  scores    float32[1, 5]  (dim model: MOS, NOI, DIS, COL, LOUD)
            scores    float32[1, 1]  (mos_only / tts models)

Everything upstream of that — decode, resample, mel-spectrogram, segmentation —
is reimplemented in Go. Everything downstream is the network itself.

Two things have to be rewritten for the graph to export at all; both are checked
against the untouched upstream model before anything is written to disk.

Run it with the Python POC's environment:

    uv sync --project ../nisqa-poc --group onnx
    uv run --project ../nisqa-poc python tools/export_onnx.py
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent.parent
POC = HERE.parent / "nisqa-poc"
VENDOR = POC / "vendor" / "NISQA"

CHECKPOINTS = {
    "dim": "nisqa.tar",
    "mos_only": "nisqa_mos_only.tar",
    "tts": "nisqa_tts.tar",
}


# --- rewrite 1: adaptive max pooling -----------------------------------------
#
# AdaptCNN pools 48x15 feature maps down to 24x7, 12x5 and 6x3. Neither ONNX
# exporter can emit adaptive_max_pool2d when the output size doesn't divide the
# input size (15 -> 7 doesn't). The spatial dims are fixed here — only the
# segment count varies — so the same windows can be spelled out as slices.

def _windows(in_size: int, out_size: int) -> list[tuple[int, int]]:
    """The exact windows torch's adaptive pooling uses."""
    return [
        (math.floor(i * in_size / out_size), math.ceil((i + 1) * in_size / out_size))
        for i in range(out_size)
    ]


def _static_adaptive_max_pool2d(x: torch.Tensor, output_size) -> torch.Tensor:
    out_h, out_w = output_size
    in_h, in_w = int(x.shape[-2]), int(x.shape[-1])
    if in_w != out_w:
        cols = [x[..., s:e].amax(dim=-1, keepdim=True) for s, e in _windows(in_w, out_w)]
        x = torch.cat(cols, dim=-1)
    if in_h != out_h:
        rows = [x[..., s:e, :].amax(dim=-2, keepdim=True) for s, e in _windows(in_h, out_h)]
        x = torch.cat(rows, dim=-2)
    return x


@contextlib.contextmanager
def exportable_pooling():
    original = F.adaptive_max_pool2d
    F.adaptive_max_pool2d = _static_adaptive_max_pool2d
    try:
        yield
    finally:
        F.adaptive_max_pool2d = original


# --- rewrite 2: drop the padding masks ---------------------------------------

class SingleClip(nn.Module):
    """NISQA specialised to one clip per call.

    Upstream threads an `n_wins` tensor through every layer so a padded batch can
    be masked. With a batch of one there is no padding: every mask is all-true,
    `pack_padded_sequence` is an identity reshape, and the masked softmax is a
    plain softmax. Dropping that machinery is what makes the graph exportable.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.cnn = model.cnn.model
        self.time_dependency = model.time_dependency
        self.time_dependency_2 = model.time_dependency_2
        # NISQA_DIM has one pooling head per dimension; NISQA has a single one.
        self.pool_layers = getattr(model, "pool_layers", None) or nn.ModuleList([model.pool])

    def forward(self, segments: torch.Tensor) -> torch.Tensor:
        x = self.cnn(segments)          # (T, 1, n_mels, seg_len) -> (T, F)
        x = x.unsqueeze(0)              # (1, T, F)
        x = self._time_dependency(self.time_dependency, x)
        x = self._time_dependency(self.time_dependency_2, x)
        return torch.cat([self._pool(p.model, x) for p in self.pool_layers], dim=1)

    @staticmethod
    def _time_dependency(td: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Self-attention or LSTM over the segment sequence, unpadded."""
        model = getattr(td, "model", None)
        if not isinstance(model, nn.Module):
            return x                                      # td='skip'
        if hasattr(model, "lstm"):
            return model.lstm(x)[0]                       # packing is identity here
        return model(x, None)[0]                          # self-attention, no key mask

    @staticmethod
    def _pool(pool: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Pooling with the all-true mask elided. Dropout is eval-mode."""
        kind = type(pool).__name__
        if kind in ("PoolAttFF", "PoolAtt"):
            if kind == "PoolAttFF":
                att = pool.linear2(pool.activation(pool.linear1(x)))
                project = pool.linear3
            else:
                att = pool.linear1(x)
                project = pool.linear2
            att = F.softmax(att.transpose(2, 1), dim=2)   # (1, 1, T)
            return project(torch.bmm(att, x).squeeze(1))  # (1, out)
        if kind == "PoolLastStepBi":
            # Bidirectional LSTM output is [forward | backward]; upstream takes
            # the forward direction's last step and the backward direction's first.
            half = x.shape[-1] // 2
            return pool.linear(torch.cat((x[:, -1, :half], x[:, 0, half:]), dim=1))
        if kind == "PoolLastStep":
            return pool.linear(x[:, -1, :])
        raise NotImplementedError(f"pooling {kind} is not handled by the exporter")


def load_model(which: str):
    sys.path.insert(0, str(VENDOR))
    original_load = torch.load
    torch.load = lambda *a, **kw: original_load(*a, **{**kw, "weights_only": False})
    from nisqa.NISQA_model import nisqaModel

    args = {
        "mode": "predict_file",
        "pretrained_model": str(VENDOR / "weights" / CHECKPOINTS[which]),
        "deg": str(POC / "samples" / "clean.wav"),
        "output_dir": None,
        "ms_channel": None,
        "tr_bs_val": 1,
        "tr_num_workers": 0,
        "bs": 1,
        "num_workers": 0,
    }
    nisqa = nisqaModel(args)
    nisqa.model.eval()
    return nisqa.model, nisqa.args


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default="dim")
    parser.add_argument("--out", default=None, help="output .onnx path")
    parser.add_argument("--segments", type=int, default=64, help="segment count used for tracing")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    opts = parser.parse_args()

    model, args = load_model(opts.model)
    n_mels, seg_len = args["ms_n_mels"], args["ms_seg_length"]
    print(f"architecture: {model.name}, time-dependency: {args['td']}, pooling: {args['pool']}")
    wrapped = SingleClip(model).eval()

    torch.manual_seed(0)
    dummy = torch.randn(opts.segments, 1, n_mels, seg_len)

    with torch.no_grad():
        expected = model(dummy.unsqueeze(0), torch.tensor([opts.segments]))
        with exportable_pooling():
            actual = wrapped(dummy)
    delta = (actual - expected).abs().max().item()
    print(f"rewritten vs upstream model: max abs diff {delta:.3e}")
    if delta > opts.tolerance:
        raise SystemExit("refusing to export: rewritten model does not match upstream")

    out = Path(opts.out) if opts.out else HERE / "models" / f"nisqa_{opts.model}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    with exportable_pooling():
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(out),
            input_names=["segments"],
            output_names=["scores"],
            dynamic_shapes={"segments": {0: torch.export.Dim("n_segments", min=2)}},
            opset_version=17,
            dynamo=True,
            optimize=True,
            # One self-contained file so the Go binary can go:embed it.
            external_data=False,
        )
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")

    # The Go front-end has to reproduce these exactly.
    meta = {
        "model": opts.model,
        "sr": int(args["ms_sr"] or 48000),
        "n_fft": args["ms_n_fft"],
        "hop_length_s": args["ms_hop_length"],
        "win_length_s": args["ms_win_length"],
        "n_mels": args["ms_n_mels"],
        "fmax": args["ms_fmax"],
        "seg_length": args["ms_seg_length"],
        "seg_hop_length": args["ms_seg_hop_length"],
        "outputs": ["mos", "noi", "dis", "col", "loud"] if args["dim"] else ["mos"],
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
