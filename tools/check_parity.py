#!/usr/bin/env python3
"""
Run the Go port and the Python POC over the same clips and report the difference.

This is the end-to-end claim the port rests on, so it is a script rather than a
number pasted into a README: decode, resample, mel front-end and network all
have to agree, for every model.

    uv run --project ../nisqa-poc python tools/check_parity.py
    uv run --project ../nisqa-poc python tools/check_parity.py --audio /tmp/other-clips
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
POC = HERE.parent / "nisqa-poc"

DIMENSIONS = ["mos", "noi", "dis", "col", "loud"]


def read_csv(path: Path, key_suffix: str = "") -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            name = Path(row["deg"]).name
            rows[name] = {
                d: float(row[d + key_suffix])
                for d in DIMENSIONS
                if (d + key_suffix) in row and row[d + key_suffix] != ""
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", default=str(POC / "samples"), help="folder of wav files to compare on")
    parser.add_argument("--models", nargs="*", default=None, help="which models to check (default: all exported)")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    opts = parser.parse_args()

    models = opts.models or sorted(p.stem.removeprefix("nisqa_") for p in (HERE / "models").glob("*.onnx"))
    failures = []

    for model in models:
        meta = json.loads((HERE / "models" / f"nisqa_{model}.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            go_csv, py_csv = Path(tmp) / "go.csv", Path(tmp) / "py.csv"
            subprocess.run(
                ["go", "run", ".", opts.audio, "--model", model, "--csv", str(go_csv)],
                cwd=HERE, check=True, stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                ["uv", "run", "predict.py", opts.audio, "--model", model, "--csv", str(py_csv)],
                cwd=POC, check=True, stdout=subprocess.DEVNULL,
            )
            go, py = read_csv(go_csv), read_csv(py_csv, "_pred")

        print(f"\n=== {model} ({', '.join(meta['outputs'])}) ===")
        worst = 0.0
        for name in sorted(go):
            deltas = {d: abs(go[name][d] - py[name][d]) for d in meta["outputs"]}
            biggest = max(deltas.values())
            worst = max(worst, biggest)
            scores = "  ".join(f"{d}={go[name][d]:.2f}" for d in meta["outputs"])
            print(f"  {name:<18} {scores}   max|Δ|={biggest:.2e}")
        verdict = "OK" if worst <= opts.tolerance else "FAIL"
        print(f"  worst difference: {worst:.3e}  [{verdict}]")
        if worst > opts.tolerance:
            failures.append(model)

    if failures:
        sys.exit(f"\nFAIL: {', '.join(failures)} exceeded tolerance {opts.tolerance}")
    print(f"\nOK: Go and Python agree within {opts.tolerance} on every model")


if __name__ == "__main__":
    main()
