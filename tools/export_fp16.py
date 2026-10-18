"""Export a checkpoint as FP16 weights for release.

    python tools/export_fp16.py --ckpt train/models/sft.pt --out dist/larme-debug-fp16.pt

The exported file keeps the config and the tokenizer vocabulary, so it loads
with the same code path, but every weight tensor is stored as float16.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from model import GPT, ModelConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    weights = {k: v.half() for k, v in state["model"].items()}
    payload = {"config": state["config"], "vocab": state["vocab"],
               "step": state.get("step"), "dtype": "float16", "model": weights}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    raw = out.read_bytes()
    # Tied embeddings are serialized twice; count unique parameters on a meta model.
    with torch.device("meta"):
        probe = GPT(ModelConfig(**state["config"]))
    params = sum(p.numel() for p in probe.parameters())
    print(json.dumps({
        "out": str(out),
        "params": params,
        "file_mib": round(len(raw) / 2**20, 1),
        "bytes_per_param": round(len(raw) / params, 2),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": payload["dtype"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
