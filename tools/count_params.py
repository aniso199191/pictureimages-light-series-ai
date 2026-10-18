"""Reproduce the parameter count of a config without allocating weights.

    python tools/count_params.py configs/series_9b.json

Runs on the meta device, so a 9B config costs no memory and no disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from model import GPT, ModelConfig  # noqa: E402


def main() -> int:
    cfg_path = Path(sys.argv[1])
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    from dataclasses import fields
    allowed = {f.name for f in fields(ModelConfig)}
    cfg = {k: v for k, v in raw.items() if k in allowed}
    torch.set_default_device("meta")
    model = GPT(ModelConfig(**cfg))
    total = sum(p.numel() for p in model.parameters())
    print(json.dumps({"config": str(cfg_path), "parameters": total,
                      "billions": round(total / 1e9, 3)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
