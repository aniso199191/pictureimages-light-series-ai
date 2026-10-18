"""Score a checkpoint and print sample answers.

    python eval.py --ckpt models/sft.pt --prompt "光合作用是什么"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import GPT, ModelConfig
from tokenizer import CharTokenizer


@torch.no_grad()
def perplexity(model, path: Path, iters: int = 20, batch: int = 4):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    block = model.cfg.block_size
    total, count = 0.0, 0
    for _ in range(iters):
        n = len(data) - block - 1
        starts = np.random.randint(0, n, size=batch)
        x = torch.from_numpy(np.stack([data[i : i + block] for i in starts]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + block] for i in starts]).astype(np.int64))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total += float(loss)
        count += 1
    mean = total / max(1, count)
    return mean, float(np.exp(mean))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max-new", type=int, default=120)
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**state["config"])
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    tok = CharTokenizer(state["vocab"])

    result = {"ckpt": args.ckpt, "params": model.num_params(), "step": state.get("step")}
    val_path = Path(args.data) / "val.bin"
    if val_path.is_file():
        loss, ppl = perplexity(model, val_path)
        result["val_loss"] = round(loss, 4)
        result["val_ppl"] = round(ppl, 2)

    samples = []
    for prompt in args.prompt:
        ids = [tok.bos_id] + tok.encode("<user>") + tok.encode(prompt) + tok.encode("<assistant>")
        idx = torch.tensor([ids])
        out = model.generate(idx, max_new_tokens=args.max_new, temperature=0.8, top_k=40, eos_id=tok.eos_id)
        text = tok.decode(out[0].tolist()[len(ids):])
        samples.append({"prompt": prompt, "answer": text})
    result["samples"] = samples
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
