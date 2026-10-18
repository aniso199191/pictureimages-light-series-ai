"""Supervised fine-tuning on instruction/response pairs.

Loss is masked to the assistant span, so the model is not trained to
reproduce the user's question. Sequences longer than --block are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, ModelConfig
from tokenizer import CharTokenizer

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
IGNORE = -100


def build_example(tok: CharTokenizer, question: str, answer: str, block: int):
    prefix = [tok.bos_id] + tok.encode("<user>") + tok.encode(question) + tok.encode("<assistant>")
    body = tok.encode(answer) + [tok.eos_id]
    ids = prefix + body
    if len(ids) > block + 1:
        return None
    labels = [IGNORE] * len(prefix) + body
    pad = block + 1 - len(ids)
    ids = ids + [tok.pad_id] * pad
    labels = labels + [IGNORE] * pad
    return torch.tensor(ids[:-1]), torch.tensor(labels[1:])


def load_pairs(paths):
    pairs = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                q = record.get("instruction") or record.get("question") or ""
                a = record.get("response") or record.get("answer") or ""
                if q and a:
                    pairs.append((q, a))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--out", default="models/sft.pt")
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--log", default="logs/sft.jsonl")
    args = parser.parse_args()

    tok = CharTokenizer.load(Path(args.data) / "vocab.json")
    state = torch.load(args.init, map_location="cpu")
    cfg = ModelConfig(**state["config"])
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    block = min(args.block, cfg.block_size)
    print(f"loaded {args.init} params={model.num_params():,} block={block}", flush=True)

    pairs = load_pairs(args.pairs)
    examples = []
    skipped = 0
    for q, a in pairs:
        ex = build_example(tok, q, a, block)
        if ex is None:
            skipped += 1
        else:
            examples.append(ex)
    if not examples:
        print("no usable examples")
        return 1
    print(f"pairs={len(pairs)} usable={len(examples)} skipped_too_long={skipped}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(examples) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    t0 = time.monotonic()
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        for i in range(0, len(examples) - args.batch + 1, args.batch):
            batch = examples[i : i + args.batch]
            x = torch.stack([ex[0] for ex in batch])
            y = torch.stack([ex[1] for ex in batch])
            step += 1
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps)))
            for group in opt.param_groups:
                group["lr"] = lr
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            if step % 10 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {float(loss):.4f}", flush=True)
        record = {"epoch": epoch, "step": step, "loss": float(loss), "seconds": round(time.monotonic() - t0, 1)}
        with Path(args.log).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "vocab": tok.vocab,
                    "sft_epoch": epoch, "loss": float(loss)}, out_path)
        print(f"saved {out_path} after epoch {epoch}", flush=True)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
