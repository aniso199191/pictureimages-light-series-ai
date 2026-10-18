"""Pretrain the small decoder-only model on a uint16 token stream.

CPU-friendly: bf16 autocast for the forward pass, optional 8-bit blockwise
AdamW so a ~90M model fits under a 3.8 GB per-process memory cap.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, ModelConfig

torch.set_num_threads(int(os.environ.get("THREADS", "4")))


class Adam8bit:
    """AdamW with int8 blockwise m/v. Cuts optimizer memory ~4x."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01, block=8192):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.wd = weight_decay
        self.block = block
        self.step_count = 0
        self.m8, self.ms, self.v8, self.vs = [], [], [], []
        for p in self.params:
            n = p.numel()
            blocks = (n + block - 1) // block
            self.m8.append(torch.zeros(n, dtype=torch.int8))
            self.ms.append(torch.ones(blocks, dtype=torch.float32))
            self.v8.append(torch.zeros(n, dtype=torch.int8))
            self.vs.append(torch.ones(blocks, dtype=torch.float32))

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        c1 = 1 - self.b1 ** self.step_count
        c2 = 1 - self.b2 ** self.step_count
        block = self.block
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue
            if torch.isnan(g).any() or torch.isinf(g).any():
                print("warn: non-finite gradient, step skipped for one tensor", flush=True)
                continue
            flat = p.data.view(-1)
            gf = g.view(-1)
            n = p.numel()
            m8, ms, v8, vs = self.m8[i], self.ms[i], self.v8[i], self.vs[i]
            for j in range(len(ms)):
                s = j * block
                e = min(n, s + block)
                m = m8[s:e].float() * ms[j]
                v = v8[s:e].float() * vs[j]
                gs = gf[s:e]
                m.mul_(self.b1).add_(gs, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(gs, gs, value=1 - self.b2)
                if self.wd:
                    flat[s:e].mul_(1 - self.lr * self.wd)
                denom = (v / c2).sqrt().add_(1e-8)
                flat[s:e].addcdiv_(m / c1, denom, value=-self.lr)
                sm = (m.abs().max() / 127.0).clamp(min=1e-6)
                m8[s:e] = (m / sm).round().clamp(-127, 127).to(torch.int8)
                ms[j] = sm
                sv = (v.abs().max() / 127.0).clamp(min=1e-6)
                v8[s:e] = (v / sv).round().clamp(-127, 127).to(torch.int8)
                vs[j] = sv


def get_batch(data: np.memmap, batch: int, block: int):
    n = len(data) - block - 1
    starts = np.random.randint(0, n, size=batch)
    x = np.stack([data[i : i + block] for i in starts]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + block] for i in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model, data, batch=8, iters=20):
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, batch, model.cfg.block_size)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total += float(loss)
    model.train()
    return total / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="models/base.pt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--head", type=int, default=12)
    parser.add_argument("--embd", type=int, default=768)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log", default="logs/train.jsonl")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    data_dir = Path(args.data)
    train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))["vocab"]

    cfg = ModelConfig(
        vocab_size=len(vocab), n_layer=args.layer, n_head=args.head,
        n_embd=args.embd, block_size=args.block,
    )
    model = GPT(cfg)
    print(f"config={cfg.to_dict()}", flush=True)
    print(f"params={model.num_params():,}", flush=True)

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        print(f"resumed from {args.resume}", flush=True)

    opt = Adam8bit(model.parameters(), lr=args.lr)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    tokens_per_step = args.batch * args.block
    model.train()
    t0 = time.monotonic()
    for step in range(1, args.steps + 1):
        if step <= args.warmup:
            lr = args.lr * step / args.warmup
        else:
            progress = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        opt.lr = lr

        x, y = get_batch(train, args.batch, args.block)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        if step % 10 == 0:
            elapsed = time.monotonic() - t0
            rate = tokens_per_step * step / elapsed
            print(f"step {step}/{args.steps} loss {float(loss):.4f} lr {lr:.2e} {rate:.0f} tok/s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val)
            record = {
                "step": step,
                "train_loss": float(loss),
                "val_loss": val_loss,
                "lr": lr,
                "tokens": tokens_per_step * step,
                "seconds": round(time.monotonic() - t0, 1),
            }
            with Path(args.log).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"eval step {step} val_loss {val_loss:.4f}", flush=True)
            torch.save({
                "model": model.state_dict(),
                "config": cfg.to_dict(),
                "vocab": vocab,
                "step": step,
                "val_loss": val_loss,
            }, out_path)
            print(f"saved {out_path}", flush=True)

    torch.save({
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "vocab": vocab,
        "step": args.steps,
        "val_loss": evaluate(model, val),
    }, out_path)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
