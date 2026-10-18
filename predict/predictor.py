"""Next-token predictor: given a prefix, guess the next character/token.

    python predict/predictor.py "Toke"
    python predict/predictor.py "Linu" --top 5

Two heads answer the same question:

* ngram  — character statistics counted from source code (instant, CPU only)
* neural — the trained Larme checkpoint, when it is present (slower, needs torch)

Every answer reports which head produced it and with what confidence, so a hit
is never confused with a lucky guess.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_NGRAM = ROOT / "ngram.json.gz"


class NgramPredictor:
    def __init__(self, path: Path = DEFAULT_NGRAM):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.order = payload["order"]
        self.table: dict[str, dict[str, int]] = payload["table"]
        self.bigram: dict[str, dict[str, int]] = payload.get("bigram", {})
        self.vocab: dict[str, int] = payload.get("vocab", {})

    def _vocab_predict(self, prefix: str, top: int) -> list[tuple[str, float]]:
        """Complete a partial identifier with the next character of real words."""
        if not prefix:
            return []
        tail = prefix.split()[-1].split("/")[-1] if prefix.split() else prefix
        rows: dict[str, int] = {}
        for word, count in self.vocab.items():
            if len(word) > len(tail) and word.startswith(tail):
                nxt = word[len(tail)]
                rows[nxt] = rows.get(nxt, 0) + count
        if not rows:
            # Case-insensitive pass, so "Stu" can complete "studio".
            lowered = {word.lower(): count for word, count in self.vocab.items()}
            low_tail = tail.lower()
            for word, count in lowered.items():
                if len(word) > len(low_tail) and word.startswith(low_tail):
                    nxt = word[len(low_tail)]
                    rows[nxt] = rows.get(nxt, 0) + count
        if not rows:
            return []
        total = sum(rows.values())
        ranked = sorted(rows.items(), key=lambda kv: -kv[1])[:top]
        return [(char, count / total) for char, count in ranked]

    def predict(self, prefix: str, top: int = 5) -> list[tuple[str, float]]:
        ctx = ("\x02" * (self.order - 1) + prefix)[-(self.order - 1):]
        tail = prefix.split()[-1] if prefix.split() else prefix
        identifier = bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{2,}", tail))
        if identifier:
            completed = self._vocab_predict(prefix, top)
            if completed and completed[0][1] >= 0.25:
                return completed
        row = self.table.get(ctx)
        if not row:
            if identifier:
                completed = self._vocab_predict(prefix, top)
                if completed:
                    return completed
            # Back off to a shorter suffix, then to the character bigram table.
            for back in range(self.order - 2, 1, -1):
                row = self.table.get(ctx[-back:])
                if row:
                    break
        if not row and prefix:
            row = self.bigram.get(prefix[-1])
        if not row:
            return []
        total = sum(row.values())
        ranked = sorted(row.items(), key=lambda kv: -kv[1])[:top]
        return [(char, count / total) for char, count in ranked]


class NeuralPredictor:
    def __init__(self, ckpt: str):
        import torch

        from model import GPT, ModelConfig
        from tokenizer import CharTokenizer

        state = torch.load(ckpt, map_location="cpu")
        self.model = GPT(ModelConfig(**state["config"]))
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.tok = CharTokenizer(state["vocab"])
        self.torch = torch

    def predict(self, prefix: str, top: int = 5) -> list[tuple[str, float]]:
        ids = [self.tok.bos_id] + self.tok.encode(prefix)
        idx = self.torch.tensor([ids])
        with self.torch.no_grad():
            logits = self.model(idx)[:, -1, :][0]
        probs = self.torch.softmax(logits, dim=-1)
        values, indices = probs.topk(top)
        return [(self.tok.decode([int(i)]), float(v)) for v, i in zip(values, indices)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = {}
    results["ngram"] = NgramPredictor().predict(args.prefix, args.top)
    if args.ckpt:
        results["neural"] = NeuralPredictor(args.ckpt).predict(args.prefix, args.top)

    if args.json:
        print(json.dumps(results, ensure_ascii=False))
        return 0

    print(f"prefix: {args.prefix!r}")
    for head, rows in results.items():
        for char, prob in rows:
            shown = {"\n": "\\n", "\t": "\\t", " ": "␣"}.get(char, char)
            bar = "#" * max(1, round(math.log10(max(prob, 1e-6)) + 6))
            print(f"  {head:6s} -> {shown!r:6s} p={prob:.3f} {bar}")
        if not rows:
            print(f"  {head:6s} -> (no statistics for this prefix)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
