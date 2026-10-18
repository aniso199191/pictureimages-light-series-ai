"""Score a fine-tuned checkpoint on the held-out instruction pairs.

    python tools/score_sft.py --ckpt train/models/sft.pt --dev data/distill_dev.jsonl

Reports held-out perplexity on the assistant span and a character-level F1
between generated answers and the reference answers. The F1 is a rough signal
(paraphrases score low even when correct), so it is reported as a comparison
number, never as an accuracy claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from model import GPT, ModelConfig  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def char_f1(pred: str, ref: str) -> float:
    from collections import Counter

    pred_chars, ref_chars = Counter(pred), Counter(ref)
    overlap = sum((pred_chars & ref_chars).values())
    if not overlap:
        return 0.0
    precision = overlap / max(1, sum(pred_chars.values()))
    recall = overlap / max(1, sum(ref_chars.values()))
    return 2 * precision * recall / (precision + recall)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--max-new", type=int, default=160)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    model = GPT(ModelConfig(**state["config"]))
    model.load_state_dict(state["model"])
    model.eval()
    tok = CharTokenizer(state["vocab"])

    pairs = []
    for line in Path(args.dev).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            pairs.append((record.get("instruction") or record.get("question"),
                          record.get("response") or record.get("answer")))
    pairs = [(q, a) for q, a in pairs if q and a][:args.limit]

    rows, losses = [], []
    for question, reference in pairs:
        ids = [tok.bos_id] + tok.encode("<user>") + tok.encode(question) + tok.encode("<assistant>")
        idx = torch.tensor([ids])
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=args.max_new, temperature=0.7,
                                 top_k=40, eos_id=tok.eos_id)
        answer = tok.decode(out[0].tolist()[len(ids):]).split("<eos>")[0].strip()

        # Teacher-forced loss over the reference answer only: inputs are the
        # prompt plus the answer, targets are the answer plus EOS, with the
        # prompt positions masked out (-100).
        answer_ids = tok.encode(reference) + [tok.eos_id]
        seq = ids + answer_ids
        inputs = torch.tensor([seq[:-1]])
        labels = torch.tensor([[-100] * (len(ids) - 1) + answer_ids])
        with torch.no_grad():
            _, loss = model(inputs, labels)
        losses.append(float(loss))
        rows.append({"question": question, "reference": reference, "answer": answer,
                     "char_f1": round(char_f1(answer, reference), 3), "loss": round(float(loss), 3)})

    mean_loss = sum(losses) / max(1, len(losses))
    result = {
        "n": len(rows),
        "assistant_loss": round(mean_loss, 3),
        "assistant_ppl": round(float(torch.exp(torch.tensor(mean_loss))), 1),
        "mean_char_f1": round(sum(r["char_f1"] for r in rows) / max(1, len(rows)), 3),
        "samples": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
