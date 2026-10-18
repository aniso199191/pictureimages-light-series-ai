"""Build docs/REPORT.md from a checkpoint: real numbers only.

    python tools/make_report.py --ckpt train/models/sft.pt --data train/data \
        --prompts prompts.txt --out docs/REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from eval import perplexity  # noqa: E402
from model import GPT, ModelConfig  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def generate(model, tok, question: str, max_new: int = 120, temperature: float = 0.8) -> str:
    ids = [tok.bos_id] + tok.encode("<user>") + tok.encode(question) + tok.encode("<assistant>")
    idx = torch.tensor([ids])
    out = model.generate(idx, max_new_tokens=max_new, temperature=temperature, top_k=40, eos_id=tok.eos_id)
    text = tok.decode(out[0].tolist()[len(ids):])
    return text.split("<eos>")[0].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default="train/data")
    parser.add_argument("--prompts", default="docs/prompts.txt")
    parser.add_argument("--out", default="docs/REPORT.md")
    parser.add_argument("--title", default="PictureImages Larme Debug — report card")
    parser.add_argument("--release-url", default="")
    parser.add_argument("--train-log", default="")
    parser.add_argument("--score-json", default="")
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    model = GPT(ModelConfig(**state["config"]))
    model.load_state_dict(state["model"])
    model.eval()
    tok = CharTokenizer(state["vocab"])

    params = model.num_params()
    val_path = Path(args.data) / "val.bin"
    val_loss = val_ppl = None
    if val_path.is_file():
        val_loss, val_ppl = perplexity(model, val_path)

    prompts = []
    prompt_path = Path(args.prompts)
    if prompt_path.is_file():
        prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    samples = [(p, generate(model, tok, p)) for p in prompts]

    lines = [
        f"# {args.title}",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from the checkpoint itself.",
        "",
        "## Measured facts",
        "",
        "| item | value |",
        "|---|---|",
        f"| parameters | {params:,} ({params / 1e6:.1f} M) |",
        f"| layers / heads / width | {model.cfg.n_layer} / {model.cfg.n_head} / {model.cfg.n_embd} |",
        f"| context (tokens) | {model.cfg.block_size} |",
        f"| trained steps | {state.get('step', 'unknown')} |",
        f"| checkpoint size | {Path(args.ckpt).stat().st_size / 2**20:.1f} MiB |",
    ]
    if val_loss is not None:
        lines.append(f"| LM loss on pretraining validation set | {val_loss:.4f} |")
        lines.append(f"| LM perplexity on the same set | {val_ppl:.2f} |")
    if state.get("sft_epoch"):
        lines.append(f"| instruction tuning | {state['sft_epoch']} epochs, {state.get('step', '?')} steps |")
    if args.release_url:
        lines.append(f"| weights | [{Path(args.ckpt).name}]({args.release_url}) |")

    if args.train_log and Path(args.train_log).is_file():
        import json as _json
        rows = []
        for line in Path(args.train_log).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = _json.loads(line)
            if "val_loss" in record:
                rows.append((record["step"], record["val_loss"], record.get("tokens", 0)))
        if rows:
            lines += ["", "## Pretraining curve (validation loss)", "",
                      "| step | tokens seen | val loss |", "|---:|---:|---:|"]
            lines += [f"| {s} | {t:,} | {v:.4f} |" for s, v, t in rows]

    if args.score_json and Path(args.score_json).is_file():
        import json as _json
        score = _json.loads(Path(args.score_json).read_text(encoding="utf-8"))
        lines += ["", "## Held-out instruction set", "",
                  f"| metric | value |", "|---|---|",
                  f"| pairs | {score.get('n')} |",
                  f"| answer loss | {score.get('assistant_loss')} |",
                  f"| answer perplexity | {score.get('assistant_ppl')} |",
                  f"| character F1 | {score.get('mean_char_f1')} |"]

    config_dir = ROOT / "configs"
    if config_dir.is_dir():
        from dataclasses import fields as _fields
        allowed = {f.name for f in _fields(ModelConfig)}
        rows = []
        for cfg_path in sorted(config_dir.glob("series_*.json")):
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            tiny = {k: v for k, v in raw.items() if k in allowed}
            with torch.device("meta"):
                probe = GPT(ModelConfig(**tiny))
            total = sum(p.numel() for p in probe.parameters())
            rows.append((cfg_path.name, total))
        if rows:
            lines += ["", "## Series sizes (architecture only, no weights)", "",
                      "| config | parameters | FP16 weights |", "|---|---:|---:|"]
            lines += [f"| `{name}` | {n:,} | {n * 2 / 2**30:.1f} GiB |" for name, n in rows]
            lines += ["", f"This build carries {params:,} parameters — the debug baseline, not a series size."]
    lines += ["", "## What this build is", ""]
    lines += [
        "This is the pipeline-validation build. It was trained and evaluated on a 4-core CPU host",
        "with no GPU, which caps it at tens of millions of parameters. It is **not** the 7B/8B/9B",
        "series models: those configs ship in `configs/` and require GPU hardware to train.",
        "",
        "## Limitations, stated plainly",
        "",
        "| limitation | detail |",
        "|---|---|",
        "| training tokens | about 8 M characters of pretraining text, roughly 100× less than a model this size needs |",
        "| instruction data | a few hundred pairs; style is learned, not knowledge |",
        "| answer quality | partially fluent; factual accuracy is not reliable |",
        "| modalities | text only |",
        "",
        "Read the samples below as a measurement of where the pipeline is, not as a product demo.",
        "",
        "## Sample answers (verbatim model output)",
        "",
    ]
    for question, answer in samples:
        lines += [f"**Q: {question}**", "", f"> {answer or '(empty)'}", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "params": params, "val_loss": val_loss,
                      "val_ppl": val_ppl, "samples": len(samples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
