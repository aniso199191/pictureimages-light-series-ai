"""Build a character n-gram next-token predictor from source code.

The predictor answers "given these few characters, what comes next?" — the same
question a large model answers with its language head, but here the statistics
are counted directly, so it is fast, inspectable and needs no GPU.

Only plain source files are read. Lines that look like they carry a credential
are skipped and never enter the model.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SKIP = re.compile(r"(api[_-]?key|secret|password|passwd|token\s*=|BEGIN [A-Z ]*PRIVATE KEY)", re.I)
TEXT_EXT = {".py", ".js", ".ts", ".c", ".h", ".cs", ".md", ".txt", ".json", ".sh"}
MAX_LINE = 400


def iter_text(roots: list[Path], limit_lines: int):
    seen = 0
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_EXT:
                continue
            if any(part in {".git", "node_modules", ".venv", "__pycache__", "build"} for part in path.parts):
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                if len(line) > MAX_LINE or SKIP.search(line):
                    continue
                if line.strip():
                    yield line
                    seen += 1
                    if limit_lines and seen >= limit_lines:
                        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--out", default="predict/ngram.json.gz")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--limit-lines", type=int, default=400_000)
    parser.add_argument("--extra-vocab", action="append", default=[])
    args = parser.parse_args()

    order = args.order
    counts: dict[str, Counter] = defaultdict(Counter)
    for line in iter_text([Path(r) for r in args.root], args.limit_lines):
        padded = "\x02" * (order - 1) + line
        for i in range(order - 1, len(padded)):
            context = padded[i - order + 1:i]
            counts[context][padded[i]] += 1

    table = {ctx: dict(counter.most_common(12)) for ctx, counter in counts.items()}
    bigram: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ctx, counter in counts.items():
        bigram[ctx[-1]].update(counter)
    bigrams = {ch: dict(sorted(row.items(), key=lambda kv: -kv[1])[:12]) for ch, row in bigram.items()}

    words: Counter[str] = Counter()
    for line in iter_text([Path(r) for r in args.root], args.limit_lines):
        words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", line))
    vocab = dict(words.most_common(120_000))
    if args.extra_vocab:
        for pattern in args.extra_vocab:
            extra = Path(pattern)
            if not extra.is_file():
                continue
            for word in extra.read_text(encoding="utf-8").split():
                word = word.strip()
                if 2 <= len(word) <= 40:
                    vocab[word] = vocab.get(word, 0) + 200

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        json.dump({"order": order, "table": table, "bigram": bigrams, "vocab": vocab},
                  handle, ensure_ascii=False)
    print(json.dumps({"contexts": len(table), "vocab": len(vocab), "bytes": out.stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
