"""Build the character vocabulary and uint16 token streams.

Reads JSONL corpora ({"text": ...}); writes vocab.json, train.bin, val.bin.
Writes in chunks so a 4-core box with a ~3.8 GB per-process limit survives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tokenizer import CharTokenizer

CHUNK_TOKENS = 4_000_000


def iter_texts(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text") or record.get("response") or ""
            if text:
                yield text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--out", default="data")
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--max-vocab", type=int, default=12000)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_files = [Path(p) for p in args.train]
    val_files = [Path(p) for p in args.val]

    # Pass 1: vocabulary from training text only.
    print("building vocabulary", flush=True)
    tok = CharTokenizer.train(
        (t for path in train_files for t in iter_texts(path)),
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
    )
    tok.save(out / "vocab.json")
    print(f"vocab_size={len(tok)}", flush=True)

    def pack(files: list[Path], name: str) -> int:
        target = out / name
        total = 0
        buffer: list[int] = []
        with target.open("wb") as handle:
            for path in files:
                for text in iter_texts(path):
                    ids = tok.encode(text)
                    buffer.extend(ids)
                    buffer.append(tok.eos_id)
                    if len(buffer) >= CHUNK_TOKENS:
                        arr = np.asarray(buffer, dtype=np.uint16)
                        handle.write(arr.tobytes())
                        total += len(buffer)
                        buffer.clear()
            if buffer:
                arr = np.asarray(buffer + [tok.eos_id], dtype=np.uint16)
                handle.write(arr.tobytes())
                total += len(arr)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        print(f"{name}: tokens={total} bytes={target.stat().st_size} sha256={digest[:16]}", flush=True)
        return total

    train_tokens = pack(train_files, "train.bin")
    val_tokens = pack(val_files, "val.bin")

    manifest = {
        "vocab_size": len(tok),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
