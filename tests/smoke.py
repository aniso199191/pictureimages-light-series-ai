"""End-to-end smoke test: prepare → train → generate, on a tiny synthetic corpus.

    python tests/smoke.py

Takes about a minute on a CPU and proves the installation works before anyone
downloads a large corpus or a checkpoint.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
PY = sys.executable

SENTENCES = [
    "今天天气很好，我们去公园散步。",
    "机器学习的核心是用数据拟合函数。",
    "把水烧开需要一百摄氏度。",
    "写代码之前先想清楚输入和输出。",
    "晚饭吃面条，加一个荷包蛋。",
] * 40


def run(args: list[str], cwd: Path) -> None:
    env = dict(os.environ, THREADS="4", PYTHONPATH=str(cwd))
    result = subprocess.run([PY] + args, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-1500:])
        print(result.stderr[-1500:])
        raise SystemExit(f"failed: {' '.join(args)}")
    tail = (result.stdout.strip().splitlines() or [""])[-1]
    print(f"  ok  {' '.join(args[:2])}  {tail[:110]}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        corpus = work / "corpus.jsonl"
        corpus.write_text(
            "\n".join(json.dumps({"text": line}, ensure_ascii=False) for line in SENTENCES) + "\n",
            encoding="utf-8")

        run([str(TRAIN / "prepare.py"), "--train", str(corpus), "--val", str(corpus),
             "--out", str(work / "data"), "--max-vocab", "600", "--min-freq", "1"], TRAIN)
        run([str(TRAIN / "train.py"), "--data", str(work / "data"), "--steps", "6",
             "--batch", "2", "--block", "64", "--layer", "2", "--head", "2", "--embd", "64",
             "--warmup", "2", "--eval-every", "6", "--out", str(work / "tiny.pt"),
             "--log", str(work / "tiny.jsonl")], TRAIN)
        run([str(TRAIN / "eval.py"), "--ckpt", str(work / "tiny.pt"),
             "--data", str(work / "data"), "--prompt", "今天天气", "--max-new", "16"], TRAIN)

    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
