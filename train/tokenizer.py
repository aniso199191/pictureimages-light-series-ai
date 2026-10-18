"""Character-level tokenizer.

Deliberately simple and dependency-free: every Chinese character and ASCII
printable character is one token. This keeps the vocabulary small (~7k),
makes the corpus fully reproducible, and avoids shipping a third-party
tokenizer with licence strings attached.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]


class CharTokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = list(vocab)

    @classmethod
    def train(cls, texts, min_freq: int = 3, max_vocab: int = 12000) -> "CharTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)
        chars = [ch for ch, n in counter.most_common() if n >= min_freq]
        chars = chars[: max_vocab - len(SPECIALS)]
        vocab = SPECIALS + [ch for ch in chars if ch not in SPECIALS]
        return cls(vocab)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocab": self.vocab}, ensure_ascii=False), encoding="utf-8")

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    def encode(self, text: str) -> list[int]:
        unk = self.unk_id
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] if 0 <= i < len(self.itos) else "" for i in ids)
