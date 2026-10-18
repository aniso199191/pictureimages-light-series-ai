"""Interactive chat with a trained checkpoint (CPU, no server needed).

    python chat.py --ckpt models/sft.pt
"""
from __future__ import annotations

import argparse

import torch

from model import GPT, ModelConfig
from tokenizer import CharTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--rope-scale", type=float, default=0.0,
                        help="override RoPE scaling for longer inputs (0 keeps the checkpoint value)")
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**state["config"])
    if args.rope_scale:
        cfg.rope_scale = args.rope_scale
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    tok = CharTokenizer(state["vocab"])

    print(f"PictureImages Larme — {model.num_params():,} parameters. Ctrl-C to exit.")
    history: list[int] = []
    try:
        while True:
            question = input("你: ").strip()
            if not question:
                continue
            ids = [tok.bos_id] + tok.encode("<user>") + tok.encode(question) + tok.encode("<assistant>")
            idx = torch.tensor([ids])
            out = model.generate(idx, max_new_tokens=args.max_new, temperature=args.temperature,
                                 top_k=args.top_k, eos_id=tok.eos_id)
            answer = tok.decode(out[0].tolist()[len(ids):]).split("<eos>")[0]
            print("Larme:", answer.strip())
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
