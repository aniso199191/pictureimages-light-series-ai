# 7B route — QLoRA on a free GPU

This host cannot train 7B: one process is capped near 3.8 GB RAM, there is no
GPU, and the disk holds 24 GB free. A 7B model is 13 GiB in FP16 and needs
110–180 GiB to train. So the 7B route runs on a borrowed GPU, and what we
publish is the resulting adapter (and optionally merged weights).

Scope: fine-tuning an existing open base model. Training 7B from scratch takes
about 13,000 A100-hours — not something a free tier can do.

## What you need

- A Kaggle account (free) with phone verification, which unlocks a T4 ×2 GPU
  for 30 hours a week.
- The files in this folder: `qlora_7b.py` and `pairs.jsonl`.

## Steps

1. Create a Kaggle notebook, upload `pairs.jsonl` as a dataset, and attach it.
2. In the notebook, install and run:

```bash
pip -q install "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" datasets accelerate
python qlora_7b.py --data /kaggle/input/pairs/pairs.jsonl --out /kaggle/working/adapter --epochs 2
```

3. The adapter lands in `/kaggle/working/adapter` (roughly 150–250 MB). Download
   it from the notebook output panel.

To also produce merged 7B weights (about 14 GB, needs a lot of RAM during the
merge step):

```bash
python qlora_7b.py --data /kaggle/input/pairs/pairs.jsonl --out /kaggle/working/adapter --epochs 2 --merge
```

## Publishing

- Adapter (small): attach to a GitHub Release directly.
- Merged weights: GitHub rejects files over 2 GB. Split with
  `split -b 1900M model.safetensors part_` and upload the parts, or push to
  GitOpen, which accepts multi-part uploads.

## Cost and time

| item | value |
|---|---|
| GPU | 1 × T4 16 GB (Kaggle free tier) |
| time for 2 epochs on 417 pairs | about 40–70 minutes |
| adapter size | 150–250 MB |
| merged FP16 weights | about 14 GB |
