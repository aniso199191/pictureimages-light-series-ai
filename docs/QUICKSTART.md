# Quick start

Everything here runs on a laptop CPU for the small configuration. A GPU is only
needed for the 7B/8B/9B configs in `configs/`.

## 1. Install

```bash
git clone <this-repository>
cd pictureimages-light-series-ai
./setup.sh                 # creates .venv and installs torch + numpy
source .venv/bin/activate
```

## 2. Prove the install works (about a minute)

```bash
python tests/smoke.py
```

It builds a tiny corpus, prepares it, trains 6 steps, and generates text. If it
prints `smoke test passed`, the whole toolchain is healthy.

## 3. Train a small model from scratch

The corpus format is one JSON object per line:

```json
{"text": "要训练的正文段落。"}
```

```bash
python train/prepare.py \
  --train data/pretrain_train.jsonl \
  --val   data/pretrain_val.jsonl \
  --out   train/data --max-vocab 9000

python train/train.py \
  --steps 2600 --batch 6 --block 512 \
  --layer 10 --head 8 --embd 512 \
  --lr 1e-4 --warmup 200 --eval-every 200 \
  --out train/models/base.pt --log train/logs/base.jsonl
```

Rough throughput on 4 CPU cores: about 650–700 tokens/second, so 2600 steps of
512-token blocks is close to four hours.

## 4. Fine-tune on instructions

`--pairs` accepts one or more JSONL files with `instruction`/`response` (or
`question`/`answer`) fields:

```bash
python train/sft.py --init train/models/base.pt \
  --pairs train/data/sft_pairs.jsonl \
  --epochs 3 --block 512 --out train/models/sft.pt
```

## 5. Talk to it

```bash
python train/chat.py --ckpt train/models/sft.pt
```

or score it:

```bash
python tools/score_sft.py --ckpt train/models/sft.pt --dev data/distill_dev.jsonl
python tools/make_report.py --ckpt train/models/sft.pt --data train/data --out docs/REPORT.md
```

## 6. Next-token predictor

The predictor ships as source plus a build step, so no large binary is stored in
the repository:

```bash
python predict/build_ngram.py --root . --extra-vocab predict/tech_terms.txt
python predict/predictor.py "Linu" --top 3
python predict/predictor.py "Toke" --top 3
python tests/test_predictor.py
```

## 7. 7B/8B/9B on borrowed hardware

The three configs are real architectures with verified parameter counts:

```bash
python tools/count_params.py configs/series_9b.json     # 9,114,755,072
```

On a GPU host, fine-tune an open base model with QLoRA:

```bash
pip install transformers peft bitsandbytes datasets
python train/qlora.py --base Qwen/Qwen2.5-7B-Instruct \
  --data train/data/sft_pairs.jsonl --out out/larme-7b-lora --epochs 2
```

`docs/HARDWARE.md` lists the memory each size needs.

## 8. Package a release

```bash
python tools/export_fp16.py --ckpt train/models/sft.pt --out dist/larme-debug-fp16.pt
python tools/publish_release.py --file dist/larme-debug-fp16.pt \
  --name "Larme Debug FP16 weights" --body "FP16 weights, see docs/REPORT.md"
```

The export prints the SHA-256 so downloaders can verify the file.
