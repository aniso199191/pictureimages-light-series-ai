# PictureImages Light Series AI

Open-source model series from the PictureImages project. Everything in this
repository is meant to run on **your** machine: we publish code, weights and the
data recipe, and we do not host inference for these models.

## Series at a glance

| Item | Value |
|---|---|
| Repository | `PictureImages Light Series AI` |
| First release | **PictureImages Larme Debug** (debug / test build) |
| Series target size | 7–9 B parameters (`configs/series_7b.json`, `series_8b.json`, `series_9b.json`) |
| Debug build size (measured) | see [`docs/REPORT.md`](docs/REPORT.md) — reported from the actual checkpoint |
| Target context | up to 1M–1.5M tokens on a machine with enough memory |
| Debug build context | 512 tokens trained; RoPE scaling supported for longer inputs |
| Framework | PyTorch reference implementation |
| Release format | FP16 weights (`tools/export_fp16.py`); "fp12" is not a standard format |

### Groups

| Group | What it is |
|---|---|
| `distill` | Training data produced by a larger teacher model through an OpenAI-compatible API, filtered and de-duplicated before use. |
| `daily` | Everyday conversation data used for the assistant-style fine-tune. |

## What the debug build is, and is not

`PictureImages Larme Debug` is the first runnable checkpoint of the pipeline:
tokenizer, pretraining, instruction tuning, evaluation and packaging, all
executed on a single CPU host. It exists so the series has a reproducible
baseline and a working toolchain.

It is **not** a 7–9 B model, and it does not generate images or video. Those
remain targets of the series, not claims about this build. Every number in
`docs/REPORT.md` is measured on the shipped checkpoint.

| Capability | This build |
|---|---|
| Chinese text chat | yes — small checkpoint, limited world knowledge |
| Code generation | partial — training data includes code text |
| Next-token prediction | yes — `predict/`, measured in `tests/test_predictor.py` |
| Instruction tuning | yes — `train/sft.py` |
| Image generation / recognition / video | no — not implemented |
| 7B/8B/9B weights | configs only; no GPU on the build host |

## Repository layout

```
train/       tokenizer, model, pretraining, SFT, evaluation, chat, QLoRA
distill/     teacher-data generator (API key from environment, never committed)
predict/     next-token predictor (ngram + vocabulary completion, build step)
configs/     7B / 8B / 9B architecture configs with verified parameter counts
tools/       parameter counter, FP16 export, scoring, report builder, releases
tests/       smoke test and predictor tests
docs/        quick start, hardware table, report card, auto-collected bug list
```

## Quick start

```bash
./setup.sh && source .venv/bin/activate
python tests/smoke.py                 # proves the toolchain works, ~1 minute
python predict/build_ngram.py --root . --extra-vocab predict/tech_terms.txt
python predict/predictor.py "Linu"    # -> x
```

The full walkthrough — training from scratch, instruction tuning, chatting,
packaging a release — is in [`docs/QUICKSTART.md`](docs/QUICKSTART.md).

## What the predictor does

Ask it for the next character and it answers from counted statistics rather than
a hardcoded rule:

```
$ python predict/predictor.py "Toke" --top 3
prefix: 'Toke'
  ngram  -> 'n'    p=1.000
$ python predict/predictor.py "Linu" --top 3
prefix: 'Linu'
  ngram  -> 'x'    p=1.000
```

`tests/test_predictor.py` checks those two completions plus twelve held-out
identifiers and fails if top-1 accuracy drops below 0.8.

## Reports, hardware and bug reports

* [`docs/REPORT.md`](docs/REPORT.md) — measured parameter count, loss, perplexity, sample answers
* [`docs/HARDWARE.md`](docs/HARDWARE.md) — what 7B/8B/9B need to train
* [`docs/BUGS.md`](docs/BUGS.md) — issues collected automatically from this repository
* [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — clone-to-inference walkthrough

## Licence

Code MIT. Weights released for research and personal use. Pretraining text keeps
its upstream licence (Wikipedia CC BY-SA 3.0/GFDL; Python documentation PSF),
listed with per-record provenance in the corpus manifest.
