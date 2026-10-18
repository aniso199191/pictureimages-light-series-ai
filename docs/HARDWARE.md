# Hardware required for the 7B / 8B / 9B builds

These files describe the three series sizes. The parameter counts below were
produced on the PyTorch meta device (`tools/count_params.py`), so they allocate
no memory and no disk — they describe the architecture, not trained weights.

| config | layers | width | heads | parameters | FP16 weights |
|---|---:|---:|---:|---:|---:|
| `series_7b.json` | 24 | 4096 | 32 | 6.97 B | 13.0 GiB |
| `series_8b.json` | 28 | 4096 | 32 | 8.04 B | 15.0 GiB |
| `series_9b.json` | 32 | 4096 | 32 | 9.12 B | 17.0 GiB |

## What each size needs to actually train

Training memory is not the weight size: it adds gradients, optimizer state,
activations and the logits tensor. A practical rule for AdamW in mixed
precision is roughly **16–20 bytes per parameter** plus activations, i.e.

| size | weights | training memory (rule of thumb) | sensible minimum |
|---|---:|---:|---|
| 7B | 13 GiB | ~110–140 GiB | 2× A100 80 GB or 4× 48 GB |
| 8B | 15 GiB | ~130–160 GiB | 2–4× A100/H100 80 GB |
| 9B | 17 GiB | ~145–180 GiB | 4× A100/H100 80 GB |

Data: a 7–9B model wants on the order of 140B–1T training tokens for a
competitive result. At ~6M tokens/second (a healthy multi-GPU cluster) 300B
tokens alone is about 14 GPU-hours per epoch-equivalent pass of 100B tokens.

## What this host can do

The host used for the debug build has 4 CPU cores, 7.4 GiB RAM with a ~3.8 GiB
per-process ceiling, no GPU, and 25 GiB free disk. That fits a **46.6 M**
parameter model for a few hours. It cannot hold a 7B weight file (13 GiB), let
alone optimizer state, so no honest 7B/8B/9B checkpoint can be produced here.

The debug release therefore ships:

* the pipeline that trains, fine-tunes, evaluates and packages a model,
* a real 46.6 M checkpoint trained on this host,
* the exact 7B/8B/9B configs above, reproducible with `tools/count_params.py`.

Given a machine with 4× 80 GB GPUs the same scripts train `series_9b.json`
without code changes.
