"""QLoRA fine-tune of a 7B base model on the Larme instruction set.

Runs on a single 16 GB GPU (Kaggle T4 / Colab) in 4-bit NF4. Produces a LoRA
adapter, and optionally a merged FP16 checkpoint.

    python qlora_7b.py --data pairs.jsonl --out /kaggle/working/adapter --epochs 2

Notes:
  * This fine-tunes an existing open base model. It does not train 7B from
    scratch — that needs roughly 13,000 A100-hours.
  * The merged checkpoint is ~14 GB FP16; GitHub rejects files over 2 GB, so
    publish the adapter, or split the merged weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_pairs(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        prompt = obj.get("prompt") or obj.get("instruction") or ""
        answer = obj.get("response") or obj.get("output") or ""
        if prompt and answer:
            rows.append({"text": f"<user>{prompt}<assistant>{answer}"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="adapter")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    rows = load_pairs(Path(args.data))
    if not rows:
        raise SystemExit("no usable pairs in --data")
    print(f"pairs: {len(rows)}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = Dataset.from_list(rows)

    def tokenize(batch):
        out = tok(batch["text"], truncation=True, max_length=args.max_len, padding=False)
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="epoch",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            optim="paged_adamw_8bit",
            report_to=[],
        ),
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"adapter saved to {args.out}")

    if args.merge:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.float16, device_map="cpu",
            trust_remote_code=True)
        merged = PeftModel.from_pretrained(base, args.out).merge_and_unload()
        merged.save_pretrained(f"{args.out}-merged", safe_serialization=True)
        tok.save_pretrained(f"{args.out}-merged")
        print(f"merged checkpoint saved to {args.out}-merged (~14 GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
