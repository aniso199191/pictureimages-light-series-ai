"""QLoRA fine-tuning of a 7–9B open base model (needs a GPU host, not this one).

This is the path to a real 7B/8B/9B artifact without owning the hardware:
borrow a GPU (Kaggle, Modal, RunPod), point this script at the base model and
the instruction data, and it saves a LoRA adapter that can be merged and
published as FP16 weights.

    python train/qlora.py --base Qwen/Qwen2.5-7B-Instruct \
        --data data/sft_pairs.jsonl --out out/larme-7b-lora --epochs 2

Requires: torch, transformers, peft, bitsandbytes, datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_pairs(path: Path, max_chars: int) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        question = record.get("instruction") or record.get("question") or ""
        answer = record.get("response") or record.get("answer") or ""
        if not question or not answer:
            continue
        if len(question) + len(answer) > max_chars:
            continue
        rows.append({"messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="HF id of the base model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--max-seq", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_pairs(Path(args.data), args.max_chars)
    if not rows:
        raise SystemExit("no usable training rows")
    print(json.dumps({"rows": len(rows), "base": args.base, "out": args.out}))
    if args.dry_run:
        print(json.dumps({"first": rows[0]}, ensure_ascii=False)[:400])
        return 0

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    def tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        enc = tokenizer(text, truncation=True, max_length=args.max_seq, padding=False)
        enc["labels"] = list(enc["input_ids"])
        return enc

    dataset = Dataset.from_list(rows).map(tokenize, remove_columns=["messages"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to=[],
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(json.dumps({"saved": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
