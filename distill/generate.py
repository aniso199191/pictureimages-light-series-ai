#!/usr/bin/env python3
"""Generate Chinese instruction-tuning data from a teacher model.

This produces *training data*, not a model. Every record keeps the teacher
model name and prompt version so provenance is auditable.

Usage:
    python generate.py --n 12 --model gpt-6-astra --dry-run
    python generate.py --n 12 --model gpt-6-astra
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"

TOPICS = [
    "光合作用", "板块运动与地震", "疫苗的工作原理", "抗生素与病毒的区别",
    "复利", "通货膨胀", "机会成本", "保险的基本原理",
    "TCP 与 UDP", "HTTPS 如何加密", "数据库索引", "缓存的作用与代价",
    "递归", "面向对象", "版本控制", "单元测试",
    "著作权与合理使用", "合同的基本要素", "劳动仲裁", "消费者权益",
    "睡眠与生物钟", "有氧运动", "蛋白质摄入", "心理压力管理",
    "中国朝代顺序", "工业革命", "丝绸之路", "文艺复兴",
    "季风气候", "洋流", "时区", "主谓一致",
    "英语时态", "写作结构", "概率基础", "相关与因果",
    "一次函数", "排列组合", "逻辑谬误", "批判性思维",
]

# (label, directive given to the teacher)
STYLES = [
    ("explain", "让用户请教学科原理，答案要讲清关键机制"),
    ("compare", "让用户询问两个相近概念的区别，答案要对比并说明适用场景"),
    ("howto", "让用户询问具体做法，答案要给出可执行步骤"),
    ("why", "让用户追问背后的原因，答案要给出因果链"),
    ("example", "让用户要求举例，答案要用贴近日常生活的例子说明"),
    ("myth", "让用户询问常见误解，答案要纠正误解并说明理由"),
]

SYSTEM_PROMPT = """你是用于构建开源中文教学数据集的教师模型。你会收到一个主题和出题方向，请产出一个高质量的中文问答对。

输出严格的 JSON，不要加代码围栏，不要有其他文字：
{"question": "用户会问的问题", "answer": "回答"}

要求：
1. question 要像真实用户随口提问，自然、具体，6-60 字，不要用“请解释……的基本原理”这种模板腔。
2. answer 150-400 字，直接回答，信息密度高。
3. 事实必须准确。涉及数字、年份、剂量、法律条款时，如果不确定就明确写出不确定性，不要编造。
4. 不要以“作为一个AI”“以下是”“希望对你有帮助”开头或结尾，不要重复问题。
5. 需要分点时用简短编号，不要使用 Markdown 标题（#）。
6. 使用简体中文，语气自然，像一位耐心的老师。"""

PROMPT_VERSION = "teacher-zh-v2"

REFUSAL_MARKERS = [
    "作为一个AI", "作为AI", "我无法", "我不能提供", "抱歉，我不能",
    "我无法回答", "作为一个语言模型", "作为一个人工智能",
]

BOILERPLATE_PREFIXES = ["以下是", "好的，", "当然可以", "希望对你", "总结如下"]


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", text).lower()


def parse_json_reply(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def clean_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def quality_ok(question: str, answer: str) -> tuple[bool, str]:
    if not question or not answer:
        return False, "empty"
    if not (6 <= len(question) <= 120):
        return False, "bad_question_length"
    if len(answer) < 80:
        return False, "too_short"
    if len(answer) > 2400:
        return False, "too_long"
    for marker in REFUSAL_MARKERS:
        if marker in answer:
            return False, "refusal"
    head = answer[:12]
    for prefix in BOILERPLATE_PREFIXES:
        if head.startswith(prefix):
            return False, "boilerplate"
    if norm(answer) == norm(question):
        return False, "echo"
    han = sum(1 for ch in answer if "\u4e00" <= ch <= "\u9fff")
    if han < 40:
        return False, "no_chinese"
    return True, "ok"


def load_env() -> None:
    for env_file in (ROOT / ".env", ROOT.parent / ".env"):
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def call_teacher(model: str, prompt: str, retries: int = 3) -> tuple[str, dict]:
    import urllib.error
    import urllib.request

    key = os.environ["HIVE_DISTILL_API_KEY"]
    base = os.environ.get("HIVE_DISTILL_BASE", "https://fengchao-api.com").rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1400,
        "temperature": 0.9,
    }
    last_error = "unknown"
    for attempt in range(retries):
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.load(resp)
            msg = (body.get("choices") or [{}])[0].get("message", {})
            return msg.get("content") or "", body.get("usage") or {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            last_error = f"HTTP {exc.code}: {detail[:200]}"
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(3 * (attempt + 1))
                continue
            break
        except Exception as exc:  # timeout, connection reset
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(last_error)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12, help="number of new samples to attempt")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="print seeds only")
    args = parser.parse_args()

    load_env()
    DATA.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    train_path = DATA / "train.jsonl"
    seen_path = DATA / "seen_seeds.txt"
    usage_path = LOGS / "usage.jsonl"

    seen: set[str] = set()
    if seen_path.is_file():
        seen = {line.strip() for line in seen_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    seeds: list[tuple[str, str, str]] = []
    for topic in TOPICS:
        for style, directive in STYLES:
            key = f"{topic}|{style}"
            if key in seen:
                continue
            seeds.append((key, topic, directive))

    seeds = seeds[: args.n]
    if args.dry_run:
        for key, topic, directive in seeds:
            print(key, "|", directive)
        print(f"seeds available: {len(seeds)}")
        return 0

    if not seeds:
        print("no new seeds left; expand TOPICS/STYLES")
        return 0

    print(f"generating {len(seeds)} samples with {args.model}", flush=True)
    kept = 0
    rejected: dict[str, int] = {}

    def work(seed: tuple[str, str, str]):
        key, topic, directive = seed
        prompt = f"主题：{topic}\n出题方向：{directive}\n请输出 JSON。"
        started = time.monotonic()
        raw, usage = call_teacher(args.model, prompt)
        return key, topic, raw, usage, round(time.monotonic() - started, 2)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, seed): seed for seed in seeds}
        for future in as_completed(futures):
            key, topic, _ = futures[future]
            try:
                key, topic, raw, usage, seconds = future.result()
                pair = parse_json_reply(raw)
                question = str(pair.get("question", "")).strip()
                answer = clean_answer(str(pair.get("answer", "")))
            except Exception as exc:
                print(f"ERROR {key}: {exc}", flush=True)
                rejected["error"] = rejected.get("error", 0) + 1
                continue

            ok, reason = quality_ok(question, answer)
            if not ok:
                rejected[reason] = rejected.get(reason, 0) + 1
                print(f"REJECT {key} ({reason})", flush=True)
                continue

            record = {
                "id": hashlib.sha256(norm(question).encode()).hexdigest()[:16],
                "instruction": question,
                "response": answer,
                "topic": topic,
                "style": key.split("|", 1)[1],
                "teacher": args.model,
                "prompt_version": PROMPT_VERSION,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seconds": seconds,
            }
            with train_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            with seen_path.open("a", encoding="utf-8") as handle:
                handle.write(key + "\n")
            with usage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "id": record["id"],
                    "seed": key,
                    "model": args.model,
                    "usage": usage,
                    "seconds": seconds,
                    "created": record["created"],
                }, ensure_ascii=False) + "\n")
            kept += 1
            print(f"OK {key} q={len(question)}ch a={len(answer)}ch {seconds}s", flush=True)

    print(json.dumps({"attempted": len(seeds), "kept": kept, "rejected": rejected,
                      "train_file": str(train_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
