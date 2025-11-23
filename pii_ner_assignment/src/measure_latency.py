import json
import time
import argparse
import statistics
import logging

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--metrics_out", default=None, help="Optional JSON file to store latency stats.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    if not texts:
        print("No texts found in input file.")
        return

    times_ms = []

    # warmup
    for _ in range(5):
        t = texts[0]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(args.device), attention_mask=enc["attention_mask"].to(args.device))

    for i in range(args.runs):
        t = texts[i % len(texts)]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(args.device), attention_mask=enc["attention_mask"].to(args.device))
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    p50 = statistics.median(times_ms)
    times_sorted = sorted(times_ms)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1]

    print(f"Latency over {args.runs} runs (batch_size=1):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")
    logging.info("Latency p50=%.2f ms p95=%.2f ms over %d runs", p50, p95, args.runs)

    if args.metrics_out:
        payload = {"runs": args.runs, "p50_ms": p50, "p95_ms": p95}
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logging.info("Wrote latency metrics to %s", args.metrics_out)


if __name__ == "__main__":
    main()
