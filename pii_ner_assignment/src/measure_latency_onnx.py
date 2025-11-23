#!/usr/bin/env python3
"""
Measure inference latency for an ONNX model.

Usage:
    python src/measure_latency_onnx.py --model_dir onnx_models/method2 --input data/dev.jsonl --runs 50
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
from transformers import AutoTokenizer

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def measure_latency_onnx(
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int = 128,
    runs: int = 50,
    warmup: int = 5
) -> dict:
    """
    Measure p50 and p95 latency for ONNX model inference.
    
    Args:
        session: ONNX Runtime inference session
        tokenizer: Hugging Face tokenizer
        texts: List of input texts to run inference on
        max_length: Maximum sequence length
        runs: Number of inference runs
        warmup: Number of warmup runs before measurement
    
    Returns:
        Dictionary with p50_ms, p95_ms, and runs
    """
    logging.info("Warming up model with %d runs...", warmup)
    for i in range(warmup):
        text = texts[i % len(texts)]
        encoded = tokenizer(
            text,
            return_tensors="np",
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        ort_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        _ = session.run(None, ort_inputs)
    
    logging.info("Measuring latency over %d runs...", runs)
    latencies = []
    
    for i in range(runs):
        text = texts[i % len(texts)]
        
        # Tokenize
        encoded = tokenizer(
            text,
            return_tensors="np",
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        
        ort_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        
        # Measure inference time
        start = time.perf_counter()
        _ = session.run(None, ort_inputs)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to milliseconds
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "runs": runs,
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure ONNX model latency")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the ONNX model")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file for latency measurement")
    parser.add_argument("--runs", type=int, default=50,
                        help="Number of inference runs (default: 50)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup runs (default: 5)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length (default: 128)")
    parser.add_argument("--metrics_out", type=str, default=None,
                        help="Output file for metrics JSON")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    onnx_path = model_dir / "model.onnx"
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    logging.info("=" * 80)
    logging.info("ONNX LATENCY MEASUREMENT")
    logging.info("=" * 80)
    logging.info("Model: %s", onnx_path)
    logging.info("Input: %s", args.input)
    logging.info("Runs: %d", args.runs)
    logging.info("Warmup: %d", args.warmup)
    logging.info("")
    
    # Load model
    logging.info("Loading ONNX model...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    # Load tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load texts from input file
    logging.info("Loading input texts...")
    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
    
    logging.info("Loaded %d texts", len(texts))
    
    # Measure latency
    metrics = measure_latency_onnx(
        session, tokenizer, texts, args.max_length, args.runs, args.warmup
    )
    
    # Calculate model size
    model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    metrics["model_size_mb"] = model_size_mb
    
    # Display results
    logging.info("")
    logging.info("=" * 80)
    logging.info("LATENCY RESULTS")
    logging.info("=" * 80)
    logging.info("p50: %.2f ms", metrics["p50_ms"])
    logging.info("p95: %.2f ms", metrics["p95_ms"])
    logging.info("Mean: %.2f ms", metrics["mean_ms"])
    logging.info("Std: %.2f ms", metrics["std_ms"])
    logging.info("Min: %.2f ms", metrics["min_ms"])
    logging.info("Max: %.2f ms", metrics["max_ms"])
    logging.info("Model size: %.2f MB", model_size_mb)
    logging.info("=" * 80)
    
    # Save metrics if requested
    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logging.info("Metrics saved to %s", args.metrics_out)


if __name__ == "__main__":
    main()

