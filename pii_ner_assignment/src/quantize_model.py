"""
Quantize a trained model checkpoint and rerun all evaluation metrics.

This script:
1. Loads a trained model from checkpoint directory
2. Applies INT8 quantization for faster CPU inference
3. Runs predictions on dev set
4. Computes evaluation metrics (overall, per-entity, PII-specific)
5. Measures latency (p50, p95)
6. Saves all results in a new quantized output directory
"""

import json
import argparse
import logging
import subprocess
import shutil
from pathlib import Path
import time
import statistics

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from labels import ID2LABEL, label_is_pii, LABELS
from dataset import PIIDataset


def quantize_model(model, device="cpu"):
    """
    Apply dynamic quantization to the model for faster CPU inference.
    
    Args:
        model: PyTorch model to quantize
        device: Target device (quantization works best on CPU)
    
    Returns:
        Quantized model (always on CPU)
    """
    logging.info("Applying INT8 dynamic quantization...")
    
    # Move model to CPU for quantization (required)
    if device != "cpu":
        logging.warning("Quantization requires CPU. Moving model to CPU...")
        model = model.cpu()
    else:
        model = model.cpu()  # Ensure on CPU
    
    # Apply dynamic quantization
    # This quantizes weights to INT8 but keeps activations in FP32
    # Note: quantize_dynamic modifies model in-place but also returns it
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )
    
    logging.info("Model quantized successfully (stays on CPU)")
    return quantized_model


def measure_latency(model, tokenizer, dev_ds, device, max_length, runs=30):
    """Measure p50 and p95 latency for the quantized model."""
    model.eval()
    
    texts = []
    for item in dev_ds:
        texts.append(item["text"])
    
    if not texts:
        logging.warning("No texts found for latency measurement")
        return {"p50_ms": 0, "p95_ms": 0, "runs": 0}
    
    times_ms = []
    
    # Warmup
    logging.info("Warming up model...")
    for _ in range(5):
        t = texts[0]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            _ = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device)
            )
    
    # Measure latency
    logging.info(f"Measuring latency over {runs} runs...")
    for i in range(runs):
        t = texts[i % len(texts)]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device)
            )
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
    
    p50 = statistics.median(times_ms)
    times_sorted = sorted(times_ms)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1]
    
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "runs": runs
    }


def generate_predictions(model, tokenizer, dev_path, output_path, device, max_length):
    """Generate predictions using the quantized model."""
    logging.info(f"Generating predictions for {dev_path}...")
    
    model.eval()
    results = {}
    count = 0
    
    def bio_to_spans(text, offsets, label_ids):
        spans = []
        current_label = None
        current_start = None
        current_end = None

        for (start, end), lid in zip(offsets, label_ids):
            if start == 0 and end == 0:
                continue
            label = ID2LABEL.get(int(lid), "O")
            if label == "O":
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                    current_label = None
                continue

            prefix, ent_type = label.split("-", 1)
            if prefix == "B":
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end
            elif prefix == "I":
                if current_label == ent_type:
                    current_end = end
                else:
                    if current_label is not None:
                        spans.append((current_start, current_end, current_label))
                    current_label = ent_type
                    current_start = start
                    current_end = end

        if current_label is not None:
            spans.append((current_start, current_end, current_label))
        return spans
    
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)
            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents
            count += 1
            if count % 100 == 0:
                logging.info("Processed %d utterances", count)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Predictions saved to {output_path}")
    return output_path


def run_evaluation(gold_path, pred_path, metrics_out_path, project_root):
    """Run evaluation script to compute metrics."""
    logging.info("Computing evaluation metrics...")
    
    eval_cmd = [
        "python", "src/eval_span_f1.py",
        "--gold", str(Path(gold_path).absolute()),
        "--pred", str(Path(pred_path).absolute()),
        "--metrics_out", str(Path(metrics_out_path).absolute())
    ]
    
    result = subprocess.run(
        eval_cmd,
        cwd=str(project_root),
        check=False,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {result.stderr}")
    
    if not metrics_out_path.exists():
        raise FileNotFoundError(f"Metrics file not created: {metrics_out_path}")
    
    with open(metrics_out_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    logging.info("Evaluation metrics computed")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a trained model and rerun all evaluation metrics"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to model checkpoint directory (e.g., out_baseline/best_model)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory containing dev.jsonl (e.g., data_baseline)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for quantized results (default: {checkpoint_dir}_quantized)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (quantization requires CPU, will override if set to cuda)"
    )
    parser.add_argument(
        "--latency_runs",
        type=int,
        default=30,
        help="Number of runs for latency measurement"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # Setup paths
    checkpoint_dir = Path(args.checkpoint_dir)
    dataset_dir = Path(args.dataset_dir)
    dev_path = dataset_dir / "dev.jsonl"
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not dev_path.exists():
        raise FileNotFoundError(f"Dev file not found: {dev_path}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_quantized"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 80)
    logging.info("MODEL QUANTIZATION & EVALUATION")
    logging.info("=" * 80)
    logging.info(f"Checkpoint: {checkpoint_dir}")
    logging.info(f"Dataset: {dataset_dir}")
    logging.info(f"Output: {output_dir}")
    logging.info(f"Device: {args.device}")
    logging.info("")
    
    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    model.eval()
    
    # Get model size before quantization
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    logging.info(f"Original model size: {model_size_mb:.2f} MB")
    
    # Quantize model (always uses CPU)
    quantized_model = quantize_model(model, device="cpu")
    quantized_device = "cpu"  # Quantized models must run on CPU
    
    # Get quantized model size (approximate)
    quantized_size_mb = model_size_mb * 0.25  # INT8 is ~4x smaller than FP32
    logging.info(f"Quantized model size (approx): {quantized_size_mb:.2f} MB")
    
    # Load dev dataset
    logging.info("Loading dev dataset...")
    dev_ds = PIIDataset(str(dev_path), tokenizer, LABELS, max_length=args.max_length, is_train=True)
    
    # Generate predictions (quantized model must use CPU)
    pred_path = output_dir / "dev_pred.json"
    generate_predictions(
        quantized_model,
        tokenizer,
        str(dev_path),
        pred_path,
        quantized_device,
        args.max_length
    )
    
    # Run evaluation
    project_root = Path(__file__).parent.parent
    metrics_path = output_dir / "detailed_metrics.json"
    detailed_metrics = run_evaluation(
        str(dev_path),
        str(pred_path),
        metrics_path,
        project_root
    )
    
    # Measure latency (quantized model must use CPU)
    logging.info("Measuring latency...")
    latency_metrics = measure_latency(
        quantized_model,
        tokenizer,
        dev_ds,
        quantized_device,
        args.max_length,
        runs=args.latency_runs
    )
    
    # Extract metrics
    pii_metrics = detailed_metrics.get("pii", {})
    non_pii_metrics = detailed_metrics.get("non_pii", {})
    per_entity = detailed_metrics.get("per_entity", {})
    
    overall_precision = detailed_metrics.get("overall_precision", 0)
    overall_recall = detailed_metrics.get("overall_recall", 0)
    overall_f1 = detailed_metrics.get("overall_f1", 0)
    
    # Save comprehensive metrics
    final_metrics = {
        "overall": {
            "pii_precision_micro": pii_metrics.get("precision", 0),
            "pii_recall_micro": pii_metrics.get("recall", 0),
            "pii_f1_micro": pii_metrics.get("f1", 0),
            "overall_precision_micro": overall_precision,
            "overall_recall_micro": overall_recall,
            "overall_f1_micro": overall_f1,
        },
        "per_entity": per_entity,
        "latency": {
            **latency_metrics,
            "model_size_mb": model_size_mb,
            "model_size_quantized_mb": quantized_size_mb,
        }
    }
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    
    # Save tokenizer config (for reference)
    if (checkpoint_dir / "tokenizer_config.json").exists():
        shutil.copy(
            checkpoint_dir / "tokenizer_config.json",
            output_dir / "tokenizer_config.json"
        )
    if (checkpoint_dir / "vocab.txt").exists():
        shutil.copy(
            checkpoint_dir / "vocab.txt",
            output_dir / "vocab.txt"
        )
    
    # Print summary
    logging.info("")
    logging.info("=" * 80)
    logging.info("QUANTIZATION RESULTS")
    logging.info("=" * 80)
    logging.info("OVERALL METRICS:")
    logging.info(f"  Precision: {overall_precision:.3f} | Recall: {overall_recall:.3f} | F1: {overall_f1:.3f}")
    logging.info("")
    logging.info("PII METRICS:")
    logging.info(f"  Precision: {final_metrics['overall']['pii_precision_micro']:.3f} | "
                f"Recall: {final_metrics['overall']['pii_recall_micro']:.3f} | "
                f"F1: {final_metrics['overall']['pii_f1_micro']:.3f}")
    logging.info("")
    logging.info("LATENCY METRICS:")
    logging.info(f"  p50: {latency_metrics['p50_ms']:.2f} ms | p95: {latency_metrics['p95_ms']:.2f} ms")
    logging.info(f"  Model size: {model_size_mb:.2f} MB → {quantized_size_mb:.2f} MB (4x reduction)")
    logging.info("")
    logging.info(f"All results saved to: {output_dir}")
    logging.info(f"  - Predictions: {pred_path}")
    logging.info(f"  - Metrics: {metrics_file}")
    logging.info(f"  - Detailed metrics: {metrics_path}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

