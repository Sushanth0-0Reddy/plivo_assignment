#!/usr/bin/env python3
"""
Complete ONNX inference pipeline: export, predict, evaluate, and measure latency.

Usage:
    python src/onnx_inference.py --checkpoint_dir out_method2/best_model --dataset_dir data_method2_overlap --output_dir onnx_runs/method2
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Complete ONNX inference and evaluation pipeline")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing the trained model checkpoint")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing the dataset (must have dev.jsonl)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save ONNX model and results")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length (default: 128)")
    parser.add_argument("--latency_runs", type=int, default=30,
                        help="Number of latency measurement runs (default: 30)")
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    dev_file = dataset_dir / "dev.jsonl"
    if not dev_file.exists():
        raise FileNotFoundError(f"Dev file not found: {dev_file}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    project_root = Path(__file__).resolve().parent.parent
    
    logging.info("=" * 80)
    logging.info("ONNX INFERENCE & EVALUATION PIPELINE")
    logging.info("=" * 80)
    logging.info("Checkpoint: %s", checkpoint_dir)
    logging.info("Dataset: %s", dataset_dir)
    logging.info("Output: %s", output_dir)
    logging.info("Max Length: %d", args.max_length)
    logging.info("")
    
    # Step 1: Export to ONNX
    logging.info("Step 1: Exporting model to ONNX...")
    export_cmd = [
        sys.executable,
        str(project_root / "src" / "export_to_onnx.py"),
        "--checkpoint_dir", str(checkpoint_dir),
        "--output_dir", str(output_dir),
        "--max_length", str(args.max_length),
    ]
    result = subprocess.run(export_cmd, cwd=project_root, capture_output=False)
    if result.returncode != 0:
        logging.error("ONNX export failed")
        sys.exit(1)
    
    logging.info("ONNX export complete")
    logging.info("")
    
    # Step 2: Run predictions
    logging.info("Step 2: Generating predictions on dev set...")
    pred_file = output_dir / "dev_pred.json"
    predict_cmd = [
        sys.executable,
        str(project_root / "src" / "predict_onnx.py"),
        "--model_dir", str(output_dir),
        "--input", str(dev_file),
        "--output", str(pred_file),
        "--max_length", str(args.max_length),
    ]
    result = subprocess.run(predict_cmd, cwd=project_root, capture_output=False)
    if result.returncode != 0:
        logging.error("Prediction failed")
        sys.exit(1)
    
    logging.info("Predictions saved to %s", pred_file)
    logging.info("")
    
    # Step 3: Evaluate metrics
    logging.info("Step 3: Computing evaluation metrics...")
    detailed_metrics_file = output_dir / "detailed_metrics.json"
    eval_cmd = [
        sys.executable,
        str(project_root / "src" / "eval_span_f1.py"),
        "--gold", str(dev_file),
        "--pred", str(pred_file),
        "--metrics_out", str(detailed_metrics_file),
    ]
    result = subprocess.run(eval_cmd, cwd=project_root, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("Evaluation failed: %s", result.stderr)
        sys.exit(1)
    
    logging.info("Detailed metrics saved to %s", detailed_metrics_file)
    
    # Load evaluation metrics
    with open(detailed_metrics_file, "r", encoding="utf-8") as f:
        detailed_metrics = json.load(f)
    
    # Step 4: Measure latency
    logging.info("Step 4: Measuring inference latency...")
    latency_file = output_dir / "latency.json"
    latency_cmd = [
        sys.executable,
        str(project_root / "src" / "measure_latency_onnx.py"),
        "--model_dir", str(output_dir),
        "--input", str(dev_file),
        "--runs", str(args.latency_runs),
        "--max_length", str(args.max_length),
        "--metrics_out", str(latency_file),
    ]
    result = subprocess.run(latency_cmd, cwd=project_root, capture_output=False)
    if result.returncode != 0:
        logging.error("Latency measurement failed")
        sys.exit(1)
    
    logging.info("Latency metrics saved to %s", latency_file)
    
    # Load latency metrics
    with open(latency_file, "r", encoding="utf-8") as f:
        latency_metrics = json.load(f)
    
    # Step 5: Combine all metrics
    logging.info("Step 5: Combining all metrics...")
    
    overall_metrics = detailed_metrics.get("overall", {})
    per_entity = detailed_metrics.get("per_entity", {})
    pii_metrics = detailed_metrics.get("pii", {})
    
    combined_metrics = {
        "overall": {
            "pii_precision_micro": pii_metrics.get("precision", overall_metrics.get("pii_precision", 0.0)),
            "pii_recall_micro": pii_metrics.get("recall", overall_metrics.get("pii_recall", 0.0)),
            "pii_f1_micro": pii_metrics.get("f1", overall_metrics.get("pii_f1", 0.0)),
            "overall_precision_micro": overall_metrics.get("overall_precision", 0.0),
            "overall_recall_micro": overall_metrics.get("overall_recall", 0.0),
            "overall_f1_micro": overall_metrics.get("overall_f1", 0.0),
        },
        "per_entity": per_entity,
        "latency": {
            "p50_ms": latency_metrics["p50_ms"],
            "p95_ms": latency_metrics["p95_ms"],
            "mean_ms": latency_metrics["mean_ms"],
            "runs": latency_metrics["runs"],
            "model_size_mb": latency_metrics["model_size_mb"],
        }
    }
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2, ensure_ascii=False)
    
    logging.info("Combined metrics saved to %s", metrics_file)
    logging.info("")
    
    # Display summary
    logging.info("=" * 80)
    logging.info("ONNX INFERENCE RESULTS")
    logging.info("=" * 80)
    logging.info("OVERALL METRICS:")
    logging.info("  Precision: %.3f | Recall: %.3f | F1: %.3f",
                 combined_metrics["overall"]["overall_precision_micro"],
                 combined_metrics["overall"]["overall_recall_micro"],
                 combined_metrics["overall"]["overall_f1_micro"])
    logging.info("")
    logging.info("PII METRICS:")
    logging.info("  Precision: %.3f | Recall: %.3f | F1: %.3f",
                 combined_metrics["overall"]["pii_precision_micro"],
                 combined_metrics["overall"]["pii_recall_micro"],
                 combined_metrics["overall"]["pii_f1_micro"])
    logging.info("")
    logging.info("LATENCY METRICS:")
    logging.info("  p50: %.2f ms | p95: %.2f ms",
                 combined_metrics["latency"]["p50_ms"],
                 combined_metrics["latency"]["p95_ms"])
    logging.info("  Model size: %.2f MB", combined_metrics["latency"]["model_size_mb"])
    logging.info("")
    logging.info("All results saved to: %s", output_dir)
    logging.info("  - ONNX Model: %s", output_dir / "model.onnx")
    logging.info("  - Predictions: %s", pred_file)
    logging.info("  - Metrics: %s", metrics_file)
    logging.info("  - Detailed metrics: %s", detailed_metrics_file)
    logging.info("  - Latency: %s", latency_file)
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

