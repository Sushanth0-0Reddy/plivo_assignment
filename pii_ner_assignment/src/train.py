import os
import argparse
import json
import logging
import subprocess
import time
import statistics
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS, ID2LABEL, label_is_pii
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_file", default=None, help="Optional path to append JSONL training logs.")
    return ap.parse_args()


def setup_logger(log_file: str = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    return log_file


def log_event(log_file, payload):
    if not log_file:
        return
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def bio_to_spans(offsets, label_ids):
    """Convert BIO predictions to spans."""
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


def compute_prf(tp, fp, fn):
    """Compute precision, recall, F1."""
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def evaluate_on_dev(model, dev_ds, device):
    """Run evaluation on dev set and return metrics."""
    model.eval()
    
    tp_pii = fp_pii = fn_pii = 0
    tp_non = fp_non = fn_non = 0
    tp_all = fp_all = fn_all = 0
    
    with torch.no_grad():
        for item in dev_ds:
            input_ids = torch.tensor([item["input_ids"]], device=device)
            attention_mask = torch.tensor([item["attention_mask"]], device=device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            
            # Get predicted and gold spans
            pred_spans = bio_to_spans(item["offset_mapping"], pred_ids)
            gold_spans = []
            
            # Reconstruct gold spans from labels
            gold_labels = item["labels"]
            gold_spans = bio_to_spans(item["offset_mapping"], gold_labels)
            
            # Convert to sets for comparison
            pred_set = set(pred_spans)
            gold_set = set(gold_spans)
            
            # Overall metrics
            tp_all += len(pred_set & gold_set)
            fp_all += len(pred_set - gold_set)
            fn_all += len(gold_set - pred_set)
            
            # PII vs non-PII metrics
            pred_pii = set((s, e, "PII") for s, e, lab in pred_spans if label_is_pii(lab))
            pred_non = set((s, e, "NON") for s, e, lab in pred_spans if not label_is_pii(lab))
            gold_pii = set((s, e, "PII") for s, e, lab in gold_spans if label_is_pii(lab))
            gold_non = set((s, e, "NON") for s, e, lab in gold_spans if not label_is_pii(lab))
            
            tp_pii += len(pred_pii & gold_pii)
            fp_pii += len(pred_pii - gold_pii)
            fn_pii += len(gold_pii - pred_pii)
            
            tp_non += len(pred_non & gold_non)
            fp_non += len(pred_non - gold_non)
            fn_non += len(gold_non - pred_non)
    
    model.train()
    
    # Compute metrics
    p_all, r_all, f1_all = compute_prf(tp_all, fp_all, fn_all)
    p_pii, r_pii, f1_pii = compute_prf(tp_pii, fp_pii, fn_pii)
    p_non, r_non, f1_non = compute_prf(tp_non, fp_non, fn_non)
    
    return {
        "overall_precision": p_all,
        "overall_recall": r_all,
        "overall_f1": f1_all,
        "pii_precision": p_pii,
        "pii_recall": r_pii,
        "pii_f1": f1_pii,
        "non_pii_precision": p_non,
        "non_pii_recall": r_non,
        "non_pii_f1": f1_non,
    }


def measure_latency(model, tokenizer, dev_ds, device, max_length, runs=30):
    """Measure p50 and p95 latency for the model."""
    model.eval()
    
    # Get sample items from dev set (up to 50 samples for variety)
    sample_size = min(50, len(dev_ds))
    sample_items = [dev_ds[i] for i in range(sample_size)]
    
    if not sample_items:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "runs": 0}
    
    times_ms = []
    
    # Warmup
    for _ in range(5):
        item = sample_items[0]
        input_ids = torch.tensor([item["input_ids"]], device=device)
        attention_mask = torch.tensor([item["attention_mask"]], device=device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Measure latency
    for i in range(runs):
        item = sample_items[i % len(sample_items)]
        input_ids = torch.tensor([item["input_ids"]], device=device)
        attention_mask = torch.tensor([item["attention_mask"]], device=device)
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
    
    p50 = statistics.median(times_ms)
    times_sorted = sorted(times_ms)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1] if len(times_sorted) > 0 else 0.0
    
    model.train()  # Set back to training mode
    
    return {
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "runs": runs
    }


def generate_detailed_metrics(model_dir: Path, dev_path: str, epoch: int, project_root: Path):
    """Generate predictions and detailed per-entity metrics for a checkpoint."""
    # Paths for outputs
    pred_file = model_dir / f"dev_pred.json"
    metrics_file = model_dir / "detailed_metrics.json"
    
    # Run prediction (from project root)
    pred_cmd = [
        "python", "src/predict.py",
        "--model_dir", str(model_dir.absolute()),
        "--input", str(Path(dev_path).absolute()),
        "--output", str(pred_file.absolute())
    ]
    result = subprocess.run(pred_cmd, cwd=str(project_root), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Prediction failed: {result.stderr}")
    
    # Run evaluation (from project root)
    eval_cmd = [
        "python", "src/eval_span_f1.py",
        "--gold", str(Path(dev_path).absolute()),
        "--pred", str(pred_file.absolute()),
        "--metrics_out", str(metrics_file.absolute())
    ]
    result = subprocess.run(eval_cmd, cwd=str(project_root), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {result.stderr}")
    
    # Load and return metrics
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not created: {metrics_file}")
    
    with open(metrics_file, "r", encoding="utf-8") as f:
        detailed = json.load(f)
    
    return detailed, pred_file


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = setup_logger(args.log_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_pii_f1 = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_dl))
        
        # Evaluate on dev set
        dev_metrics = evaluate_on_dev(model, dev_ds, args.device)
        
        # Save checkpoint for this epoch
        epoch_dir = Path(args.out_dir) / f"checkpoint_epoch{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        
        # Measure latency for this epoch
        logging.info("Measuring latency for epoch %d...", epoch + 1)
        latency_metrics = measure_latency(model, tokenizer, dev_ds, args.device, args.max_length, runs=30)
        
        # Generate detailed metrics with per-entity breakdown
        try:
            logging.info("Generating detailed metrics for epoch %d...", epoch + 1)
            project_root = Path(__file__).parent.parent
            detailed_metrics, pred_file = generate_detailed_metrics(epoch_dir, args.dev, epoch + 1, project_root)
            
            # Extract metrics from eval_span_f1.py output format
            # eval_span_f1.py returns: {"pii": {...}, "non_pii": {...}, "per_entity": {...}, "overall_*": ...}
            pii_metrics = detailed_metrics.get("pii", {})
            non_pii_metrics = detailed_metrics.get("non_pii", {})
            per_entity = detailed_metrics.get("per_entity", {})
            
            # Use overall metrics from eval_span_f1.py if available, otherwise fallback to in-memory eval
            overall_precision = detailed_metrics.get("overall_precision", dev_metrics["overall_precision"])
            overall_recall = detailed_metrics.get("overall_recall", dev_metrics["overall_recall"])
            overall_f1 = detailed_metrics.get("overall_f1", dev_metrics["overall_f1"])
            
            # Save detailed metrics in requested format matching assignment requirements
            epoch_metrics_file = epoch_dir / "metrics.json"
            metrics_output = {
                "overall": {
                    "pii_precision_micro": pii_metrics.get("precision", dev_metrics["pii_precision"]),
                    "pii_recall_micro": pii_metrics.get("recall", dev_metrics["pii_recall"]),
                    "pii_f1_micro": pii_metrics.get("f1", dev_metrics["pii_f1"]),
                    "overall_precision_micro": overall_precision,
                    "overall_recall_micro": overall_recall,
                    "overall_f1_micro": overall_f1,
                },
                "per_entity": per_entity,
                "latency": latency_metrics,
            }
            
            with open(epoch_metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_output, f, indent=2, ensure_ascii=False)
            
            # Print comprehensive metrics summary
            logging.info("")
            logging.info("=" * 80)
            logging.info(f"EPOCH {epoch + 1}/{args.epochs} SUMMARY")
            logging.info("=" * 80)
            logging.info(f"Training Loss: {avg_loss:.4f}")
            logging.info("")
            logging.info("OVERALL METRICS:")
            logging.info(f"  Precision: {overall_precision:.3f} | Recall: {overall_recall:.3f} | F1: {overall_f1:.3f}")
            logging.info("")
            logging.info("PII METRICS (Key Target):")
            logging.info(f"  Precision: {metrics_output['overall']['pii_precision_micro']:.3f} | "
                        f"Recall: {metrics_output['overall']['pii_recall_micro']:.3f} | "
                        f"F1: {metrics_output['overall']['pii_f1_micro']:.3f}")
            logging.info("")
            logging.info("PER-ENTITY METRICS:")
            # Sort entities by F1 score for better visibility
            sorted_entities = sorted(per_entity.items(), key=lambda x: x[1].get("f1", 0), reverse=True)
            for entity_type, metrics in sorted_entities:
                p = metrics.get("precision", 0)
                r = metrics.get("recall", 0)
                f1 = metrics.get("f1", 0)
                logging.info(f"  {entity_type:15s} | P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f}")
            logging.info("")
            logging.info("LATENCY METRICS:")
            logging.info(f"  p50: {latency_metrics['p50_ms']:.2f} ms | p95: {latency_metrics['p95_ms']:.2f} ms "
                        f"(target: p95 ≤ 20 ms)")
            logging.info("")
            logging.info(f"✓ Checkpoint saved to: {epoch_dir}")
            logging.info(f"✓ Metrics saved to: {epoch_metrics_file}")
            if pred_file.exists():
                logging.info(f"✓ Predictions saved to: {pred_file}")
            logging.info("=" * 80)
            logging.info("")
            
        except Exception as e:
            logging.warning("Could not generate detailed metrics: %s", e)
            import traceback
            logging.warning(traceback.format_exc())
            # Fallback: save simple metrics from in-memory evaluation
            epoch_metrics_file = epoch_dir / "metrics.json"
            fallback_metrics = {
                "overall": {
                    "pii_precision_micro": dev_metrics["pii_precision"],
                    "pii_recall_micro": dev_metrics["pii_recall"],
                    "pii_f1_micro": dev_metrics["pii_f1"],
                    "overall_precision_micro": dev_metrics["overall_precision"],
                    "overall_recall_micro": dev_metrics["overall_recall"],
                    "overall_f1_micro": dev_metrics["overall_f1"],
                },
                "per_entity": {},  # Empty if detailed eval failed
                "latency": latency_metrics,
                "note": "Detailed metrics generation failed, using in-memory evaluation only"
            }
            with open(epoch_metrics_file, "w", encoding="utf-8") as f:
                json.dump(fallback_metrics, f, indent=2, ensure_ascii=False)
            
            # Print fallback summary
            logging.info("")
            logging.info("=" * 80)
            logging.info(f"EPOCH {epoch + 1}/{args.epochs} SUMMARY (Fallback - Detailed eval failed)")
            logging.info("=" * 80)
            logging.info(f"Training Loss: {avg_loss:.4f}")
            logging.info("")
            logging.info("OVERALL METRICS:")
            logging.info(f"  Precision: {dev_metrics['overall_precision']:.3f} | "
                        f"Recall: {dev_metrics['overall_recall']:.3f} | "
                        f"F1: {dev_metrics['overall_f1']:.3f}")
            logging.info("")
            logging.info("PII METRICS:")
            logging.info(f"  Precision: {dev_metrics['pii_precision']:.3f} | "
                        f"Recall: {dev_metrics['pii_recall']:.3f} | "
                        f"F1: {dev_metrics['pii_f1']:.3f}")
            logging.info("")
            logging.info("LATENCY METRICS:")
            logging.info(f"  p50: {latency_metrics['p50_ms']:.2f} ms | p95: {latency_metrics['p95_ms']:.2f} ms")
            logging.info("")
            logging.info(f"✓ Checkpoint saved to: {epoch_dir}")
            logging.info(f"✓ Metrics saved to: {epoch_metrics_file}")
            logging.info("=" * 80)
            logging.info("")
        
        # Save best model based on PII F1
        current_pii_f1 = dev_metrics["pii_f1"]
        if current_pii_f1 > best_pii_f1:
            best_pii_f1 = current_pii_f1
            best_epoch = epoch + 1
            best_model_dir = Path(args.out_dir) / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            logging.info("✓ New best model (PII F1: %.3f) also saved to %s", best_pii_f1, best_model_dir)
        
        log_event(
            log_file,
            {
                "event": "epoch_end",
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "steps": len(train_dl),
                "model_name": args.model_name,
                "dev_metrics": dev_metrics,
                "is_best": current_pii_f1 == best_pii_f1,
                "checkpoint_dir": str(epoch_dir),
            },
        )

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    logging.info("Saved final model to %s", args.out_dir)
    logging.info("Best model (epoch %d, PII F1: %.3f) saved to %s/best_model", best_epoch, best_pii_f1, args.out_dir)
    log_event(
        log_file,
        {
            "event": "training_complete",
            "epochs": args.epochs,
            "out_dir": args.out_dir,
            "best_epoch": best_epoch,
            "best_pii_f1": best_pii_f1,
        },
    )


if __name__ == "__main__":
    main()
