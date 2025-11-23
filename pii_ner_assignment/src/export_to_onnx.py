#!/usr/bin/env python3
"""
Export a trained token classification model to ONNX format.

Usage:
    python src/export_to_onnx.py --checkpoint_dir out_method2/best_model --output_dir onnx_models/method2
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def export_to_onnx(checkpoint_dir: Path, output_dir: Path, max_length: int = 128):
    """
    Export a Hugging Face token classification model to ONNX format.
    
    Args:
        checkpoint_dir: Directory containing the PyTorch checkpoint
        output_dir: Directory to save the ONNX model
        max_length: Maximum sequence length for the model
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("Loading model and tokenizer from %s", checkpoint_dir)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    
    model.eval()
    
    # Create dummy input for tracing
    logging.info("Creating dummy input (max_length=%d)", max_length)
    dummy_text = "This is a sample sentence for ONNX export tracing."
    dummy_input = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    # Define input and output names
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    # Dynamic axes for variable batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }
    
    onnx_path = output_dir / "model.onnx"
    logging.info("Exporting model to ONNX: %s", onnx_path)
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
        )
    
    logging.info("ONNX model saved to %s", onnx_path)
    
    # Copy tokenizer files
    logging.info("Copying tokenizer files...")
    tokenizer.save_pretrained(output_dir)
    
    # Save config for reference
    config_path = output_dir / "export_config.json"
    config = {
        "source_checkpoint": str(checkpoint_dir),
        "max_length": max_length,
        "opset_version": 14,
        "input_names": input_names,
        "output_names": output_names,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    logging.info("Export configuration saved to %s", config_path)
    
    # Verify the export by loading it
    try:
        import onnxruntime as ort
        logging.info("Verifying ONNX model...")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        
        # Run a test inference
        ort_inputs = {
            "input_ids": dummy_input["input_ids"].numpy(),
            "attention_mask": dummy_input["attention_mask"].numpy(),
        }
        ort_outputs = session.run(None, ort_inputs)
        
        # Compare with PyTorch output
        with torch.no_grad():
            pt_outputs = model(dummy_input["input_ids"], dummy_input["attention_mask"])
        
        pt_logits = pt_outputs.logits.numpy()
        ort_logits = ort_outputs[0]
        
        max_diff = abs(pt_logits - ort_logits).max()
        logging.info("Max difference between PyTorch and ONNX: %.6f", max_diff)
        
        if max_diff < 1e-4:
            logging.info("ONNX export verification passed!")
        else:
            logging.warning("ONNX export has numerical differences > 1e-4")
        
    except ImportError:
        logging.warning("onnxruntime not installed; skipping verification")
        logging.warning("Install with: pip install onnxruntime")
    
    logging.info("Export complete! Files saved to: %s", output_dir)
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export token classification model to ONNX")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing the trained model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the ONNX model")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length (default: 128)")
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    logging.info("=" * 80)
    logging.info("ONNX MODEL EXPORT")
    logging.info("=" * 80)
    logging.info("Checkpoint: %s", checkpoint_dir)
    logging.info("Output: %s", output_dir)
    logging.info("Max Length: %d", args.max_length)
    logging.info("")
    
    export_to_onnx(checkpoint_dir, output_dir, args.max_length)
    
    logging.info("=" * 80)
    logging.info("To use the ONNX model, run:")
    logging.info("  python src/predict_onnx.py --model_dir %s --input data/dev.jsonl --output predictions.json", output_dir)
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

