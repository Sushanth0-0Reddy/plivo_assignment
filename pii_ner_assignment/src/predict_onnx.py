#!/usr/bin/env python3
"""
Run inference using an ONNX model for token classification.

Usage:
    python src/predict_onnx.py --model_dir onnx_models/method2 --input data/dev.jsonl --output predictions.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from transformers import AutoTokenizer

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

from labels import ID2LABEL, label_is_pii

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def bio_to_spans(tokens: List[str], labels: List[str], char_offsets: List[tuple], original_text: str) -> List[Dict[str, Any]]:
    """
    Convert BIO labels to character-level spans.
    
    Args:
        tokens: List of tokens
        labels: List of BIO labels
        char_offsets: List of (start, end) character offsets for each token
        original_text: Original input text
    
    Returns:
        List of entity dictionaries with start, end, label, text, and pii fields
    """
    entities = []
    current_entity = None
    
    for token, label, (start, end) in zip(tokens, labels, char_offsets):
        if label.startswith("B-"):
            # Save previous entity if exists
            if current_entity:
                entities.append(current_entity)
            
            # Start new entity
            entity_type = label[2:]
            current_entity = {
                "start": int(start),  # Convert numpy int64 to Python int
                "end": int(end),
                "label": entity_type,
                "text": original_text[int(start):int(end)],
                "pii": label_is_pii(entity_type)
            }
        
        elif label.startswith("I-") and current_entity:
            # Continue current entity
            entity_type = label[2:]
            if entity_type == current_entity["label"]:
                current_entity["end"] = int(end)
                current_entity["text"] = original_text[current_entity["start"]:int(end)]
        
        else:  # O label or mismatched I- tag
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Don't forget the last entity
    if current_entity:
        entities.append(current_entity)
    
    return entities


def predict_onnx(session: ort.InferenceSession, tokenizer: AutoTokenizer, text: str, max_length: int = 128) -> List[Dict[str, Any]]:
    """
    Run ONNX inference on a single text and return entity spans.
    
    Args:
        session: ONNX Runtime inference session
        tokenizer: Hugging Face tokenizer
        text: Input text
        max_length: Maximum sequence length
    
    Returns:
        List of entity dictionaries
    """
    # Tokenize
    encoded = tokenizer(
        text,
        return_tensors="np",
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True
    )
    
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    offset_mapping = encoded["offset_mapping"][0]
    
    # Run ONNX inference
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    ort_outputs = session.run(None, ort_inputs)
    logits = ort_outputs[0][0]  # Shape: (seq_len, num_labels)
    
    # Get predicted labels
    pred_label_ids = np.argmax(logits, axis=-1)
    
    # Convert to BIO labels and extract spans
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    labels = [ID2LABEL[label_id] for label_id in pred_label_ids]
    
    # Filter out special tokens and padding
    valid_indices = []
    valid_tokens = []
    valid_labels = []
    valid_offsets = []
    
    for i, (token, label, (start, end)) in enumerate(zip(tokens, labels, offset_mapping)):
        if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token] and end > 0:
            valid_indices.append(i)
            valid_tokens.append(token)
            valid_labels.append(label)
            valid_offsets.append((start, end))
    
    # Convert BIO to spans
    entities = bio_to_spans(valid_tokens, valid_labels, valid_offsets, text)
    
    return entities


def main():
    parser = argparse.ArgumentParser(description="Run inference using ONNX model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the ONNX model")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for predictions")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length (default: 128)")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    onnx_path = model_dir / "model.onnx"
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    logging.info("Loading ONNX model from %s", onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    logging.info("Loading tokenizer from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    logging.info("Processing input file: %s", args.input)
    predictions = {}
    
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)
            text = data["text"]
            utt_id = data.get("id", f"utt_{i:04d}")
            
            entities = predict_onnx(session, tokenizer, text, args.max_length)
            
            # Store as dict: {utterance_id: [entities]}
            predictions[utt_id] = entities
            
            if i % 100 == 0:
                logging.info("Processed %d utterances", i)
    
    logging.info("Saving predictions to %s", args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    logging.info("Inference complete! Processed %d utterances", len(predictions))


if __name__ == "__main__":
    main()

