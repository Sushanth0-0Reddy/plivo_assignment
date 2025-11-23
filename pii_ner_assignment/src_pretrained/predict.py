import json
import argparse
import logging
import os

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from labels import hf_label_to_assignment, label_is_pii


def tokens_to_assignment_labels(pred_ids, id2label):
    assignment_labels = []
    for lid in pred_ids:
        key = str(lid)
        label_name = id2label.get(key, id2label.get(lid, "O"))
        assignment_label = hf_label_to_assignment(label_name)
        assignment_labels.append(assignment_label if assignment_label else "O")
    return assignment_labels


def labels_to_spans(offsets, assignment_labels):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), label in zip(offsets, assignment_labels):
        if start == 0 and end == 0:
            continue
        if label == "O" or label is None:
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue
        if current_label is None:
            current_label = label
            current_start = start
            current_end = end
        elif current_label == label:
            current_end = end
        else:
            spans.append((current_start, current_end, current_label))
            current_label = label
            current_start = start
            current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out_pretrained")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out_pretrained/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_file", default=None, help="Optional file to append prediction logs.")
    args = ap.parse_args()
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    model_path = args.model_dir
    if args.model_name:
        model_path = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()
    id2label = model.config.id2label

    results = {}
    count = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            assignment_labels = tokens_to_assignment_labels(pred_ids, id2label)
            spans = labels_to_spans(offsets, assignment_labels)
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info("Wrote predictions for %d utterances to %s", len(results), args.output)
    if args.log_file:
        with open(args.log_file, "a", encoding="utf-8") as logf:
            logf.write(
                json.dumps(
                    {
                        "event": "prediction_complete",
                        "inputs": args.input,
                        "output": args.output,
                        "count": len(results),
                        "model_dir": args.model_dir,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
