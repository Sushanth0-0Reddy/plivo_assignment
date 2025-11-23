import os
import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="../bert-pii-detection", help="HF repo or local folder to load")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out_pretrained")
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


def load_hf_label_list(model_name: str):
    config = AutoConfig.from_pretrained(model_name)
    id2label = config.id2label
    labels = []
    for idx in range(len(id2label)):
        key = str(idx) if str(idx) in id2label else idx
        labels.append(id2label[key])
    return labels


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = setup_logger(args.log_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hf_labels = load_hf_label_list(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, hf_labels, max_length=args.max_length, is_train=True)

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
        logging.info("Epoch %s average loss: %.4f", epoch + 1, avg_loss)
        log_event(
            log_file,
            {
                "event": "epoch_end",
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "steps": len(train_dl),
                "model_name": args.model_name,
            },
        )

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    logging.info("Saved model + tokenizer to %s", args.out_dir)
    log_event(
        log_file,
        {
            "event": "training_complete",
            "epochs": args.epochs,
            "out_dir": args.out_dir,
        },
    )


if __name__ == "__main__":
    main()
