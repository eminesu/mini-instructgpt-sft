"""
train_sft.py
------------
Person 1 — Step 2: Supervised Fine-Tuning of DistilGPT-2.

Fine-tunes DistilGPT-2 on the 50 polite customer-support demonstrations,
tracks training/validation loss, saves the SFT model, and plots the loss curve.

This is the SFT model that becomes the frozen reference for Person 2's PPO step.

Usage:
    python train_sft.py --epochs 8 --batch_size 4 --lr 5e-5
"""

import argparse
import json
import os
import math
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_NAME = "distilgpt2"
MAX_LEN = 192


class SFTDataset(Dataset):
    """Causal-LM dataset. Loss is masked on the prompt tokens so the model is
    only trained to generate the agent response, not to reproduce the prompt."""

    def __init__(self, records: List[Dict[str, str]], tokenizer):
        self.records = records
        self.tok = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt_ids = self.tok(rec["prompt_only"], add_special_tokens=False)["input_ids"]
        full_ids = self.tok(rec["text"], add_special_tokens=False)["input_ids"][:MAX_LEN]

        input_ids = full_ids
        labels = list(full_ids)
        # Mask the prompt portion from the loss
        n_prompt = min(len(prompt_ids), len(labels))
        for i in range(n_prompt):
            labels[i] = -100
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def load_records(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total += out.loss.item()
        n += 1
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "data")
    out_dir = os.path.join(here, "..", "outputs")
    model_dir = os.path.join(out_dir, "sft_model")
    os.makedirs(model_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    train_records = load_records(os.path.join(data_dir, "sft_train.json"))
    val_records = load_records(os.path.join(data_dir, "sft_val.json"))

    train_ds = SFTDataset(train_records, tokenizer)
    val_ds = SFTDataset(val_records, tokenizer)
    pad_id = tokenizer.pad_token_id

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, pad_id))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    history = {"step": [], "train_loss": [], "epoch": [], "val_loss": []}
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            history["step"].append(step)
            history["train_loss"].append(loss.item())

        val_loss = evaluate(model, val_loader, device)
        history["epoch"].append(epoch)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={loss.item():.4f} "
              f"| val_loss={val_loss:.4f} | val_ppl={math.exp(val_loss):.2f}")

    # ---- Save model + tokenizer (this is Person 2's frozen reference) -------
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"\nSFT model saved to: {model_dir}")

    with open(os.path.join(out_dir, "sft_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ---- Plot loss curves ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["step"], history["train_loss"], color="#2563eb")
    ax1.set_title("SFT training loss (per step)")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss"); ax1.grid(alpha=0.3)

    ax2.plot(history["epoch"], history["val_loss"], marker="o", color="#dc2626")
    ax2.set_title("SFT validation loss (per epoch)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "sft_loss_curve.png")
    fig.savefig(plot_path, dpi=130)
    print(f"Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    main()
