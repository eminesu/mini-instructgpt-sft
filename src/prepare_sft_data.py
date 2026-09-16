"""
prepare_sft_data.py
--------------------
Person 1 — Step 1: SFT dataset preparation.

Loads the 60 polite customer-support demonstrations, formats each one into a
single SFT training string using a fixed chat template, and produces a
train / validation split (50 / 10).

The template is shared across SFT training, generation, and evaluation so the
model sees a consistent format everywhere.
"""

import json
import os
from typing import Dict, List

# ---- Fixed format template (used everywhere: training + inference) ----------
PROMPT_TEMPLATE = "### Customer:\n{prompt}\n\n### Agent:\n"
RESPONSE_TEMPLATE = "{response}"
EOS = "<|endoftext|>"  # DistilGPT-2 / GPT-2 end-of-text token


def build_training_text(prompt: str, response: str) -> str:
    """Full sequence the model is trained on (prompt + target + EOS)."""
    return PROMPT_TEMPLATE.format(prompt=prompt) + RESPONSE_TEMPLATE.format(response=response) + EOS


def build_prompt_only(prompt: str) -> str:
    """Prompt portion only — used at inference time to let the model complete."""
    return PROMPT_TEMPLATE.format(prompt=prompt)


def load_demonstrations(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["data"]


def make_splits(demos: List[Dict[str, str]], n_val: int = 10):
    """Last `n_val` demonstrations become the validation set."""
    train = demos[:-n_val]
    val = demos[-n_val:]
    return train, val


def to_records(demos: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Attach formatted fields used downstream."""
    records = []
    for d in demos:
        records.append({
            "prompt": d["prompt"],
            "response": d["response"],
            "prompt_only": build_prompt_only(d["prompt"]),
            "text": build_training_text(d["prompt"], d["response"]),
        })
    return records


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "..", "data", "sft_demonstrations.json")
    out_dir = os.path.join(here, "..", "data")

    demos = load_demonstrations(data_path)
    print(f"Loaded {len(demos)} demonstrations.")

    train_demos, val_demos = make_splits(demos, n_val=10)
    train_records = to_records(train_demos)
    val_records = to_records(val_demos)

    train_path = os.path.join(out_dir, "sft_train.json")
    val_path = os.path.join(out_dir, "sft_val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_records)} examples -> {train_path}")
    print(f"Val:   {len(val_records)} examples -> {val_path}")
    print("\nExample training text:\n" + "-" * 40)
    print(train_records[0]["text"])


if __name__ == "__main__":
    main()
