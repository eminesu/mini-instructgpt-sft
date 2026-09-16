"""
evaluate_base_vs_sft.py
-----------------------
Person 1 — Steps 3 & 4: Base vs SFT comparison and evaluation.

For 20 held-out prompts:
  * Generate a reply from the Base (pretrained distilgpt2) model
  * Generate a reply from the SFT model
  * Score both with the constraint-following rubric
  * Save side-by-side examples + a metrics table (JSON + CSV + bar chart)
  * Emit a human-evaluation sheet for the manual "which is more natural?" rating

This answers Person 1's question: "Why is SFT necessary, and why is it not
enough on its own?"  (The PPO column is filled in later by Person 2 in the
joint Base -> SFT -> PPO evaluation.)
"""

import csv
import json
import os
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import load_model, generate_reply
from constraints import score_constraints, constraint_rate

# 20 held-out customer prompts (not in the SFT training set) -------------------
HELD_OUT_PROMPTS = [
    "My order is stuck in customs.",
    "The screen on my new tablet is cracked.",
    "I was double-billed for shipping.",
    "Can you help me find my tracking number?",
    "The wrong size shoes were delivered.",
    "I've been waiting on hold for an hour.",
    "My loyalty points disappeared from my account.",
    "The fabric started tearing after one wear.",
    "I need to update the email on my account.",
    "My replacement order also arrived damaged.",
    "Do you price match competitors?",
    "The gift card I bought isn't working.",
    "My package was marked delivered but it's missing.",
    "Can I cancel an order placed five minutes ago?",
    "The instructions are in a language I can't read.",
    "I was charged a fee I don't recognize.",
    "How do I return a faulty headphone set?",
    "The promised next-day delivery never came.",
    "My account got locked for no reason.",
    "Can I get an invoice for my business records?",
]

SEED = 42


def run():
    import torch
    torch.manual_seed(SEED)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    sft_path = os.path.join(out_dir, "sft_model")

    print("Loading Base model (distilgpt2)...")
    base_model, base_tok = load_model("distilgpt2")
    print("Loading SFT model...")
    sft_model, sft_tok = load_model(sft_path)

    rows = []
    base_rates, sft_rates = [], []

    for i, prompt in enumerate(HELD_OUT_PROMPTS, 1):
        base_reply = generate_reply(base_model, base_tok, prompt)
        sft_reply = generate_reply(sft_model, sft_tok, prompt)

        b_rate = constraint_rate(base_reply)
        s_rate = constraint_rate(sft_reply)
        base_rates.append(b_rate)
        sft_rates.append(s_rate)

        rows.append({
            "id": i,
            "prompt": prompt,
            "base_reply": base_reply,
            "sft_reply": sft_reply,
            "base_constraint_rate": round(b_rate, 3),
            "sft_constraint_rate": round(s_rate, 3),
            "base_breakdown": score_constraints(base_reply),
            "sft_breakdown": score_constraints(sft_reply),
        })
        print(f"[{i:2d}/20] base={b_rate:.2f}  sft={s_rate:.2f}  | {prompt}")

    # ---- Save side-by-side examples (JSON) ----------------------------------
    with open(os.path.join(out_dir, "base_vs_sft_examples.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # ---- Constraint-following summary table ---------------------------------
    constraint_keys = list(rows[0]["base_breakdown"].keys())
    summary = {"constraint": [], "base": [], "sft": []}
    for key in constraint_keys:
        summary["constraint"].append(key)
        summary["base"].append(mean(r["base_breakdown"][key] for r in rows))
        summary["sft"].append(mean(r["sft_breakdown"][key] for r in rows))
    summary["constraint"].append("OVERALL_RATE")
    summary["base"].append(mean(base_rates))
    summary["sft"].append(mean(sft_rates))

    with open(os.path.join(out_dir, "constraint_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "constraint_summary.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["constraint", "base", "sft"])
        for c, b, s in zip(summary["constraint"], summary["base"], summary["sft"]):
            w.writerow([c, f"{b:.3f}", f"{s:.3f}"])

    print("\n=== Constraint-following summary (mean over 20 prompts) ===")
    print(f"{'constraint':<18}{'base':>8}{'sft':>8}")
    for c, b, s in zip(summary["constraint"], summary["base"], summary["sft"]):
        print(f"{c:<18}{b:>8.3f}{s:>8.3f}")

    # ---- Bar chart ----------------------------------------------------------
    keys = summary["constraint"]
    x = range(len(keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - width / 2 for i in x], summary["base"], width,
           label="Base", color="#9ca3af")
    ax.bar([i + width / 2 for i in x], summary["sft"], width,
           label="SFT", color="#2563eb")
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_ylabel("Pass rate")
    ax.set_title("Constraint-following: Base vs SFT (20 held-out prompts)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "constraint_comparison.png"), dpi=130)

    # ---- Human-evaluation sheet ("which is more natural?") ------------------
    # Randomize left/right per row so the rater can't tell which is SFT.
    import random
    rng = random.Random(SEED)
    human_rows = []
    for r in rows:
        if rng.random() < 0.5:
            left, right, mapping = r["base_reply"], r["sft_reply"], ("A=base", "B=sft")
        else:
            left, right, mapping = r["sft_reply"], r["base_reply"], ("A=sft", "B=base")
        human_rows.append({
            "id": r["id"],
            "prompt": r["prompt"],
            "reply_A": left,
            "reply_B": right,
            "winner (fill: A / B / tie)": "",
            "_hidden_mapping": mapping,
        })
    with open(os.path.join(out_dir, "human_eval_sheet.json"), "w",
              encoding="utf-8") as f:
        json.dump(human_rows, f, ensure_ascii=False, indent=2)

    print("\nArtifacts written to outputs/:")
    for name in ["base_vs_sft_examples.json", "constraint_summary.json",
                 "constraint_summary.csv", "constraint_comparison.png",
                 "human_eval_sheet.json"]:
        print("  -", name)


if __name__ == "__main__":
    run()
