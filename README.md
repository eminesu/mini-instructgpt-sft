# Mini-InstructGPT: Supervised Fine-Tuning (SFT)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=flat)

A from-scratch implementation of **Step 1 (Supervised Fine-Tuning)** of the InstructGPT
pipeline from Ouyang et al. (2022), *"Training language models to follow instructions with
human feedback"* ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)), on a customer-support task.

> This was **my contribution ("Person 1")** to a group seminar project: the SFT stage plus the
> analysis motivating RLHF. A teammate ("Person 2") built the Reward Model + PPO on top of the
> SFT model produced here. This repo contains my part.

## Scope (Person 1)
1. **Dataset** — 60 polite customer-support demonstrations, prompt/response
   format, 50/10 train/val split.
2. **Supervised Fine-Tuning** — fine-tune DistilGPT-2, track train/val loss.
3. **Base vs SFT comparison** — same prompts, two models, side by side.
4. **Evaluation** — 20 held-out prompts: constraint-following rate + a
   blind human-eval sheet ("which reply is more natural?").

## Pipeline mapping to the paper
```
Pretraining (frozen distilgpt2)          <- starting point
        |
        v
Supervised Fine-Tuning  ---------------- Person 1  (this repo)
        |
        v
Preference Data / Reward Model / PPO ---- Person 2
        |
        v
Evaluation (Base -> SFT -> PPO)  -------- joint
```
The SFT model saved at `outputs/sft_model/` is exactly the **frozen reference
policy** Person 2's PPO regularizes against with a KL penalty.

## Layout
```
person1_sft/
├── data/
│   └── sft_demonstrations.json     # 60 demonstrations (source data)
├── src/
│   ├── prepare_sft_data.py         # Step 1: format + train/val split
│   ├── train_sft.py                # Step 2: SFT training + loss plot
│   ├── generate.py                 # shared generation helpers
│   ├── constraints.py              # constraint-following rubric
│   └── evaluate_base_vs_sft.py     # Steps 3 & 4: comparison + evaluation
├── requirements.txt
├── run_all.sh
└── outputs/                        # models, plots, tables (created on run)
```

## How to run
```bash
pip install -r requirements.txt
bash run_all.sh
```
Or step by step:
```bash
cd src
python prepare_sft_data.py
python train_sft.py --epochs 8 --batch_size 4 --lr 5e-5
python evaluate_base_vs_sft.py
```

## Outputs
| File | What it is |
|------|------------|
| `outputs/sft_model/` | Fine-tuned DistilGPT-2 (Person 2's reference policy) |
| `outputs/sft_loss_curve.png` | Training/validation loss curves |
| `outputs/base_vs_sft_examples.json` | 20 side-by-side Base vs SFT replies |
| `outputs/constraint_summary.{json,csv}` | Per-constraint + overall pass rates |
| `outputs/constraint_comparison.png` | Base vs SFT bar chart |
| `outputs/human_eval_sheet.json` | Blind A/B sheet for the "more natural?" rating |

## What this section argues (paper connection)
- The **base** model can produce fluent text but does not reliably follow the
  *intent* of a support request — it drifts, ignores constraints, or gives
  unhelpful replies. This is the paper's "language-modeling objective is
  misaligned" point.
- **SFT** sharply improves instruction/constraint following (empathy, asking
  for the order number, offering a next step) — visible in the constraint table.
- But SFT only imitates demonstrations; it doesn't *optimize* for human
  preference, which is the gap RLHF (Person 2) fills.

## Notes
- DistilGPT-2 is tiny, so absolute reply quality is modest; the point is the
  **relative** Base → SFT improvement, not state-of-the-art replies.
- Loss is masked over the prompt tokens so SFT trains only on the agent reply.
- A fixed seed (42) is used for reproducibility.
