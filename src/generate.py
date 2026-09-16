"""
generate.py
-----------
Person 1 — shared generation helpers for Base vs SFT comparison and evaluation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_sft_data import build_prompt_only

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path_or_name: str):
    tok = AutoTokenizer.from_pretrained(path_or_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path_or_name).to(DEVICE)
    model.eval()
    return model, tok


@torch.no_grad()
def generate_reply(model, tok, customer_prompt: str,
                   max_new_tokens: int = 80, temperature: float = 0.7,
                   top_p: float = 0.9) -> str:
    """Generate an agent reply for a single customer message."""
    prompt_text = build_prompt_only(customer_prompt)
    inputs = tok(prompt_text, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    # Decode only the newly generated portion
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)
    # Trim at any new section marker the model may hallucinate
    for marker in ["### Customer:", "### Agent:"]:
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()
