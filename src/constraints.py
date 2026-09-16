"""
constraints.py
--------------
Person 1 — Step 4: automatic constraint-following metric.

Defines a small rubric capturing the polite-support behaviours the SFT model is
supposed to learn, and scores any reply against it. This gives the
"constraint-following rate" reported in the evaluation table.

The checks are intentionally simple and transparent (keyword / heuristic based)
so the metric is explainable, matching the InstructGPT idea of measuring whether
the model "reliably follows explicit constraints in the instruction."
"""

import re
from typing import Dict

# Heuristic lexicons -----------------------------------------------------------
_POLITE = ["please", "thank", "happy to", "glad to", "of course",
           "i understand", "i'd be", "i would be"]
_EMPATHY = ["sorry", "apologi", "frustrat", "understand how", "that's not",
            "disappoint", "i understand"]
_ASK_INFO = ["order number", "could you", "can you", "please share",
             "please provide", "confirm", "let me know", "tell me"]
_NEXT_STEP = ["refund", "replacement", "replace", "track", "arrange",
              "process", "ship", "exchange", "resolve", "help", "label",
              "update", "send"]
_RUDE = ["i don't know", "not my problem", "whatever", "deal with it",
         "no idea", "can't help"]


def _has_any(text: str, words) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def score_constraints(reply: str) -> Dict[str, int]:
    """Return per-constraint pass/fail (1/0) for a single reply."""
    t = reply.lower().strip()
    polite = int(_has_any(t, _POLITE))
    empathy = int(_has_any(t, _EMPATHY))
    asks = int(_has_any(t, _ASK_INFO))
    next_step = int(_has_any(t, _NEXT_STEP))
    not_rude = int(not _has_any(t, _RUDE))
    nonempty = int(len(t.split()) >= 5)  # penalize empty / stub replies

    return {
        "polite": polite,
        "empathy": empathy,
        "asks_for_info": asks,
        "offers_next_step": next_step,
        "not_rude": not_rude,
        "substantive": nonempty,
    }


def constraint_rate(reply: str) -> float:
    """Fraction of constraints satisfied (0..1) for one reply."""
    s = score_constraints(reply)
    return sum(s.values()) / len(s)
