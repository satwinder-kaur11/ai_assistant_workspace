"""
Token counting and cost calculation utility for WorkMate.

Pricing reference (as of 2025):
  - Claude 3.5 Sonnet:  input $3.00 / 1M tokens, output $15.00 / 1M tokens
  - Claude 3 Haiku:     input $0.25 / 1M tokens, output $1.25  / 1M tokens
  - Ollama (local):     $0.00 (runs on local hardware)
  - Rule-based fallback: $0.00
"""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing table — (input_cost_per_1k, output_cost_per_1k) in USD
# ---------------------------------------------------------------------------
PRICING: Dict[str, tuple[float, float]] = {
    # Claude models
    "claude-3-5-sonnet-20241022": (0.003,   0.015),
    "claude-3-5-sonnet":          (0.003,   0.015),
    "claude-3-sonnet-20240229":   (0.003,   0.015),
    "claude-3-haiku-20240307":    (0.00025, 0.00125),
    "claude-3-opus-20240229":     (0.015,   0.075),
    # Ollama — free (local inference)
    "ollama":                     (0.0,     0.0),
    # Rule-based fallback — free
    "rule-based":                 (0.0,     0.0),
}

# Default model identifier if we can't detect the exact model
_DEFAULT_MODEL = "rule-based"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """
    Estimate token count for a string.
    Uses tiktoken (cl100k_base) when available, otherwise falls back to the
    same character-division heuristic used elsewhere in the codebase (~4 chars/token).
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Graceful fallback — matches existing chunk token_count logic
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str,
) -> float:
    """
    Return the total USD cost for a given number of prompt + completion tokens.
    Looks up the model in PRICING; unknown models default to $0.
    """
    # Normalise model name for lookup
    model_key = _resolve_model_key(model_name)
    input_rate, output_rate = PRICING.get(model_key, (0.0, 0.0))

    cost = (prompt_tokens / 1000 * input_rate) + (completion_tokens / 1000 * output_rate)
    return round(cost, 8)   # keep 8 decimal places for micro-transactions


def _resolve_model_key(model_name: str) -> str:
    """Map a raw model name to a key in the PRICING table."""
    if not model_name:
        return _DEFAULT_MODEL
    name_lower = model_name.lower()
    # Exact match first
    if name_lower in PRICING:
        return name_lower
    # Prefix match
    for key in PRICING:
        if name_lower.startswith(key) or key in name_lower:
            return key
    # Ollama catch-all
    if "ollama" in name_lower:
        return "ollama"
    return _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Usage aggregation helper
# ---------------------------------------------------------------------------

def empty_usage(model_name: str = _DEFAULT_MODEL) -> dict:
    """Return a zeroed token_usage dict ready to accumulate into."""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "model_name": model_name,
    }


def add_usage(
    existing: dict,
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str | None = None,
) -> dict:
    """
    Accumulate token counts into an existing usage dict.
    If model_name is provided, it will overwrite the stored model.
    """
    existing["prompt_tokens"] += prompt_tokens
    existing["completion_tokens"] += completion_tokens
    if model_name:
        existing["model_name"] = model_name
    return existing
