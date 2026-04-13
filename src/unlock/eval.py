"""Benchmark evaluation for capability directions.

Loads a held-out dataset (MATH by default), runs a causal LM with optional
residual-stream steering via :class:`unlock.transfer.DirectionContext`, and
reports exact-match accuracy.
"""

from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlock.transfer import DirectionContext, load_direction


def load_math_subset(n: int = 200) -> list[dict[str, str]]:
    """Load the first ``n`` test examples from hendrycks/competition_math.

    Returns a list of ``{"question": str, "answer": str}``. The ``"answer"``
    field is taken from the dataset's ``"solution"`` column verbatim — callers
    that need the boxed final answer should post-process.
    """
    ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
    ds = ds.select(range(min(n, len(ds))))
    return [
        {"question": row["problem"], "answer": row["solution"]}
        for row in ds
    ]


def exact_match(prediction: str, gold: str) -> bool:
    """Whitespace-stripped, lowercased string equality."""
    return prediction.strip().lower() == gold.strip().lower()


def run_benchmark(
    model_name: str,
    dataset: list[dict[str, str]],
    direction_path: str | None = None,
    alpha: float = 1.0,
    device: str = "cpu",
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Run ``model_name`` over ``dataset`` and return exact-match accuracy.

    If ``direction_path`` is supplied, generation is wrapped in a
    :class:`DirectionContext` so each hooked layer has
    ``alpha * direction`` added to its residual stream.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    directions = None
    if direction_path is not None:
        directions, _ = load_direction(direction_path)

    def _generate(question: str) -> str:
        inputs = tokenizer(question, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)

    correct = 0
    total = len(dataset)

    if directions is not None:
        with DirectionContext(model, tokenizer, directions, alpha=alpha):
            for ex in tqdm(dataset, desc="eval"):
                pred = _generate(ex["question"])
                if exact_match(pred, ex["answer"]):
                    correct += 1
    else:
        for ex in tqdm(dataset, desc="eval"):
            pred = _generate(ex["question"])
            if exact_match(pred, ex["answer"]):
                correct += 1

    accuracy = correct / total if total else 0.0
    return {
        "accuracy": accuracy,
        "n": total,
        "model": model_name,
        "direction": direction_path,
    }
