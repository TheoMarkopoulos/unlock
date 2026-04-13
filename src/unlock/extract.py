"""Activation collection and capability-direction extraction.

This module implements the core extraction primitives for the unlock pipeline:

    1. ``collect_activations`` — run a causal LM over a batch of prompts and
       capture residual-stream activations at user-specified layers using
       ``nnsight`` hooks.
    2. ``extract_capability_direction`` — compute a unit-norm difference-of-means
       direction per layer from two activation sets (e.g. a "source" model that
       exhibits some capability vs. a "base" model that does not).
    3. ``save_direction`` — persist the direction dictionary plus metadata as a
       single ``.pt`` file via ``torch.save``.

The residual stream is read at ``model.model.layers[i].output[0]``, which is
the standard hidden-state tensor for decoder-only HF architectures
(Llama, Mistral, Qwen, GPT-NeoX, etc. — layer outputs are tuples whose first
element is the hidden states).

Per-prompt activations are pooled by taking the last non-padding token,
producing a single ``(hidden_dim,)`` vector per prompt per layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from nnsight import NNsight
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def collect_activations(
    model_name: str,
    prompts: list[str],
    layers: list[int],
    device: str = "cpu",
    batch_size: int = 4,
) -> dict[int, np.ndarray]:
    """Collect residual-stream activations for ``prompts`` at the given layers.

    For each prompt, the activation at the last non-padding token position is
    taken as the per-prompt representation for that layer.

    Args:
        model_name: HuggingFace model id or local path passed to
            ``AutoModelForCausalLM.from_pretrained`` / ``AutoTokenizer.from_pretrained``.
        prompts: Raw input strings. Tokenized internally with left/right padding
            as configured by the tokenizer; truncation is enabled.
        layers: Indices into ``model.model.layers`` whose output residual stream
            should be captured.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ...).
            The HF model is moved to this device before tracing.
        batch_size: Number of prompts per forward pass. A tqdm progress bar is
            shown over the batches.

    Returns:
        Mapping ``layer_idx -> np.ndarray`` of shape ``(len(prompts), hidden_dim)``
        and dtype ``float32``. Row order matches ``prompts``.

    Raises:
        ValueError: If ``prompts`` is empty or ``layers`` is empty.
    """
    if not prompts:
        raise ValueError("`prompts` must contain at least one string.")
    if not layers:
        raise ValueError("`layers` must contain at least one layer index.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Decoder-only tokenizers (e.g. Llama) often ship without a pad token;
        # reuse EOS so we can batch-pad without introducing a new vocab entry.
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_model.to(device)
    hf_model.eval()

    model = NNsight(hf_model)

    collected: dict[int, list[np.ndarray]] = {i: [] for i in layers}

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, len(prompts), batch_size),
        total=num_batches,
        desc="Collecting activations",
    ):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        saved: dict[int, Any] = {}
        with torch.no_grad():
            with model.trace(
                enc["input_ids"], attention_mask=enc["attention_mask"]
            ):
                for i in layers:
                    # Decoder layer outputs are tuples; [0] is hidden states of
                    # shape (batch, seq, hidden_dim).
                    saved[i] = model.model.layers[i].output[0].save()

        # Pool: last non-pad token per row.
        attention_mask = enc["attention_mask"]
        last_idx = attention_mask.sum(dim=1) - 1  # (batch,)

        for i in layers:
            hidden = saved[i].value  # (batch, seq, hidden_dim)
            batch_range = torch.arange(hidden.size(0), device=hidden.device)
            last_hidden = hidden[batch_range, last_idx]  # (batch, hidden_dim)
            collected[i].append(last_hidden.detach().float().cpu().numpy())

    return {i: np.concatenate(collected[i], axis=0) for i in layers}


def extract_capability_direction(
    source_acts: dict[int, np.ndarray],
    base_acts: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Compute a unit-norm capability direction per layer via mean difference.

    For each layer ``i``, the direction is

        d_i = normalize(mean(source_acts[i]) - mean(base_acts[i]))

    This is the standard "difference-of-means" probe used for steering and
    activation patching: it points from the base distribution toward the
    source distribution in residual-stream space.

    Args:
        source_acts: Activations from prompts exhibiting the target capability,
            as returned by :func:`collect_activations`.
        base_acts: Activations from matched control prompts lacking the
            capability. Must contain the same layer keys as ``source_acts`` and
            matching ``hidden_dim`` per layer.

    Returns:
        Mapping ``layer_idx -> np.ndarray`` of shape ``(hidden_dim,)``, dtype
        ``float32``, L2-normalized. If the raw difference has zero norm at a
        layer (degenerate case), the unnormalized zero vector is returned for
        that layer rather than raising.

    Raises:
        KeyError: If ``source_acts`` and ``base_acts`` do not share layer keys.
        ValueError: If the two sets disagree on ``hidden_dim`` at any layer.
    """
    if set(source_acts.keys()) != set(base_acts.keys()):
        raise KeyError(
            "source_acts and base_acts must share the same layer keys; "
            f"got {sorted(source_acts)} vs {sorted(base_acts)}."
        )

    directions: dict[int, np.ndarray] = {}
    for layer in source_acts:
        src = source_acts[layer]
        bas = base_acts[layer]
        if src.shape[1] != bas.shape[1]:
            raise ValueError(
                f"hidden_dim mismatch at layer {layer}: "
                f"source has {src.shape[1]}, base has {bas.shape[1]}."
            )

        diff = src.mean(axis=0) - bas.mean(axis=0)
        norm = float(np.linalg.norm(diff))
        if norm > 0.0:
            diff = diff / norm
        directions[layer] = diff.astype(np.float32)

    return directions


def save_direction(
    directions: dict[int, np.ndarray],
    out_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Save directions and metadata to a ``.pt`` file.

    The on-disk payload is a dict with two top-level keys:

        - ``"directions"``: the ``directions`` mapping as-is.
        - ``"metadata"``: the ``metadata`` mapping as-is.

    Expected (but not enforced) metadata keys:
        ``source_model``, ``base_model``, ``capability``, ``hidden_dim``,
        ``layers``, ``timestamp``.

    Args:
        directions: Mapping ``layer_idx -> np.ndarray`` direction vectors,
            typically produced by :func:`extract_capability_direction`.
        out_path: Destination file path. Parent directories are created if
            missing. A ``.pt`` suffix is conventional but not required.
        metadata: Free-form provenance dict saved alongside the directions.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"directions": directions, "metadata": metadata}
    torch.save(payload, out_path)
