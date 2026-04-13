"""Cross-model direction alignment and residual-stream steering.

This module implements the transfer half of the unlock pipeline:

    1. :func:`compute_alignment` — project per-layer capability directions from a
       source model's residual-stream space into a target model's space by
       fitting a least-squares linear map on paired anchor activations.
    2. :class:`DirectionContext` — a context manager that installs
       forward hooks adding ``alpha * direction`` to the residual stream at
       each hooked layer of a Hugging Face causal LM, and cleanly removes
       them on exit. Works transparently with ``model.generate(...)``.
    3. :func:`load_direction` — inverse of :func:`unlock.extract.save_direction`.

The residual stream is addressed at ``model.model.layers[i].output[0]``,
matching the convention used in :mod:`unlock.extract`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from unlock.extract import collect_activations


def compute_alignment(
    source_directions: dict[int, np.ndarray],
    target_model_name: str,
    anchor_prompts: list[str],
    layers: list[int],
    device: str = "cpu",
    *,
    source_anchor_acts: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Align per-layer directions from a source model into a target model's space.

    For each layer ``i`` we solve the least-squares problem

        ``source_anchor_acts[i] @ W_i ≈ target_anchor_acts[i]``

    where ``W_i`` has shape ``(hidden_dim_source, hidden_dim_target)``. The
    aligned direction is then ``source_directions[i] @ W_i``, i.e. the source
    direction pushed through the fitted linear map. This is the standard
    "Procrustes-style" activation-space alignment used for cross-model
    representation transfer.

    Target anchor activations are collected internally via
    :func:`unlock.extract.collect_activations`. Source anchor activations must
    be supplied by the caller (typically collected ahead of time from the
    source model on the same ``anchor_prompts`` in matched order).

    Args:
        source_directions: Mapping ``layer_idx -> np.ndarray`` of shape
            ``(hidden_dim_source,)`` — the directions to transfer.
        target_model_name: HF model id / local path for the target model.
        anchor_prompts: Prompts whose activations in both models are used as
            pairs to fit the alignment map. Same list used for source and
            target collection; row order must match ``source_anchor_acts``.
        layers: Indices of decoder layers to align. Must be present as keys in
            both ``source_directions`` and ``source_anchor_acts``.
        device: Torch device string for the target-model forward pass.
        source_anchor_acts: Mapping ``layer_idx -> np.ndarray`` of shape
            ``(len(anchor_prompts), hidden_dim_source)`` — activations from
            the source model on ``anchor_prompts`` in the same row order.

    Returns:
        Mapping ``layer_idx -> np.ndarray`` of shape ``(hidden_dim_target,)``,
        dtype ``float32``. One aligned direction per requested layer.

    Raises:
        ValueError: If ``anchor_prompts`` is empty, ``layers`` is empty, or
            per-layer anchor-activation row counts disagree across source and
            target (shape mismatch would otherwise be caught by ``lstsq``).
        KeyError: If a requested layer is missing from ``source_directions``
            or ``source_anchor_acts``.
    """
    if not anchor_prompts:
        raise ValueError("`anchor_prompts` must contain at least one string.")
    if not layers:
        raise ValueError("`layers` must contain at least one layer index.")

    for layer in layers:
        if layer not in source_directions:
            raise KeyError(f"source_directions missing layer {layer}.")
        if layer not in source_anchor_acts:
            raise KeyError(f"source_anchor_acts missing layer {layer}.")

    target_anchor_acts = collect_activations(
        model_name=target_model_name,
        prompts=anchor_prompts,
        layers=layers,
        device=device,
    )

    aligned: dict[int, np.ndarray] = {}
    for layer in layers:
        src_acts = source_anchor_acts[layer].astype(np.float64)
        tgt_acts = target_anchor_acts[layer].astype(np.float64)
        if src_acts.shape[0] != tgt_acts.shape[0]:
            raise ValueError(
                f"anchor row count mismatch at layer {layer}: "
                f"source has {src_acts.shape[0]}, target has {tgt_acts.shape[0]}."
            )

        # Solve src_acts @ W = tgt_acts for W of shape (d_src, d_tgt).
        # lstsq is the minimum-norm least-squares solution — well-behaved
        # when n_anchors < min(d_src, d_tgt), which is the common case.
        w, *_ = np.linalg.lstsq(src_acts, tgt_acts, rcond=None)

        src_dir = source_directions[layer].astype(np.float64)
        aligned_dir = src_dir @ w
        aligned[layer] = aligned_dir.astype(np.float32)

    return aligned


class DirectionContext:
    """Context manager that adds a steering vector to the residual stream.

    On ``__enter__``, a forward hook is registered on each
    ``model.model.layers[i]`` whose index appears in ``directions``. Each hook
    replaces the layer's hidden-state output with

        ``hidden_state + alpha * direction``

    broadcast over the batch and sequence dimensions. On ``__exit__`` all
    hooks are removed, restoring the model to its original behavior — even if
    the ``with`` block raised.

    The hooks compose transparently with ``model.generate(...)``, so the
    intended usage is::

        with DirectionContext(model, tokenizer, directions, alpha=2.0):
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            out = model.generate(ids, max_new_tokens=128)

    Notes:
        * ``model`` must expose ``model.model.layers`` (the standard decoder-
          only HF layout used by Llama/Mistral/Qwen etc.), matching the
          convention of :mod:`unlock.extract`.
        * Direction vectors are cast once at ``__enter__`` to the dtype and
          device of the corresponding layer's parameters, so steering works
          under fp16/bf16 / GPU without per-step conversion overhead.
        * ``tokenizer`` is accepted for API symmetry with callers that also
          need to tokenize inputs; it is not used internally.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        directions: dict[int, np.ndarray],
        alpha: float = 1.0,
    ) -> None:
        """Construct the context manager (no hooks installed yet).

        Args:
            model: A causal LM exposing ``model.model.layers``.
            tokenizer: Matching tokenizer (not used internally; kept in the
                signature for caller ergonomics).
            directions: Mapping ``layer_idx -> np.ndarray`` of shape
                ``(hidden_dim,)``. Only the listed layers are hooked.
            alpha: Scalar multiplier applied to every direction at injection
                time. Positive values steer toward the direction; negative
                away.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.directions = directions
        self.alpha = float(alpha)
        self._handles: list[RemovableHandle] = []

    def __enter__(self) -> "DirectionContext":
        """Install the forward hooks and return ``self``."""
        layers = self.model.model.layers
        for layer_idx, direction in self.directions.items():
            layer_module = layers[layer_idx]
            param = next(layer_module.parameters())
            tensor = torch.as_tensor(
                direction, dtype=param.dtype, device=param.device
            )
            handle = layer_module.register_forward_hook(
                self._make_hook(tensor, self.alpha)
            )
            self._handles.append(handle)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Remove all installed hooks; never suppresses exceptions."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @staticmethod
    def _make_hook(direction: torch.Tensor, alpha: float):
        """Build a forward hook that adds ``alpha * direction`` to hidden states.

        Decoder layer outputs are tuples ``(hidden_states, ...)``. We mutate
        index 0 and pass the rest through untouched. The direction tensor is
        broadcast across batch and sequence dims.
        """

        def hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> Any:
            if isinstance(output, tuple):
                hidden = output[0]
                hidden = hidden + alpha * direction
                return (hidden,) + output[1:]
            return output + alpha * direction

        return hook


def load_direction(
    path: str | Path,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Load a direction artifact written by :func:`unlock.extract.save_direction`.

    Args:
        path: Path to a ``.pt`` file whose payload is a dict with top-level
            keys ``"directions"`` and ``"metadata"``.

    Returns:
        A pair ``(directions, metadata)`` where ``directions`` is the
        ``layer_idx -> np.ndarray`` mapping and ``metadata`` is the
        provenance dict, both restored as-is from the saved payload.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the payload is missing the expected top-level keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Direction file not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "directions" not in payload or "metadata" not in payload:
        raise KeyError(
            "Payload missing expected keys 'directions' and/or 'metadata'; "
            f"got {sorted(payload.keys()) if isinstance(payload, dict) else type(payload)}."
        )
    return payload["directions"], payload["metadata"]
