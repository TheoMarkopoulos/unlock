"""Hugging Face Hub push/pull for capability-direction artifacts.

This module wraps ``huggingface_hub`` to publish and retrieve ``.pt`` direction
files produced by :mod:`unlock.extract` / :mod:`unlock.transfer`, together with
a human-readable model card describing provenance (capability type, source
model, transfer method, optional accuracy delta) and CLI usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from unlock.transfer import load_direction

_ARTIFACT_FILENAME = "direction.pt"
_CARD_FILENAME = "README.md"


def _render_model_card(
    repo_id: str,
    capability: str,
    source_model: str,
    metadata: dict[str, Any],
    accuracy_delta: float | None,
) -> str:
    """Build the README model card shown on the Hub repo page."""
    target_model = metadata.get("target_model")
    base_model = metadata.get("base_model")
    layers = metadata.get("layers")
    hidden_dim = metadata.get("aligned_hidden_dim") or metadata.get("hidden_dim")

    # Transfer method: if the artifact was run through `unlock transfer`,
    # metadata carries `target_model`; otherwise it's a raw extraction.
    if target_model:
        method = (
            f"Least-squares activation-space alignment from `{source_model}` "
            f"into `{target_model}` using paired anchor-prompt activations "
            "(see `unlock.transfer.compute_alignment`)."
        )
    else:
        method = (
            f"Difference-of-means residual-stream direction between "
            f"`{source_model}` and `{base_model}` "
            "(see `unlock.extract.extract_capability_direction`)."
        )

    accuracy_line = (
        f"**Accuracy delta vs. baseline:** {accuracy_delta:+.4f}\n"
        if accuracy_delta is not None
        else ""
    )

    tags = ["unlock", capability]
    frontmatter_tags = "\n".join(f"  - {t}" for t in tags)

    target_for_usage = target_model or "<target-model>"

    return f"""---
tags:
{frontmatter_tags}
library_name: unlock
---

# {repo_id}

Capability-steering direction produced by [unlock](https://github.com/).

- **Capability:** `{capability}`
- **Source model:** `{source_model}`
- **Target model:** `{target_model or "(none — raw extraction)"}`
- **Layers:** `{layers}`
- **Hidden dim:** `{hidden_dim}`

## Transfer method

{method}

{accuracy_line}
## CLI usage

```bash
# Download and evaluate with steering applied
unlock pull --repo-id {repo_id} --out direction.pt
unlock eval --model {target_for_usage} --direction direction.pt --alpha 1.0
```
"""


def push_direction(
    local_path: str | Path,
    repo_id: str,
    capability: str,
    source_model: str,
    accuracy_delta: float | None = None,
) -> str:
    """Publish a direction ``.pt`` and model card to a Hugging Face repo.

    Creates the repo if it does not already exist, uploads the artifact under
    a fixed filename, renders a README model card from the artifact's
    embedded metadata, and tags the repo with ``unlock`` + the capability name.

    Args:
        local_path: Path to a ``.pt`` file written by
            :func:`unlock.extract.save_direction`.
        repo_id: Target repo id in ``"<user-or-org>/<name>"`` form.
        capability: Capability label (e.g. ``"cot"``); used as a tag and in
            the model card.
        source_model: HF id of the model the direction was extracted from;
            recorded in the model card for provenance.
        accuracy_delta: Optional accuracy change vs. baseline (e.g. from
            ``unlock eval``). Included in the card when provided.

    Returns:
        The repo URL as a string.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Direction file not found: {local_path}")

    _, metadata = load_direction(local_path)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=_ARTIFACT_FILENAME,
        repo_id=repo_id,
        repo_type="model",
    )

    card = _render_model_card(
        repo_id=repo_id,
        capability=capability,
        source_model=source_model,
        metadata=metadata,
        accuracy_delta=accuracy_delta,
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo=_CARD_FILENAME,
        repo_id=repo_id,
        repo_type="model",
    )

    # Tags live in repo settings; the README frontmatter above also exposes
    # them for Hub search, but the explicit API call keeps them authoritative.
    try:
        api.update_repo_settings(
            repo_id=repo_id, repo_type="model", tags=["unlock", capability]
        )
    except (AttributeError, TypeError):
        # Older huggingface_hub versions lack `update_repo_settings` / `tags`;
        # the frontmatter tags still apply, so this is a soft failure.
        pass

    return f"https://huggingface.co/{repo_id}"


def pull_direction(repo_id: str, out_path: str | Path) -> Path:
    """Download a direction ``.pt`` from the Hub and print its model card.

    Args:
        repo_id: Source repo id in ``"<user-or-org>/<name>"`` form.
        out_path: Local destination path for the downloaded ``.pt``. Parent
            directories are created if missing.

    Returns:
        The absolute path of the written file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cached = hf_hub_download(
        repo_id=repo_id, filename=_ARTIFACT_FILENAME, repo_type="model"
    )
    data = Path(cached).read_bytes()
    out_path.write_bytes(data)

    try:
        card_path = hf_hub_download(
            repo_id=repo_id, filename=_CARD_FILENAME, repo_type="model"
        )
        print(Path(card_path).read_text(encoding="utf-8"))
    except EntryNotFoundError:
        print(f"(no {_CARD_FILENAME} found in {repo_id})")

    return out_path.resolve()
