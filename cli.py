import json
from datetime import datetime, timezone
from pathlib import Path

import click

from unlock.eval import load_math_subset, run_benchmark
from unlock.extract import (
    collect_activations,
    extract_capability_direction,
    save_direction,
)
from unlock.hub import pull_direction, push_direction
from unlock.transfer import compute_alignment, load_direction


def _parse_layers(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _read_prompts(path: str) -> list[str]:
    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["text"])
    return prompts


@click.group()
def unlock():
    """Unlock CLI."""


@unlock.command()
@click.option("--source", "source", required=True, type=str,
              help="Capability-present model name.")
@click.option("--base", "base", required=True, type=str,
              help="Base model without capability.")
@click.option("--prompts", "prompts_path", required=True, type=str,
              help="JSONL file where each line is {\"text\": \"...\"}.")
@click.option("--layers", "layers", type=str, default="16,24,32",
              show_default=True, help="Comma-separated layer indices.")
@click.option("--out", "out", type=str, default="direction.pt",
              show_default=True, help="Output .pt path.")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--capability", type=str, default="cot", show_default=True)
def extract(
    source: str,
    base: str,
    prompts_path: str,
    layers: str,
    out: str,
    device: str,
    capability: str,
) -> None:
    """Extract a capability direction from source vs. base activations."""
    layer_ids = _parse_layers(layers)
    prompts = _read_prompts(prompts_path)

    source_acts = collect_activations(source, prompts, layer_ids, device=device)
    base_acts = collect_activations(base, prompts, layer_ids, device=device)

    directions = extract_capability_direction(source_acts, base_acts)

    hidden_dim = int(next(iter(directions.values())).shape[0])
    metadata = {
        "source_model": source,
        "base_model": base,
        "capability": capability,
        "hidden_dim": hidden_dim,
        "layers": layer_ids,
        "num_prompts": len(prompts),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_direction(directions, out, metadata)

    summary = {
        "status": "ok",
        "out": str(Path(out).resolve()),
        "layers": layer_ids,
        "hidden_dim": hidden_dim,
        "num_prompts": len(prompts),
        "source_model": source,
        "base_model": base,
        "capability": capability,
    }
    click.echo(json.dumps(summary, indent=2))


@unlock.command()
@click.option("--direction", "direction_path", required=True, type=str,
              help="Path to .pt direction file from `unlock extract`.")
@click.option("--target", "target", required=True, type=str,
              help="Target model name.")
@click.option("--anchor-prompts", "anchor_prompts_path", required=True, type=str,
              help="JSONL file for alignment calibration.")
@click.option("--out", "out", type=str, default="aligned_direction.pt",
              show_default=True, help="Output .pt path for aligned direction.")
@click.option("--device", type=str, default="cpu", show_default=True)
def transfer(
    direction_path: str,
    target: str,
    anchor_prompts_path: str,
    out: str,
    device: str,
) -> None:
    """Align a capability direction into a target model's residual space."""
    source_directions, metadata = load_direction(direction_path)
    anchor_prompts = _read_prompts(anchor_prompts_path)

    layer_ids = sorted(int(k) for k in source_directions.keys())
    source_model = metadata.get("source_model")
    if not source_model:
        raise click.ClickException(
            "Direction metadata missing 'source_model'; cannot collect "
            "source anchor activations."
        )

    source_anchor_acts = collect_activations(
        source_model, anchor_prompts, layer_ids, device=device
    )

    aligned = compute_alignment(
        source_directions=source_directions,
        target_model_name=target,
        anchor_prompts=anchor_prompts,
        layers=layer_ids,
        device=device,
        source_anchor_acts=source_anchor_acts,
    )

    aligned_hidden_dim = int(next(iter(aligned.values())).shape[0])
    aligned_metadata = {
        **metadata,
        "target_model": target,
        "aligned_hidden_dim": aligned_hidden_dim,
        "num_anchor_prompts": len(anchor_prompts),
        "aligned_from": str(Path(direction_path).resolve()),
        "alignment_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_direction(aligned, out, aligned_metadata)

    summary = {
        "status": "ok",
        "out": str(Path(out).resolve()),
        "layers": layer_ids,
        "aligned_hidden_dim": aligned_hidden_dim,
        "num_anchor_prompts": len(anchor_prompts),
        "source_model": source_model,
        "target_model": target,
    }
    click.echo(json.dumps(summary, indent=2))


@unlock.command()
@click.option("--model", "model", required=True, type=str,
              help="HF model id or local path to evaluate.")
@click.option("--dataset", "dataset_name", type=str, default="math",
              show_default=True, help="Benchmark dataset identifier.")
@click.option("--direction", "direction_path", type=str, default=None,
              help="Optional .pt direction file to apply during generation.")
@click.option("--alpha", type=float, default=1.0, show_default=True,
              help="Steering strength for the direction (ignored without --direction).")
@click.option("--n", "n", type=int, default=200, show_default=True,
              help="Number of examples to evaluate.")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--out", "out", type=str, default=None,
              help="Optional path to save results JSON.")
def eval(
    model: str,
    dataset_name: str,
    direction_path: str | None,
    alpha: float,
    n: int,
    device: str,
    out: str | None,
) -> None:
    """Evaluate a model (optionally with a steering direction) on a benchmark."""
    if dataset_name == "math":
        data = load_math_subset(n=n)
    else:
        raise click.ClickException(f"Unknown dataset: {dataset_name}")

    results = run_benchmark(
        model_name=model,
        dataset=data,
        direction_path=direction_path,
        alpha=alpha,
        device=device,
    )
    results["dataset"] = dataset_name
    results["alpha"] = alpha if direction_path else None

    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        results["out"] = str(out_path.resolve())

    click.echo(json.dumps(results, indent=2))


@unlock.command()
@click.option("--direction", "direction_path", required=True, type=str,
              help="Local .pt direction file to publish.")
@click.option("--repo-id", "repo_id", required=True, type=str,
              help="Target HF repo id, e.g. 'user/unlock-cot-qwen'.")
@click.option("--capability", required=True, type=str,
              help="Capability label, used as a tag and in the model card.")
@click.option("--source-model", "source_model", required=True, type=str,
              help="HF id of the model this direction was extracted from.")
@click.option("--accuracy-delta", "accuracy_delta", type=float, default=None,
              help="Optional accuracy change vs. baseline to record in the card.")
def push(
    direction_path: str,
    repo_id: str,
    capability: str,
    source_model: str,
    accuracy_delta: float | None,
) -> None:
    """Push a direction artifact and model card to the Hugging Face Hub."""
    url = push_direction(
        local_path=direction_path,
        repo_id=repo_id,
        capability=capability,
        source_model=source_model,
        accuracy_delta=accuracy_delta,
    )
    click.echo(json.dumps({"status": "ok", "repo_id": repo_id, "url": url}, indent=2))


@unlock.command()
@click.option("--repo-id", "repo_id", required=True, type=str,
              help="Source HF repo id to pull from.")
@click.option("--out", "out", required=True, type=str,
              help="Local destination path for the downloaded .pt file.")
def pull(repo_id: str, out: str) -> None:
    """Pull a direction artifact from the Hugging Face Hub and print its card."""
    written = pull_direction(repo_id=repo_id, out_path=out)
    click.echo(json.dumps({"status": "ok", "out": str(written)}, indent=2))


if __name__ == "__main__":
    unlock()
