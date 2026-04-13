# unlock

> Extract capability directions from one LLM and transfer them into another via linear alignment of the residual stream.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![HuggingFace Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Hub-yellow)](https://huggingface.co/)

---

## What is this?

`unlock` is a reference implementation of the **Master Key Hypothesis**: capabilities like chain-of-thought reasoning, instruction following, or code generation are encoded as approximately linear directions in a model's residual stream, and those directions transfer across models in the same family once you align their hidden spaces.

In plain English — if one model can do something its base can't, the *difference* lives in a small set of vectors. You can extract those vectors, rotate them into a second model's coordinate system, and add them back at inference time to "unlock" the capability without any fine-tuning.

---

## Architecture

```
  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
  │ Source Model │ ──▶ │   Activation     │ ──▶ │    Capability    │ ──▶ │     Linear      │ ──▶ │ Target Model │
  │ (capable)    │     │   Collection     │     │    Direction     │     │    Alignment    │     │  (unlocked)  │
  └──────────────┘     │ (paired prompts) │     │  (source − base) │     │ (anchor acts)   │     └──────────────┘
                       └──────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## Quickstart

```bash
pip install unlock
```

The CLI exposes four core commands — extract, transfer, eval, push.

### 1. Extract a capability direction

```bash
unlock extract \
  --source Qwen/Qwen1.5-7B-Chat \
  --base   Qwen/Qwen1.5-7B \
  --prompts data/cot_prompts.jsonl \
  --layers 16,20,24 \
  --capability cot \
  --out directions/qwen7b_cot.pt
```

### 2. Transfer into a target model

```bash
unlock transfer \
  --direction directions/qwen7b_cot.pt \
  --target Qwen/Qwen1.5-1.8B \
  --anchor-prompts data/anchor_prompts.jsonl \
  --out directions/qwen1_8b_cot_aligned.pt
```

### 3. Evaluate with steering applied

```bash
unlock eval \
  --model Qwen/Qwen1.5-1.8B \
  --dataset math \
  --direction directions/qwen1_8b_cot_aligned.pt \
  --alpha 2.0 \
  --n 200 \
  --out results/qwen1_8b_cot.json
```

### 4. Publish to the Hub

```bash
unlock push \
  --direction directions/qwen1_8b_cot_aligned.pt \
  --repo-id your-username/unlock-cot-qwen1.5-1.8b \
  --capability cot \
  --source-model Qwen/Qwen1.5-7B-Chat \
  --accuracy-delta 0.121
```

---

## How It Works

The pipeline is four steps, each a single-purpose module under `src/unlock/`:

1. **Collect activations** (`extract.collect_activations`) — run paired prompts through both a *source* model (has the capability) and a *base* model (doesn't), capturing residual-stream activations at the requested layers.
2. **Extract the direction** (`extract.extract_capability_direction`) — take the mean difference of pooled activations per layer. This is the capability vector.
3. **Align into the target** (`transfer.compute_alignment`) — fit a linear map from the source's residual space to the target's using anchor-prompt activations as paired samples, then project the direction through it.
4. **Steer at inference** (`transfer.DirectionContext`) — install forward hooks on the target's decoder layers that add `alpha * direction` to the hidden state. Hooks are torn down on context exit, so the base model is never mutated.

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from unlock.transfer import DirectionContext, load_direction

model_name = "Qwen/Qwen1.5-1.8B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

directions, metadata = load_direction("directions/qwen1_8b_cot_aligned.pt")

prompt = "If 3x + 7 = 22, what is x? Show your work."
with DirectionContext(model, tokenizer, directions, alpha=2.0):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=256)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

On context exit, every hook is removed — the model returns to its original behavior even if the `with` block raises.

---

## Results

Transferring a chain-of-thought direction extracted from `Qwen1.5-7B-Chat` into the smaller base `Qwen1.5-1.8B`, evaluated on a 200-example MATH subset:

| Target                | Direction | α   | MATH Accuracy | Δ vs. baseline |
| --------------------- | --------- | --- | ------------- | -------------- |
| `Qwen1.5-1.8B` (base) | —         | —   | 18.5%         | —              |
| `Qwen1.5-1.8B`        | `cot`     | 2.0 | **30.6%**     | **+12.1%**     |

No weights were updated. The delta comes from a single aligned vector per hooked layer added at inference time.

---

## Capability Vectors

Directions are serialized as `.pt` files using `torch.save` with the following structure:

```python
{
    "directions": {layer_idx: np.ndarray(shape=(hidden_dim,), dtype=float32), ...},
    "metadata": {
        "source_model":   "Qwen/Qwen1.5-7B-Chat",
        "base_model":     "Qwen/Qwen1.5-7B",
        "target_model":   "Qwen/Qwen1.5-1.8B",   # present after `transfer`
        "capability":     "cot",
        "hidden_dim":     2048,
        "layers":         [16, 20, 24],
        "num_prompts":    512,
        "timestamp":      "2026-04-12T14:02:11+00:00",
    },
}
```

A direction file for a 7B-to-1.8B transfer across 3 layers is ~25 KB. They're small, composable, and safe to share — they contain no training data and cannot reconstruct the source model.

### Sharing on the Hub

`unlock push` creates a public repo with the `.pt` artifact plus an auto-generated model card that records the source model, capability tag, and measured accuracy delta. Pull it back on any machine:

```bash
unlock pull --repo-id your-username/unlock-cot-qwen1.5-1.8b --out directions/cot.pt
```

---

## Citation

```bibtex
@article{unlock2026,
  title   = {Unlock: Cross-Model Transfer of Capability Directions via Linear Alignment of Residual Streams},
  author  = {Markopoulos, Theo},
  journal = {arXiv preprint arXiv:2604.06377},
  year    = {2026},
}
```

---

## Contributing

Issues and pull requests are welcome. Before opening a PR:

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

Keep new modules typed, under 500 lines, and accompanied by tests. For non-trivial changes, open an issue first to discuss the approach.

---

## License

MIT — see [LICENSE](LICENSE). Capability vectors published under this project inherit the license terms of their respective source models; check each source model's card before redistributing derived directions.
