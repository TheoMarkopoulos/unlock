"""Tests for extraction, direction I/O, and exact-match scoring."""

from __future__ import annotations

import numpy as np
import pytest

from unlock.eval import exact_match
from unlock.extract import extract_capability_direction, save_direction
from unlock.transfer import load_direction


def test_extract_direction_shape():
    rng = np.random.default_rng(0)
    layers = [0, 1]
    hidden = 64
    source_acts = {i: rng.standard_normal((10, hidden)).astype(np.float32) for i in layers}
    base_acts = {i: rng.standard_normal((10, hidden)).astype(np.float32) for i in layers}

    directions = extract_capability_direction(source_acts, base_acts)

    assert set(directions.keys()) == set(layers)
    for i in layers:
        assert directions[i].shape == (hidden,)
        assert np.linalg.norm(directions[i]) == pytest.approx(1.0, abs=1e-5)


def test_save_load_direction(tmp_path):
    rng = np.random.default_rng(1)
    directions = {
        0: rng.standard_normal(32).astype(np.float32),
        1: rng.standard_normal(32).astype(np.float32),
    }
    metadata = {
        "source_model": "src-model",
        "base_model": "base-model",
        "capability": "math",
        "hidden_dim": 32,
        "layers": [0, 1],
        "timestamp": "2026-04-12T00:00:00Z",
    }

    out_path = tmp_path / "test.pt"
    save_direction(directions, out_path, metadata)

    loaded_dirs, loaded_meta = load_direction(out_path)

    assert set(loaded_dirs.keys()) == set(directions.keys())
    for i in directions:
        np.testing.assert_allclose(loaded_dirs[i], directions[i])
    assert loaded_meta == metadata


def test_exact_match():
    assert exact_match("42", "42") is True
    assert exact_match("42 ", "42") is True
    assert exact_match("41", "42") is False
    assert exact_match("ANSWER", "answer") is True
