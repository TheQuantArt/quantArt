"""Reproducibility and seed isolation tests."""

from __future__ import annotations

import numpy as np
import pytest

from rwbm.animation import AnimationConfig, expand_n_sequence, preview_frame_png, style_for_theme
from rwbm.simulation import (
    make_shared_increments,
    simulate_random_walk,
    simulate_scaled_walk,
    simulate_scaled_walk_coupled,
)
from rwbm.statistics import clt_samples


def test_random_walk_seed_reproducible():
    a = simulate_random_walk(500, seed=42)
    b = simulate_random_walk(500, seed=42)
    c = simulate_random_walk(500, seed=43)
    np.testing.assert_array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)


def test_scaled_walk_seed_reproducible():
    a = simulate_scaled_walk(n=250, T=1.0, seed=7)
    b = simulate_scaled_walk(n=250, T=1.0, seed=7)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.t, b.t)


def test_coupled_deterministic_given_increments():
    shared = make_shared_increments(1000, T=1.0, seed=1)
    a = simulate_scaled_walk_coupled(100, shared, T=1.0)
    b = simulate_scaled_walk_coupled(100, shared, T=1.0)
    np.testing.assert_array_equal(a.values, b.values)


def test_clt_samples_reproducible():
    a = clt_samples(n=80, t=0.5, n_paths=500, seed=3)
    b = clt_samples(n=80, t=0.5, n_paths=500, seed=3)
    np.testing.assert_array_equal(a, b)


def test_presentation_resample_does_not_change_seed_path():
    """Same increments ⇒ same grid path; resampling is a view, not new noise."""
    walk_grid = simulate_scaled_walk(n=100, T=1.0, seed=9)
    walk_hi = simulate_scaled_walk(n=100, T=1.0, seed=9, n_plot_points=1000)
    # Interpolating the grid path onto a fine mesh should match direct resample
    interp = np.interp(walk_hi.t, walk_grid.t, walk_grid.values)
    np.testing.assert_allclose(walk_hi.values, interp, atol=1e-12)


def test_expand_n_sequence_smooth():
    seq = expand_n_sequence([10, 1000], smooth=True)
    assert seq[0] == 10
    assert seq[-1] == 1000
    assert len(seq) > 2


def test_preview_frame_png_bytes():
    cfg = AnimationConfig(
        n_sequence=[10, 50],
        T=1.0,
        seed=0,
        style=style_for_theme("dark", figsize=(6.4, 3.6), dpi=50),
        hold_frames=1,
        smooth_progression=False,
    )
    png = preview_frame_png(cfg, n=50)
    assert isinstance(png, (bytes, bytearray))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
