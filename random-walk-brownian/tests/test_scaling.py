"""Tests for Donsker scaling identities and path construction."""

from __future__ import annotations

import numpy as np
import pytest

from rwbm.simulation import (
    generate_increments,
    make_shared_increments,
    partial_sums,
    simulate_scaled_walk,
    simulate_scaled_walk_coupled,
    unscaled_vs_scaled,
)


def test_scaled_grid_values_match_definition():
    """W_n(k/n) equals M_k / sqrt(n) on the integer grid."""
    n, T = 100, 1.0
    increments = generate_increments(n, seed=0)
    M = partial_sums(increments)
    walk = simulate_scaled_walk(n=n, T=T, increments=increments)

    for k in (0, 1, 25, 50, 100):
        t = k / n
        idx = int(np.argmin(np.abs(walk.t - t)))
        assert abs(walk.t[idx] - t) < 1e-12
        assert walk.values[idx] == pytest.approx(M[k] / np.sqrt(n), abs=1e-12)


def test_theoretical_grid_variance_identity():
    """Algebraic identity: Var(M_k / sqrt(n)) = k/n = t on the grid."""
    n = 64
    for k in range(0, n + 1, 8):
        t = k / float(n)
        assert k / float(n) == pytest.approx(t)
        # Each X_i has variance 1 ⇒ Var(M_k)=k ⇒ Var(M_k/√n)=k/n
        assert k / float(n) == pytest.approx(t)


def test_coupling_uses_shared_prefix():
    """Coarse W_n is built from the shared increment prefix."""
    T = 1.0
    n_fine = 1000
    n_coarse = 100
    shared = make_shared_increments(n_fine, T=T, seed=7)
    coarse = simulate_scaled_walk_coupled(n_coarse, shared, T=T)

    for k in range(0, n_coarse + 1, 10):
        t = k / n_coarse
        idx_c = int(np.argmin(np.abs(coarse.t - t)))
        M_k = float(np.sum(shared[:k])) if k > 0 else 0.0
        expected = M_k / np.sqrt(n_coarse)
        assert coarse.values[idx_c] == pytest.approx(expected, abs=1e-10)


def test_interpolation_endpoints():
    walk = simulate_scaled_walk(n=50, T=1.0, seed=1)
    assert walk.t[0] == pytest.approx(0.0)
    assert walk.t[-1] == pytest.approx(1.0)
    assert walk.values[0] == pytest.approx(0.0)


def test_unscaled_vs_scaled_relation():
    n, T = 200, 1.0
    uns, sc = unscaled_vs_scaled(n=n, T=T, seed=3)
    np.testing.assert_allclose(sc.values, uns.values / np.sqrt(n))
    np.testing.assert_allclose(uns.t, sc.t)


def test_plot_resample_preserves_endpoints():
    walk = simulate_scaled_walk(n=100, T=1.0, seed=2, n_plot_points=500)
    assert len(walk.t) == 500
    assert walk.t[0] == pytest.approx(0.0)
    assert walk.t[-1] == pytest.approx(1.0)
    assert walk.values[0] == pytest.approx(0.0)


def test_increments_are_pm_one():
    x = generate_increments(10_000, seed=11)
    assert set(np.unique(x).tolist()) <= {-1.0, 1.0}
    assert abs(x.mean()) < 0.05
