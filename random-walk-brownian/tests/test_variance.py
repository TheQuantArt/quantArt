"""Monte Carlo checks for mean ≈ 0 and variance ≈ t."""

from __future__ import annotations

import numpy as np
import pytest

from rwbm.statistics import clt_samples, empirical_mean_var, variance_along_time
from rwbm.simulation import simulate_many_scaled


def test_empirical_variance_near_theory_at_terminal():
    n, T, n_paths = 200, 1.0, 3000
    t, emp_mean, emp_var, theory_var = variance_along_time(
        n=n, n_paths=n_paths, T=T, seed=123, n_eval=30
    )
    # Terminal
    assert abs(emp_mean[-1]) < 0.08
    assert emp_var[-1] == pytest.approx(theory_var[-1], rel=0.12, abs=0.08)
    assert theory_var[-1] == pytest.approx(T, abs=1.0 / n + 1e-12)


def test_grid_variance_exact_in_population():
    """
    With huge samples the sample variance should approach k/n.
    Here we check the *population* construction: each path's second moment
    structure via many paths at a grid time.
    """
    n, t, n_paths = 100, 0.5, 5000
    samples = clt_samples(n=n, t=t, n_paths=n_paths, seed=99)
    mean, var = empirical_mean_var(samples)
    assert mean == pytest.approx(0.0, abs=0.06)
    assert var == pytest.approx(t, rel=0.1, abs=0.05)
    assert np.floor(n * t) / n == pytest.approx(t)


def test_unscaled_variance_grows_with_n():
    """Without 1/sqrt(n), terminal variance of M_{⌊nT⌋} equals ⌊nT⌋ ≈ nT."""
    T = 1.0
    seed = 5
    vars_unscaled = []
    for n in (50, 200):
        total = int(np.floor(n * T))
        # Simulate unscaled terminals via scaled * sqrt(n)
        _, paths = simulate_many_scaled(n=n, n_paths=2000, T=T, seed=seed, times=np.array([T]))
        terminals_scaled = paths[:, 0]
        terminals_unscaled = terminals_scaled * np.sqrt(n)
        vars_unscaled.append(float(np.var(terminals_unscaled, ddof=1)))
    assert vars_unscaled[1] > vars_unscaled[0] * 2.5  # roughly 4× when n×4


def test_scaled_variance_stable_across_n():
    """Scaled terminal variance should stay near T for different n."""
    T = 1.0
    estimates = []
    for n in (50, 200, 500):
        samples = clt_samples(n=n, t=T, n_paths=4000, seed=17)
        _, var = empirical_mean_var(samples)
        estimates.append(var)
        assert var == pytest.approx(T, rel=0.12, abs=0.08)
    # All estimates in the same ballpark
    assert max(estimates) - min(estimates) < 0.25
