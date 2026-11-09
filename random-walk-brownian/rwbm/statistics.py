"""Statistical helpers for variance experiments and CLT demonstrations."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import stats

from .simulation import simulate_many_scaled


def empirical_mean_var(samples: np.ndarray) -> Tuple[float, float]:
    """Return sample mean and unbiased sample variance (ddof=1)."""
    samples = np.asarray(samples, dtype=np.float64).ravel()
    if samples.size < 2:
        mean = float(samples.mean()) if samples.size else 0.0
        return mean, 0.0
    return float(samples.mean()), float(samples.var(ddof=1))


def variance_along_time(
    n: int,
    n_paths: int,
    T: float = 1.0,
    seed: Optional[int] = None,
    n_eval: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Empirical mean and variance of W_n(t) along a time grid.

    At grid times t = k/n, Var(W_n(t)) = t exactly in theory
    (because Var(M_k) = k and Var(M_k/√n) = k/n = t).

    Returns
    -------
    t, emp_mean, emp_var, theory_var
    """
    # Prefer grid-aligned times when possible, plus a uniform overlay
    total = int(np.floor(n * T))
    if total <= n_eval:
        times = np.arange(total + 1, dtype=np.float64) / n
        if times[-1] < T - 1e-15:
            times = np.append(times, T)
    else:
        # Mix of uniform points, snapped toward grid for interpretability
        times = np.linspace(0.0, T, n_eval)
        # Snap interior points to nearest grid time for exact theory line match
        grid = np.arange(total + 1, dtype=np.float64) / n
        snapped = np.empty_like(times)
        for i, ti in enumerate(times):
            j = int(np.argmin(np.abs(grid - ti)))
            snapped[i] = grid[j]
        snapped[0] = 0.0
        # Ensure T present
        if snapped[-1] != T and abs(snapped[-1] - T) > 1e-12:
            # last theoretical grid point <= T
            snapped[-1] = grid[-1] if grid[-1] <= T else T
        times = np.unique(snapped)

    t, paths = simulate_many_scaled(n=n, n_paths=n_paths, T=T, seed=seed, times=times)
    emp_mean = paths.mean(axis=0)
    emp_var = paths.var(axis=0, ddof=1) if n_paths > 1 else np.zeros_like(t)
    # Theoretical variance for the interpolated process equals
    # floor(n t)/n  which equals t on the grid; elsewhere ≤ t.
    theory_var = np.floor(n * t) / n
    return t, emp_mean, emp_var, theory_var


def clt_samples(
    n: int,
    t: float,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample W_n(t) for many independent paths (terminal / fixed-time law).

    Prefer grid-aligned t = k/n so that Var(W_n(t)) = t exactly.
    """
    if t < 0:
        raise ValueError("t must be non-negative")
    times = np.array([t], dtype=np.float64)
    # Need T >= t
    T = max(t, 1e-12)
    _, paths = simulate_many_scaled(
        n=n, n_paths=n_paths, T=T, seed=seed, times=times
    )
    return paths[:, 0]


def gaussian_density(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """N(mean, var) density; returns zeros if var <= 0."""
    if var <= 0:
        return np.zeros_like(x, dtype=np.float64)
    return stats.norm.pdf(x, loc=mean, scale=np.sqrt(var))


def is_grid_aligned(t: float, n: int, tol: float = 1e-10) -> bool:
    """True if t is (approximately) an integer multiple of 1/n."""
    if t == 0:
        return True
    return abs(n * t - round(n * t)) < tol
