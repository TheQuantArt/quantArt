"""
Core random-walk and scaled-walk simulations.

Symmetric random walk
---------------------
    X_i ∈ {+1, −1} with P(X_i = ±1) = 1/2
    M_k = X_1 + … + X_k

Scaled process (Donsker scaling)
--------------------------------
    W_n(t) = M_{⌊n t⌋} / √n ,   t ∈ [0, T]

Linear interpolation is used between the grid points k/n.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class WalkResult:
    """Container for a (possibly scaled) walk path."""

    t: np.ndarray  # time grid
    values: np.ndarray  # path values
    n: int  # steps per unit time (1 for unscaled step-indexed walk)
    T: float
    seed: Optional[int]
    increments: Optional[np.ndarray] = None  # underlying ±1 sequence used


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_increments(
    n_steps: int,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate n_steps i.i.d. symmetric ±1 increments."""
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if rng is None:
        rng = _rng(seed)
    bits = rng.integers(0, 2, size=n_steps, dtype=np.int8)
    return (2 * bits - 1).astype(np.float64)


def partial_sums(increments: np.ndarray) -> np.ndarray:
    """Return M_0=0, M_1, …, M_k from increments."""
    if increments.size == 0:
        return np.array([0.0], dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(increments, dtype=np.float64)])


def simulate_random_walk(
    n_steps: int,
    seed: Optional[int] = None,
    step_size: float = 1.0,
    increments: Optional[np.ndarray] = None,
) -> WalkResult:
    """
    Simulate an unscaled symmetric random walk of ``n_steps`` steps.

    The returned time axis is the step index 0, 1, …, n_steps (not calendar time).
    Positions are step_size * M_k.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if increments is None:
        increments = generate_increments(n_steps, seed=seed)
    elif len(increments) < n_steps:
        raise ValueError("Not enough increments provided")
    else:
        increments = np.asarray(increments[:n_steps], dtype=np.float64)

    M = partial_sums(increments)
    t = np.arange(n_steps + 1, dtype=np.float64)
    return WalkResult(
        t=t,
        values=step_size * M,
        n=1,
        T=float(n_steps),
        seed=seed,
        increments=increments,
    )


def _interp_scaled_from_increments(
    increments: np.ndarray,
    n: int,
    T: float,
    n_plot_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build W_n(t) = M_{⌊n t⌋} / √n on [0, T] with linear interpolation.

    If ``n_plot_points`` is None, return the exact grid {0, 1/n, …, ⌊nT⌋/n}
    plus the endpoint T if needed. Otherwise resample onto a uniform plot grid
    via linear interpolation (useful for fixed-axis animations).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if T <= 0:
        raise ValueError("T must be positive")

    total_needed = int(np.floor(n * T))
    if len(increments) < total_needed:
        raise ValueError(
            f"Need {total_needed} increments for n={n}, T={T}, got {len(increments)}"
        )

    M = partial_sums(increments[:total_needed])
    scale = 1.0 / np.sqrt(n)
    # Grid times k/n for k = 0..total_needed, and ensure T is included
    grid_t = np.arange(total_needed + 1, dtype=np.float64) / n
    grid_w = scale * M

    if grid_t[-1] < T - 1e-15:
        # Hold the last value until T (no more increments past ⌊nT⌋)
        grid_t = np.append(grid_t, T)
        grid_w = np.append(grid_w, grid_w[-1])
    elif grid_t[-1] > T:
        # Rare floating edge: clip
        grid_t[-1] = T

    if n_plot_points is None:
        return grid_t, grid_w

    t_plot = np.linspace(0.0, T, n_plot_points)
    w_plot = np.interp(t_plot, grid_t, grid_w)
    return t_plot, w_plot


def simulate_scaled_walk(
    n: int,
    T: float = 1.0,
    seed: Optional[int] = None,
    n_plot_points: Optional[int] = None,
    increments: Optional[np.ndarray] = None,
) -> WalkResult:
    """
    Simulate one scaled random walk W_n on [0, T].

    Parameters
    ----------
    n :
        Number of steps per unit time.
    T :
        Terminal time.
    seed :
        RNG seed (ignored if ``increments`` is provided).
    n_plot_points :
        If set, resample the path onto this many uniform points in [0, T].
    increments :
        Optional pre-generated ±1 increments (length at least ⌊n T⌋).
    """
    total_needed = int(np.floor(n * T))
    if increments is None:
        increments = generate_increments(total_needed, seed=seed)
    t, values = _interp_scaled_from_increments(
        increments, n=n, T=T, n_plot_points=n_plot_points
    )
    return WalkResult(
        t=t,
        values=values,
        n=n,
        T=T,
        seed=seed,
        increments=np.asarray(increments[:total_needed], dtype=np.float64),
    )


def max_increments_needed(n_values: Sequence[int], T: float) -> int:
    """Total increments required to cover all resolutions in ``n_values``."""
    return int(np.floor(max(n_values) * T))


def simulate_scaled_walk_coupled(
    n: int,
    shared_increments: np.ndarray,
    T: float = 1.0,
    n_plot_points: Optional[int] = None,
    seed: Optional[int] = None,
) -> WalkResult:
    """
    Build W_n from a shared increment sequence (nested coupling).

    Using the same (X_i) for every n yields a mathematically natural coupling:
    coarser and finer approximations are driven by the same underlying noise.
    Convergence of the laws still holds; pathwise visual coherence improves
    dramatically for animations compared with independent draws.
    """
    return simulate_scaled_walk(
        n=n,
        T=T,
        seed=seed,
        n_plot_points=n_plot_points,
        increments=shared_increments,
    )


def make_shared_increments(
    n_max: int,
    T: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate enough ±1 increments for resolution ``n_max`` on [0, T]."""
    return generate_increments(int(np.floor(n_max * T)), seed=seed)


def simulate_many_scaled(
    n: int,
    n_paths: int,
    T: float = 1.0,
    seed: Optional[int] = None,
    times: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate many independent scaled walks efficiently (vectorized).

    Returns
    -------
    t : ndarray, shape (n_times,)
        Evaluation times (default: full grid 0, 1/n, …, ⌊nT⌋/n, and T).
    paths : ndarray, shape (n_paths, n_times)
        Path values W_n(t).
    """
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    total_needed = int(np.floor(n * T))
    rng = _rng(seed)

    # (n_paths, total_needed) of ±1
    bits = rng.integers(0, 2, size=(n_paths, total_needed), dtype=np.int8)
    increments = (2 * bits - 1).astype(np.float64)
    M = np.cumsum(increments, axis=1, dtype=np.float64)
    # prepend M_0 = 0
    M = np.concatenate([np.zeros((n_paths, 1)), M], axis=1)
    scale = 1.0 / np.sqrt(n)

    grid_t = np.arange(total_needed + 1, dtype=np.float64) / n
    grid_w = scale * M  # (n_paths, total_needed+1)

    if times is None:
        if grid_t[-1] < T - 1e-15:
            t = np.append(grid_t, T)
            # hold last value
            paths = np.concatenate([grid_w, grid_w[:, -1:]], axis=1)
        else:
            t = grid_t.copy()
            if t[-1] > T:
                t[-1] = T
            paths = grid_w
        return t, paths

    times = np.asarray(times, dtype=np.float64)
    # Vectorized interpolation per path
    paths = np.empty((n_paths, len(times)), dtype=np.float64)
    g_t = grid_t
    if g_t[-1] < T - 1e-15:
        g_t = np.append(g_t, T)
        grid_w = np.concatenate([grid_w, grid_w[:, -1:]], axis=1)
    for i in range(n_paths):
        paths[i] = np.interp(times, g_t, grid_w[i])
    return times, paths


def unscaled_vs_scaled(
    n: int,
    T: float = 1.0,
    seed: Optional[int] = None,
    increments: Optional[np.ndarray] = None,
) -> Tuple[WalkResult, WalkResult]:
    """
    Return the unscaled walk (step size 1 on calendar time k/n) and the
    properly scaled walk (step size 1/√n) from the same increments.
    """
    total = int(np.floor(n * T))
    if increments is None:
        increments = generate_increments(total, seed=seed)
    else:
        increments = np.asarray(increments[:total], dtype=np.float64)

    M = partial_sums(increments)
    t = np.arange(total + 1, dtype=np.float64) / n
    if t[-1] < T - 1e-15:
        t = np.append(t, T)
        M = np.append(M, M[-1])

    unscaled = WalkResult(
        t=t,
        values=M.copy(),
        n=n,
        T=T,
        seed=seed,
        increments=increments,
    )
    scaled = WalkResult(
        t=t,
        values=M / np.sqrt(n),
        n=n,
        T=T,
        seed=seed,
        increments=increments,
    )
    return unscaled, scaled


def n_grid_default() -> Tuple[int, ...]:
    return (10, 50, 100, 500, 1000, 5000, 10000)


def parse_n_value(value: Union[int, float, str]) -> int:
    """Parse UI n values that may contain thousand separators."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).replace(",", "").replace(" ", "").replace("_", "")
    return int(float(text))
