"""Plotly figure builders for the Streamlit UI."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from .statistics import gaussian_density


# Calm educational palette (works on light and dark Streamlit themes)
COLOR_WALK = "#2E86AB"
COLOR_SCALED = "#E94F37"
COLOR_THEORY = "#F6AE2D"
COLOR_UNSCALED = "#1B998B"
COLOR_HIST = "#6C63FF"
COLOR_GRID = "rgba(128,128,128,0.25)"


def _base_layout(
    title: str,
    x_title: str,
    y_title: str,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    height: int = 480,
    show_legend: bool = True,
) -> dict:
    layout = dict(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(
            title=x_title,
            range=list(x_range) if x_range else None,
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=True,
        ),
        yaxis=dict(
            title=y_title,
            range=list(y_range) if y_range else None,
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=True,
        ),
        height=height,
        margin=dict(l=60, r=30, t=60, b=50),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
    )
    return layout


def fig_random_walk(
    t: np.ndarray,
    values: np.ndarray,
    show_markers: bool = False,
    title: str = "Symmetric random walk",
    y_range: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    mode = "lines+markers" if show_markers else "lines"
    fig = go.Figure(
        data=[
            go.Scatter(
                x=t,
                y=values,
                mode=mode,
                name="M_k",
                line=dict(color=COLOR_WALK, width=2),
                marker=dict(size=5),
            )
        ]
    )
    fig.update_layout(
        **_base_layout(
            title=title,
            x_title="step k",
            y_title="position",
            x_range=(float(t[0]), float(t[-1])),
            y_range=y_range,
        )
    )
    return fig


def fig_scaled_walk(
    t: np.ndarray,
    values: np.ndarray,
    n: int,
    T: float,
    y_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_markers: bool = True,
    color: str = COLOR_SCALED,
) -> go.Figure:
    """Scaled walk with fixed axes — markers highlight the discrete grid."""
    # Show markers only when the grid is coarse enough to remain readable
    n_grid = max(int(np.floor(n * T)), 1)
    use_markers = show_markers and n_grid <= 200
    mode = "lines+markers" if use_markers else "lines"
    marker_size = 6 if n_grid <= 50 else 4

    if title is None:
        title = f"Scaled walk W<sub>n</sub>(t)  —  n = {n:,}  ·  1/√n = {1/np.sqrt(n):.4f}"

    fig = go.Figure(
        data=[
            go.Scatter(
                x=t,
                y=values,
                mode=mode,
                name=f"W_{n}(t)",
                line=dict(color=color, width=2),
                marker=dict(size=marker_size, color=color),
            )
        ]
    )
    fig.update_layout(
        **_base_layout(
            title=title,
            x_title="t",
            y_title="W_n(t)",
            x_range=(0.0, T),
            y_range=y_range,
        )
    )
    # Annotation with scaling factor
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.98,
        text=f"n = {n:,}<br>1/√n = {1/np.sqrt(n):.5f}",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font=dict(size=13),
    )
    return fig


def fig_scaled_vs_unscaled(
    t: np.ndarray,
    unscaled: np.ndarray,
    scaled: np.ndarray,
    n: int,
    T: float,
    y_range_unscaled: Optional[Tuple[float, float]] = None,
    y_range_scaled: Optional[Tuple[float, float]] = None,
    lock_axes: bool = True,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Unscaled  (step size 1)  —  n = {n:,}",
            f"Scaled  (step size 1/√n ≈ {1/np.sqrt(n):.4f})",
        ),
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=unscaled,
            mode="lines",
            name="unscaled",
            line=dict(color=COLOR_UNSCALED, width=2),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=scaled,
            mode="lines",
            name="scaled",
            line=dict(color=COLOR_SCALED, width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="t", range=[0, T], row=1, col=1)
    fig.update_xaxes(title_text="t", range=[0, T], row=1, col=2)
    fig.update_yaxes(title_text="M_{⌊nt⌋}", row=1, col=1)
    fig.update_yaxes(title_text="W_n(t)", row=1, col=2)

    if lock_axes:
        if y_range_unscaled:
            fig.update_yaxes(range=list(y_range_unscaled), row=1, col=1)
        if y_range_scaled:
            fig.update_yaxes(range=list(y_range_scaled), row=1, col=2)

    fig.update_layout(
        height=480,
        margin=dict(l=50, r=30, t=60, b=50),
        template="plotly_white",
    )
    return fig


def fig_variance(
    t: np.ndarray,
    emp_var: np.ndarray,
    theory_var: np.ndarray,
    emp_mean: Optional[np.ndarray] = None,
    T: float = 1.0,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Empirical mean Ê[W_n(t)] ≈ 0", "Variance: empirical vs theoretical"),
        vertical_spacing=0.12,
        row_heights=[0.35, 0.65],
    )
    if emp_mean is not None:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=emp_mean,
                mode="lines",
                name="empirical mean",
                line=dict(color=COLOR_WALK, width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=t,
            y=emp_var,
            mode="lines",
            name="empirical Var",
            line=dict(color=COLOR_SCALED, width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=theory_var,
            mode="lines",
            name="theory ⌊nt⌋/n",
            line=dict(color=COLOR_THEORY, width=2, dash="dash"),
        ),
        row=2,
        col=1,
    )
    # Reference line Var(B_t) = t
    fig.add_trace(
        go.Scatter(
            x=np.array([0.0, T]),
            y=np.array([0.0, T]),
            mode="lines",
            name="Var(B_t) = t",
            line=dict(color="black", width=1, dash="dot"),
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="t", range=[0, T], row=2, col=1)
    fig.update_yaxes(title_text="mean", row=1, col=1)
    fig.update_yaxes(title_text="variance", row=2, col=1)
    fig.update_layout(
        height=560,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def fig_clt_histogram(
    samples: np.ndarray,
    t: float,
    n: int,
    n_bins: int = 40,
) -> go.Figure:
    var_theory = t  # limiting; exact on grid is floor(n t)/n
    exact_var = np.floor(n * t) / n
    x_min = float(samples.min()) if samples.size else -3 * np.sqrt(max(t, 1e-9))
    x_max = float(samples.max()) if samples.size else 3 * np.sqrt(max(t, 1e-9))
    pad = 0.1 * (x_max - x_min + 1e-9)
    x_grid = np.linspace(x_min - pad, x_max + pad, 400)
    density = gaussian_density(x_grid, 0.0, exact_var if exact_var > 0 else var_theory)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=samples,
            histnorm="probability density",
            nbinsx=n_bins,
            name="empirical",
            marker_color=COLOR_HIST,
            opacity=0.65,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=density,
            mode="lines",
            name=f"N(0, {exact_var:.4f})",
            line=dict(color=COLOR_THEORY, width=3),
        )
    )
    emp_mean = float(samples.mean()) if samples.size else 0.0
    emp_var = float(samples.var(ddof=1)) if samples.size > 1 else 0.0
    fig.update_layout(
        **_base_layout(
            title=(
                f"Law of W<sub>n</sub>(t) at t = {t}  ·  n = {n:,}  ·  "
                f"mean̂ = {emp_mean:.3f}  ·  var̂ = {emp_var:.3f}  (theory {exact_var:.3f})"
            ),
            x_title="W_n(t)",
            y_title="density",
            height=480,
        )
    )
    return fig


def suggest_y_range(
    values_list: Sequence[np.ndarray],
    pad_ratio: float = 0.15,
    min_span: float = 2.0,
) -> Tuple[float, float]:
    """Suggest symmetric-ish fixed y-limits covering all provided paths."""
    lo = min(float(np.min(v)) for v in values_list)
    hi = max(float(np.max(v)) for v in values_list)
    span = max(hi - lo, min_span)
    mid = 0.5 * (hi + lo)
    half = 0.5 * span * (1.0 + pad_ratio)
    # Prefer including 0 when reasonable
    if lo <= 0 <= hi:
        half = max(abs(lo), abs(hi)) * (1.0 + pad_ratio)
        return (-half, half)
    return (mid - half, mid + half)


def ks_normality_pvalue(samples: np.ndarray, var: float) -> float:
    """Kolmogorov–Smirnov p-value against N(0, var)."""
    if samples.size < 5 or var <= 0:
        return float("nan")
    std = np.sqrt(var)
    standardized = samples / std
    return float(stats.kstest(standardized, "norm").pvalue)
