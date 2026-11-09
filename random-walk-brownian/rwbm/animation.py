"""
Matplotlib Animation Studio for high-quality YouTube exports.

Prefer a nested (shared-increment) coupling so successive frames refine the
*same* underlying noise rather than flashing unrelated paths.
"""

from __future__ import annotations

import io
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib as mpl

# Non-interactive backend for reproducible exports / headless CI
mpl.use("Agg", force=False)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation as mpl_animation

from .simulation import (
    make_shared_increments,
    simulate_scaled_walk,
    simulate_scaled_walk_coupled,
)


@dataclass
class AnimationStyle:
    theme: str = "dark"  # "dark" | "light"
    figsize: Tuple[float, float] = (19.2, 10.8)  # inches → 1920×1080 at 100 dpi
    dpi: int = 100
    linewidth: float = 2.5
    line_color: str = "#E94F37"
    bg_color: str = "#0E1117"
    axes_color: str = "#FAFAFA"
    grid: bool = False
    show_legend: bool = False
    show_title: bool = True
    show_n_label: bool = True
    show_scaling_label: bool = True
    title: str = r"From random walk to Brownian motion: $W_n(t)=M_{\lfloor nt\rfloor}/\sqrt{n}$"
    xlabel: str = r"$t$"
    ylabel: str = r"$W_n(t)$"
    font_size: float = 18
    label_size: float = 16
    minimal: bool = True


def style_for_theme(theme: str, **overrides) -> AnimationStyle:
    theme = theme.lower()
    if theme == "light":
        base = AnimationStyle(
            theme="light",
            line_color="#C73E1D",
            bg_color="#FFFFFF",
            axes_color="#1A1A1A",
        )
    else:
        base = AnimationStyle(
            theme="dark",
            line_color="#FF6B4A",
            bg_color="#0E1117",
            axes_color="#F5F5F5",
        )
    for k, v in overrides.items():
        if hasattr(base, k):
            setattr(base, k, v)
    return base


@dataclass
class AnimationConfig:
    n_sequence: Sequence[int] = field(
        default_factory=lambda: [10, 50, 100, 500, 1000, 5000, 10000]
    )
    T: float = 1.0
    seed: int = 42
    y_range: Tuple[float, float] = (-3.0, 3.0)
    fps: int = 2
    hold_frames: int = 2  # pause frames at each n
    n_plot_points: int = 2000
    coupled: bool = True
    style: AnimationStyle = field(default_factory=lambda: style_for_theme("dark"))
    # When True, insert intermediate n values for smoother visual progression
    smooth_progression: bool = False


def expand_n_sequence(n_sequence: Sequence[int], smooth: bool) -> List[int]:
    """Optionally densify the n ladder for smoother animations."""
    seq = sorted({int(n) for n in n_sequence if int(n) >= 1})
    if not smooth or len(seq) < 2:
        return seq
    out: List[int] = []
    for a, b in zip(seq[:-1], seq[1:]):
        out.append(a)
        # Geometric midpoints (unique ints)
        ratio = (b / a) ** (1 / 3)
        mids = [int(round(a * ratio)), int(round(a * ratio * ratio))]
        for m in mids:
            if a < m < b and m not in out:
                out.append(m)
    out.append(seq[-1])
    # Deduplicate preserving order
    seen = set()
    unique = []
    for n in out:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _apply_style(ax: plt.Axes, fig: plt.Figure, style: AnimationStyle) -> None:
    fig.patch.set_facecolor(style.bg_color)
    ax.set_facecolor(style.bg_color)
    for spine in ax.spines.values():
        spine.set_color(style.axes_color)
        spine.set_linewidth(1.2)
    ax.tick_params(colors=style.axes_color, labelsize=style.label_size)
    ax.xaxis.label.set_color(style.axes_color)
    ax.yaxis.label.set_color(style.axes_color)
    ax.title.set_color(style.axes_color)
    if style.grid:
        ax.grid(True, color=style.axes_color, alpha=0.15, linewidth=0.8)
    else:
        ax.grid(False)
    if style.minimal:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def build_animation(config: AnimationConfig):
    """
    Construct a Matplotlib FuncAnimation of W_n as n increases.

    Returns (fig, anim, n_used).
    """
    style = config.style
    n_seq = expand_n_sequence(config.n_sequence, config.smooth_progression)
    if not n_seq:
        raise ValueError("n_sequence is empty")

    n_max = max(n_seq)
    if config.coupled:
        shared = make_shared_increments(n_max, T=config.T, seed=config.seed)
    else:
        shared = None

    # Precompute all paths for smooth playback
    frames_data = []
    for n in n_seq:
        if config.coupled:
            walk = simulate_scaled_walk_coupled(
                n=n,
                shared_increments=shared,
                T=config.T,
                n_plot_points=config.n_plot_points,
                seed=config.seed,
            )
        else:
            # Independent path per n — still seeded reproducibly via offset
            walk = simulate_scaled_walk(
                n=n,
                T=config.T,
                seed=config.seed + n,
                n_plot_points=config.n_plot_points,
            )
        for _ in range(max(1, config.hold_frames)):
            frames_data.append((n, walk.t, walk.values))

    mpl.rcParams.update(
        {
            "font.size": style.font_size,
            "axes.labelsize": style.font_size,
            "axes.titlesize": style.font_size + 2,
            "figure.dpi": style.dpi,
        }
    )

    fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)
    _apply_style(ax, fig, style)

    (line,) = ax.plot(
        [],
        [],
        color=style.line_color,
        lw=style.linewidth,
        solid_capstyle="round",
    )
    ax.set_xlim(0.0, config.T)
    ax.set_ylim(*config.y_range)
    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    if style.show_title:
        ax.set_title(style.title, pad=16)

    info_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=style.axes_color,
        fontsize=style.font_size,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=style.bg_color,
            edgecolor=style.axes_color,
            alpha=0.75,
        ),
    )
    if not (style.show_n_label or style.show_scaling_label):
        info_text.set_visible(False)

    def init():
        line.set_data([], [])
        info_text.set_text("")
        return line, info_text

    def update(frame_idx: int):
        n, t, values = frames_data[frame_idx]
        line.set_data(t, values)
        parts = []
        if style.show_n_label:
            parts.append(f"n = {n:,}")
        if style.show_scaling_label:
            parts.append(f"1/√n = {1/np.sqrt(n):.5f}")
        info_text.set_text("\n".join(parts))
        return line, info_text

    # blit=False is more reliable across backends (Agg, macOS, headless CI)
    anim = mpl_animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames_data),
        interval=1000 / max(config.fps, 1),
        blit=False,
        repeat=True,
    )
    return fig, anim, n_seq


def export_animation(
    config: AnimationConfig,
    output_path: str,
    fmt: str = "mp4",
) -> Path:
    """
    Export animation to MP4 (FFmpeg) or GIF (Pillow / ImageMagick).

    Raises RuntimeError with install hints if the writer is unavailable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, anim, _ = build_animation(config)

    fmt = fmt.lower().lstrip(".")
    try:
        if fmt == "mp4":
            if not ffmpeg_available():
                raise RuntimeError(
                    "FFmpeg introuvable. Sur macOS : brew install ffmpeg\n"
                    "Puis relancez l'export MP4."
                )
            writer = mpl_animation.FFMpegWriter(
                fps=config.fps,
                metadata={"title": "Random Walk to Brownian Motion", "artist": "rwbm"},
                bitrate=2400,
            )
            anim.save(str(output_path), writer=writer, dpi=config.style.dpi)
        elif fmt == "gif":
            writer = mpl_animation.PillowWriter(fps=config.fps)
            anim.save(str(output_path), writer=writer, dpi=config.style.dpi)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
    finally:
        plt.close(fig)

    return output_path


def preview_frame_png(
    config: AnimationConfig,
    n: Optional[int] = None,
) -> bytes:
    """Render a single still frame as PNG bytes (for Streamlit preview)."""
    style = config.style
    n_seq = expand_n_sequence(config.n_sequence, config.smooth_progression)
    n_use = int(n if n is not None else n_seq[-1])
    n_max = max(max(n_seq), n_use)

    if config.coupled:
        shared = make_shared_increments(n_max, T=config.T, seed=config.seed)
        walk = simulate_scaled_walk_coupled(
            n=n_use,
            shared_increments=shared,
            T=config.T,
            n_plot_points=config.n_plot_points,
            seed=config.seed,
        )
    else:
        walk = simulate_scaled_walk(
            n=n_use,
            T=config.T,
            seed=config.seed + n_use,
            n_plot_points=config.n_plot_points,
        )

    fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)
    _apply_style(ax, fig, style)
    ax.plot(walk.t, walk.values, color=style.line_color, lw=style.linewidth)
    ax.set_xlim(0.0, config.T)
    ax.set_ylim(*config.y_range)
    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    if style.show_title:
        ax.set_title(style.title, pad=16)
    if style.show_n_label or style.show_scaling_label:
        parts = []
        if style.show_n_label:
            parts.append(f"n = {n_use:,}")
        if style.show_scaling_label:
            parts.append(f"1/√n = {1/np.sqrt(n_use):.5f}")
        ax.text(
            0.02,
            0.96,
            "\n".join(parts),
            transform=ax.transAxes,
            va="top",
            ha="left",
            color=style.axes_color,
            fontsize=style.font_size,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=style.bg_color,
                edgecolor=style.axes_color,
                alpha=0.75,
            ),
        )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=style.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def default_export_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "rwbm_exports"
    d.mkdir(parents=True, exist_ok=True)
    return d
