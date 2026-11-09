"""
From Random Walk to Brownian Motion
===================================
Interactive Streamlit application for educational visualizations
of Donsker's theorem and the CLT scaling of symmetric random walks.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import streamlit as st

from rwbm.animation import (
    AnimationConfig,
    AnimationStyle,
    default_export_dir,
    export_animation,
    ffmpeg_available,
    preview_frame_png,
    style_for_theme,
)
from rwbm.math_explanations import DISTINCTION_NOTE, MATH_MARKDOWN
from rwbm.plots import (
    fig_clt_histogram,
    fig_random_walk,
    fig_scaled_vs_unscaled,
    fig_scaled_walk,
    fig_variance,
    suggest_y_range,
)
from rwbm.simulation import (
    make_shared_increments,
    n_grid_default,
    simulate_random_walk,
    simulate_scaled_walk,
    simulate_scaled_walk_coupled,
    unscaled_vs_scaled,
)
from rwbm.statistics import (
    clt_samples,
    empirical_mean_var,
    is_grid_aligned,
    variance_along_time,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Random Walk → Brownian Motion",
    page_icon="∫",
    layout="wide",
    initial_sidebar_state="expanded",
)

N_CHOICES_WALK = [10, 100, 1_000, 10_000, 100_000]
N_CHOICES_SCALE = list(n_grid_default())


# ---------------------------------------------------------------------------
# Cached simulation layers (data only — presentation changes do not recompute)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_random_walk(n_steps: int, seed: int, step_size: float):
    walk = simulate_random_walk(n_steps=n_steps, seed=seed, step_size=step_size)
    return walk.t, walk.values


@st.cache_data(show_spinner=False)
def _cached_shared_increments(n_max: int, T: float, seed: int) -> np.ndarray:
    return make_shared_increments(n_max, T=T, seed=seed)


@st.cache_data(show_spinner=False)
def _cached_scaled(
    n: int,
    T: float,
    seed: int,
    coupled: bool,
    n_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if coupled:
        shared = _cached_shared_increments(n_max, T, seed)
        walk = simulate_scaled_walk_coupled(
            n=n, shared_increments=shared, T=T, seed=seed
        )
    else:
        walk = simulate_scaled_walk(n=n, T=T, seed=seed)
    return walk.t, walk.values


@st.cache_data(show_spinner=False)
def _cached_unscaled_scaled(n: int, T: float, seed: int):
    unscaled, scaled = unscaled_vs_scaled(n=n, T=T, seed=seed)
    return unscaled.t, unscaled.values, scaled.values


@st.cache_data(show_spinner=False)
def _cached_variance(n: int, n_paths: int, T: float, seed: int, n_eval: int):
    return variance_along_time(n=n, n_paths=n_paths, T=T, seed=seed, n_eval=n_eval)


@st.cache_data(show_spinner=False)
def _cached_clt(n: int, t: float, n_paths: int, seed: int) -> np.ndarray:
    return clt_samples(n=n, t=t, n_paths=n_paths, seed=seed)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "rw_seed": 42,
        "rw_nonce": 0,
        "scale_seed": 42,
        "scale_nonce": 0,
        "cmp_seed": 42,
        "cmp_nonce": 0,
        "cmp_anim_idx": 0,
        "cmp_anim_running": False,
        "var_seed": 42,
        "clt_seed": 42,
        "anim_seed": 42,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _effective_seed(base_key: str, nonce_key: str) -> int:
    return int(st.session_state[base_key]) + 1_000_003 * int(st.session_state[nonce_key])


def math_expander():
    with st.expander("Mathematical background", expanded=False):
        st.markdown(MATH_MARKDOWN)
        st.markdown(DISTINCTION_NOTE)


# ---------------------------------------------------------------------------
# Sidebar — global presentation defaults
# ---------------------------------------------------------------------------
def sidebar_globals() -> dict:
    st.sidebar.title("Controls")
    st.sidebar.caption("Parameters for filming & exploration")

    T = st.sidebar.number_input(
        "Horizon T",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Fixed time interval [0, T] for scaled processes.",
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Fixed y-axis (scaled plots)")
    y_min = st.sidebar.number_input("y min", value=-3.0, step=0.5)
    y_max = st.sidebar.number_input("y max", value=3.0, step=0.5)
    if y_min >= y_max:
        st.sidebar.error("y min must be < y max")
        y_min, y_max = -3.0, 3.0

    st.sidebar.markdown("---")
    show_math = st.sidebar.checkbox("Show math section on tabs", value=True)
    return {"T": float(T), "y_range": (float(y_min), float(y_max)), "show_math": show_math}


# ===========================================================================
# Tab 1 — Random Walk
# ===========================================================================
def tab_random_walk(show_math: bool):
    st.header("Symmetric random walk")
    st.latex(r"M_k = X_1 + \cdots + X_k,\quad X_i = \pm 1 \text{ with equal probability}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_steps = st.selectbox("Number of steps", N_CHOICES_WALK, index=2)
    with c2:
        seed = st.number_input("Random seed", value=int(st.session_state.rw_seed), step=1, key="rw_seed_input")
        st.session_state.rw_seed = int(seed)
    with c3:
        step_size = st.number_input("Step size", value=1.0, min_value=0.01, step=0.1)
    with c4:
        show_steps = st.checkbox("Show individual steps", value=n_steps <= 100)

    anim_speed = st.slider(
        "Animation speed (points revealed per frame)",
        min_value=1,
        max_value=max(50, n_steps // 20),
        value=max(1, min(20, n_steps // 50)),
    )
    animate = st.checkbox("Animate path drawing", value=False)

    b1, b2, _ = st.columns([1, 1, 4])
    with b1:
        if st.button("New realization", key="rw_new"):
            st.session_state.rw_nonce += 1
    with b2:
        if st.button("Reset seed path", key="rw_reset"):
            st.session_state.rw_nonce = 0

    seed_eff = _effective_seed("rw_seed", "rw_nonce")
    with st.spinner("Simulating…"):
        t, values = _cached_random_walk(int(n_steps), seed_eff, float(step_size))

    plot_slot = st.empty()
    if animate and n_steps <= 10_000:
        # Progressive reveal — keeps UI responsive for moderate sizes
        n_total = len(t)
        step = max(1, int(anim_speed))
        for end in range(2, n_total + 1, step):
            fig = fig_random_walk(
                t[:end],
                values[:end],
                show_markers=show_steps and end <= 200,
                title=f"Symmetric random walk — {end - 1} / {n_steps} steps",
            )
            plot_slot.plotly_chart(fig, use_container_width=True)
            time.sleep(0.02)
        fig = fig_random_walk(t, values, show_markers=show_steps and n_steps <= 200)
        plot_slot.plotly_chart(fig, use_container_width=True)
    else:
        if animate and n_steps > 10_000:
            st.info("Animation disabled above 10 000 steps to keep the UI responsive.")
        fig = fig_random_walk(
            t,
            values,
            show_markers=show_steps and n_steps <= 200,
            title=f"Symmetric random walk — {n_steps:,} steps (seed={seed_eff})",
        )
        plot_slot.plotly_chart(fig, use_container_width=True)

    mean, var = empirical_mean_var(values[1:])  # exclude M_0 if desired; show terminal stats
    st.caption(
        f"Terminal position Mₙ = **{values[-1]:.0f}** · "
        f"theoretical Var(Mₙ) = **{n_steps}** · "
        f"seed effectif = `{seed_eff}`"
    )
    if show_math:
        math_expander()


# ===========================================================================
# Tab 2 — Scaling Experiment  (PRIORITY)
# ===========================================================================
def tab_scaling(T: float, y_range: Tuple[float, float], show_math: bool):
    st.header("The scaling experiment")
    st.latex(r"W_n(t) = \frac{M_{\lfloor n t \rfloor}}{\sqrt{n}}\,,\qquad t\in[0,T]")
    st.markdown(
        "Increase **n** (steps per unit time). Axes stay fixed so the emergence "
        "of Brownian-like paths is visually unambiguous."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.select_slider(
            "n — steps per unit time",
            options=N_CHOICES_SCALE,
            value=100,
            key="scale_n",
        )
    with c2:
        coupling = st.radio(
            "Randomness across n",
            options=["Coupled (shared increments)", "Independent paths"],
            index=0,
            help=(
                "Coupled: one increment sequence reused at every resolution — "
                "visually coherent refinement. Independent: new path each n — "
                "illustrates convergence in distribution only."
            ),
        )
        coupled = coupling.startswith("Coupled")
    with c3:
        seed = st.number_input(
            "Random seed",
            value=int(st.session_state.scale_seed),
            step=1,
            key="scale_seed_input",
        )
        st.session_state.scale_seed = int(seed)

    b1, b2, b3 = st.columns([1, 1, 3])
    with b1:
        if st.button("New realization", key="scale_new"):
            st.session_state.scale_nonce += 1
    with b2:
        show_markers = st.checkbox("Highlight time grid", value=True)

    seed_eff = _effective_seed("scale_seed", "scale_nonce")
    n_max = max(N_CHOICES_SCALE)

    with st.spinner("Computing scaled path…"):
        t, values = _cached_scaled(
            n=int(n),
            T=T,
            seed=seed_eff,
            coupled=coupled,
            n_max=n_max,
        )

    fig = fig_scaled_walk(
        t,
        values,
        n=int(n),
        T=T,
        y_range=y_range,
        show_markers=show_markers,
    )
    st.plotly_chart(fig, use_container_width=True)

    if coupled:
        st.info(
            "**Coupled mode.** The same underlying ±1 sequence drives every n. "
            "You are watching a refinement of one noise source — ideal for video. "
            "Donsker’s theorem still asserts **convergence in distribution**; "
            "coupling makes the progression pathwise coherent."
        )
    else:
        st.warning(
            "**Independent mode.** Each n draws a new path. Do **not** expect "
            "pathwise convergence — only the *law* of the process approaches "
            "Wiener measure. Prefer coupled mode for YouTube filming."
        )

    # Quick n sweep strip
    st.subheader("Resolution strip")
    st.caption("Same coupling / seed — glance across several n without leaving the tab.")
    cols = st.columns(len(N_CHOICES_SCALE))
    for col, ni in zip(cols, N_CHOICES_SCALE):
        with col:
            ti, vi = _cached_scaled(int(ni), T, seed_eff, coupled, n_max)
            mini = fig_scaled_walk(
                ti,
                vi,
                n=int(ni),
                T=T,
                y_range=y_range,
                title=f"n={ni}",
                show_markers=False,
            )
            mini.update_layout(height=220, margin=dict(l=20, r=10, t=40, b=30))
            mini.layout.annotations = ()  # drop large annotation in strip
            st.plotly_chart(mini, use_container_width=True, config={"displayModeBar": False})

    if show_math:
        math_expander()


# ===========================================================================
# Tab 3 — Scaled vs Unscaled
# ===========================================================================
def tab_compare(T: float, y_range: Tuple[float, float], show_math: bool):
    st.header("Scaled vs. unscaled")
    st.markdown(
        "Left: raw walk on calendar time (variance grows like **n t**). "
        "Right: Donsker scaling (variance stays order **t**)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.select_slider("n", options=N_CHOICES_SCALE, value=100, key="cmp_n")
    with c2:
        seed = st.number_input(
            "Random seed",
            value=int(st.session_state.cmp_seed),
            step=1,
            key="cmp_seed_input",
        )
        st.session_state.cmp_seed = int(seed)
    with c3:
        lock_axes = st.checkbox("Lock fixed axes", value=True)

    # Unscaled y-range grows with sqrt(n T) roughly — offer auto or manual
    auto_unscaled = st.checkbox("Auto y-limits for unscaled panel", value=not lock_axes)
    y_unscaled_manual = st.slider(
        "Unscaled y half-range (used when lock / manual)",
        min_value=10,
        max_value=500,
        value=max(50, int(3 * np.sqrt(n * T))),
        step=10,
    )

    b1, b2, b3 = st.columns([1, 1, 3])
    with b1:
        if st.button("New realization", key="cmp_new"):
            st.session_state.cmp_nonce += 1
    with b2:
        animate_n = st.button("Animate n ↑", key="cmp_anim")

    seed_eff = _effective_seed("cmp_seed", "cmp_nonce")
    plot_slot = st.empty()
    caption_slot = st.empty()

    def _render(ni: int):
        t, uns, sc = _cached_unscaled_scaled(int(ni), T, seed_eff)
        if auto_unscaled and not lock_axes:
            y_u = suggest_y_range([uns], min_span=10.0)
        else:
            half = float(y_unscaled_manual)
            y_u = (-half, half)
        y_s = y_range if lock_axes else suggest_y_range([sc])
        fig = fig_scaled_vs_unscaled(
            t,
            uns,
            sc,
            n=int(ni),
            T=T,
            y_range_unscaled=y_u,
            y_range_scaled=y_s,
            lock_axes=True,
        )
        plot_slot.plotly_chart(fig, use_container_width=True)
        caption_slot.caption(
            f"n = {ni:,} · 1/√n = {1/np.sqrt(ni):.5f} · "
            f"unscaled terminal ≈ {uns[-1]:.0f} · scaled terminal ≈ {sc[-1]:.3f}"
        )

    if animate_n:
        for ni in N_CHOICES_SCALE:
            _render(ni)
            time.sleep(0.55)
    else:
        _render(int(n))

    if show_math:
        math_expander()


# ===========================================================================
# Tab 4 — Variance
# ===========================================================================
def tab_variance(T: float, show_math: bool):
    st.header("Variance experiment")
    st.latex(r"\mathbb{E}[W_n(t)] \approx 0\,,\qquad \mathrm{Var}(W_n(t)) \approx t")
    st.markdown(
        "At grid times $t = k/n$, the identity $\\mathrm{Var}(W_n(t)) = t$ is **exact**. "
        "The chart compares Monte Carlo variance to $\\lfloor n t \\rfloor / n$ and to $t$."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n = st.select_slider("n", options=N_CHOICES_SCALE, value=100, key="var_n")
    with c2:
        n_paths = st.slider("Number of paths", 100, 5000, 1000, step=100)
    with c3:
        seed = st.number_input("Seed", value=int(st.session_state.var_seed), step=1, key="var_seed_input")
        st.session_state.var_seed = int(seed)
    with c4:
        n_eval = st.slider("Evaluation points", 20, 100, 40)

    with st.spinner(f"Simulating {n_paths:,} paths…"):
        t, emp_mean, emp_var, theory_var = _cached_variance(
            int(n), int(n_paths), T, int(seed), int(n_eval)
        )

    fig = fig_variance(t, emp_var, theory_var, emp_mean=emp_mean, T=T)
    st.plotly_chart(fig, use_container_width=True)

    # Snapshot at t = T
    idx = -1
    st.metric(
        label=f"At t ≈ {t[idx]:.4f}",
        value=f"var̂ = {emp_var[idx]:.4f}",
        delta=f"theory {theory_var[idx]:.4f} (Δ = {emp_var[idx] - theory_var[idx]:+.4f})",
        delta_color="off",
    )
    st.caption(f"Empirical mean at terminal time: {emp_mean[idx]:.4f} (theory 0)")

    if show_math:
        math_expander()


# ===========================================================================
# Tab 5 — CLT
# ===========================================================================
def tab_clt(T: float, show_math: bool):
    st.header("Central limit theorem")
    st.markdown(
        "Histogram of many independent copies of $W_n(t)$ vs. the Gaussian density "
        "$\\mathcal{N}(0, \\lfloor n t \\rfloor / n)$."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n = st.select_slider("n", options=N_CHOICES_SCALE, value=100, key="clt_n")
    with c2:
        # Prefer grid-aligned default
        default_t = min(T, round(T * 0.5 * n) / n)
        t = st.number_input(
            "Time t",
            min_value=0.0,
            max_value=float(T),
            value=float(default_t) if default_t > 0 else float(T),
            step=1.0 / max(n, 1),
        )
    with c3:
        n_paths = st.slider("Paths", 200, 10000, 2000, step=100, key="clt_paths")
    with c4:
        seed = st.number_input("Seed", value=int(st.session_state.clt_seed), step=1, key="clt_seed_input")
        st.session_state.clt_seed = int(seed)

    snap = st.checkbox("Snap t to grid k/n (exact variance)", value=True)
    if snap and t > 0:
        t = round(t * n) / n
        t = min(max(t, 1.0 / n), T)

    aligned = is_grid_aligned(t, int(n))
    if aligned:
        st.success(f"t = {t} is grid-aligned → Var(Wₙ(t)) = {t} exactly.")
    else:
        st.info(
            f"t = {t} is off-grid → exact Var = ⌊nt⌋/n = {np.floor(n * t) / n:.6f}."
        )

    n_bins = st.slider("Histogram bins", 10, 100, 40)

    with st.spinner("Sampling…"):
        samples = _cached_clt(int(n), float(t), int(n_paths), int(seed))

    fig = fig_clt_histogram(samples, t=float(t), n=int(n), n_bins=int(n_bins))
    st.plotly_chart(fig, use_container_width=True)

    mean, var = empirical_mean_var(samples)
    m1, m2, m3 = st.columns(3)
    m1.metric("Empirical mean", f"{mean:.4f}")
    m2.metric("Empirical variance", f"{var:.4f}")
    m3.metric("Theoretical variance", f"{np.floor(n * t) / n:.4f}")

    if show_math:
        math_expander()


# ===========================================================================
# Tab 6 — Animation Studio  (PRIORITY)
# ===========================================================================
def tab_animation_studio(T: float, y_range: Tuple[float, float], show_math: bool):
    st.header("Animation Studio")
    st.markdown(
        "High-resolution, YouTube-ready animations of the scaling limit. "
        "Uses a **coupled** increment sequence by default so frames refine one path."
    )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Content")
        theme = st.selectbox("Theme", ["dark", "light"], index=0)
        n_options = st.multiselect(
            "n sequence (pause points)",
            options=N_CHOICES_SCALE,
            default=[10, 50, 100, 500, 1000, 5000, 10000],
        )
        if not n_options:
            n_options = [100, 1000]
        smooth = st.checkbox("Smooth progression (insert intermediate n)", value=True)
        coupled = st.checkbox("Coupled shared increments", value=True)
        seed = st.number_input(
            "Seed",
            value=int(st.session_state.anim_seed),
            step=1,
            key="anim_seed_input",
        )
        st.session_state.anim_seed = int(seed)
        hold = st.slider("Hold frames per n (pause)", 1, 8, 3)
        fps = st.slider("FPS", 1, 12, 2)

        st.subheader("Look")
        linewidth = st.slider("Line thickness", 0.5, 6.0, 2.5, 0.5)
        line_color = st.color_picker(
            "Line color",
            value="#FF6B4A" if theme == "dark" else "#C73E1D",
        )
        show_title = st.checkbox("Show title", value=True)
        show_n = st.checkbox("Show n on screen", value=True)
        show_scale = st.checkbox("Show 1/√n on screen", value=True)
        show_legend = st.checkbox("Show legend", value=False)
        show_grid = st.checkbox("Show grid", value=False)
        minimal = st.checkbox("Minimalist (hide top/right spines)", value=True)

        export_fmt = st.selectbox("Export format", ["mp4", "gif"], index=0)
        resolution = st.selectbox(
            "Resolution",
            ["1920×1080 (16:9)", "1280×720 (16:9)"],
            index=0,
        )
        if resolution.startswith("1920"):
            figsize, dpi = (19.2, 10.8), 100
        else:
            figsize, dpi = (12.8, 7.2), 100

    style = style_for_theme(
        theme,
        linewidth=float(linewidth),
        line_color=line_color,
        show_title=show_title,
        show_n_label=show_n,
        show_scaling_label=show_scale,
        show_legend=show_legend,
        grid=show_grid,
        minimal=minimal,
        figsize=figsize,
        dpi=dpi,
    )
    # Slightly smaller preview figsize to keep Streamlit snappy
    preview_style = AnimationStyle(**{**style.__dict__})
    preview_style.figsize = (11.2, 6.3)
    preview_style.dpi = 100

    config = AnimationConfig(
        n_sequence=sorted(n_options),
        T=T,
        seed=int(seed),
        y_range=y_range,
        fps=int(fps),
        hold_frames=int(hold),
        coupled=coupled,
        style=style,
        smooth_progression=smooth,
    )
    preview_config = AnimationConfig(
        n_sequence=sorted(n_options),
        T=T,
        seed=int(seed),
        y_range=y_range,
        fps=int(fps),
        hold_frames=1,
        coupled=coupled,
        style=preview_style,
        smooth_progression=False,
    )

    with right:
        st.subheader("Preview")
        preview_n = st.select_slider(
            "Preview at n",
            options=sorted(n_options),
            value=sorted(n_options)[min(2, len(n_options) - 1)],
        )
        png = preview_frame_png(preview_config, n=int(preview_n))
        st.image(png, caption=f"Still frame · n = {preview_n:,}", use_container_width=True)

        st.markdown("#### Export")
        if export_fmt == "mp4":
            if ffmpeg_available():
                st.success("FFmpeg detected — MP4 export available.")
            else:
                st.warning(
                    "FFmpeg not found. Install on macOS with:\n\n"
                    "```bash\nbrew install ffmpeg\n```\n\n"
                    "GIF export works without FFmpeg (Pillow)."
                )

        out_name = st.text_input(
            "Output filename",
            value=f"rwbm_scaling_{theme}.{export_fmt}",
        )
        export_dir = default_export_dir()
        out_path = export_dir / out_name

        if st.button("Render & export", type="primary"):
            with st.spinner(
                f"Rendering {export_fmt.upper()} — this may take a minute…"
            ):
                try:
                    path = export_animation(config, str(out_path), fmt=export_fmt)
                    st.success(f"Exported to `{path}`")
                    data = path.read_bytes()
                    mime = "video/mp4" if export_fmt == "mp4" else "image/gif"
                    st.download_button(
                        f"Download {export_fmt.upper()}",
                        data=data,
                        file_name=path.name,
                        mime=mime,
                    )
                    if export_fmt == "gif":
                        st.image(str(path), caption="Exported GIF")
                except Exception as exc:  # noqa: BLE001 — show to educator user
                    st.error(str(exc))

        st.caption(f"Exports directory: `{export_dir}`")

    st.markdown("---")
    st.markdown(
        """
**Filming notes**

1. Prefer **coupled** mode — successive n refine one Brownian-like path.
2. Hold 2–4 frames at pedagogical values (10, 100, 1 000, 10 000).
3. Keep axes fixed (sidebar y-limits) across the whole clip.
4. Dark theme often grades better under video compression; light theme suits slides.
5. Narrate: *convergence in distribution* / Donsker, while showing the coupled refinement.
        """
    )
    if show_math:
        math_expander()


# ===========================================================================
# Main
# ===========================================================================
def main():
    _init_state()
    st.title("From Random Walk to Brownian Motion")
    st.caption(
        "Interactive demonstration of Donsker’s invariance principle — "
        "built for stochastic calculus & mathematical finance education."
    )

    globals_ = sidebar_globals()
    T = globals_["T"]
    y_range = globals_["y_range"]
    show_math = globals_["show_math"]

    tabs = st.tabs(
        [
            "Random Walk",
            "Scaling Experiment",
            "Scaled vs. Unscaled",
            "Variance",
            "Central Limit Theorem",
            "Animation Studio",
        ]
    )
    with tabs[0]:
        tab_random_walk(show_math)
    with tabs[1]:
        tab_scaling(T, y_range, show_math)
    with tabs[2]:
        tab_compare(T, y_range, show_math)
    with tabs[3]:
        tab_variance(T, show_math)
    with tabs[4]:
        tab_clt(T, show_math)
    with tabs[5]:
        tab_animation_studio(T, y_range, show_math)


if __name__ == "__main__":
    main()
