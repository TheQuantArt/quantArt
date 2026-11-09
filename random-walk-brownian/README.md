# From Random Walk to Brownian Motion

Interactive educational app showing how a **symmetric random walk**, after the
classical Donsker scaling

$$
W_n(t) = \frac{M_{\lfloor nt\rfloor}}{\sqrt{n}},
$$

converges in distribution to **Brownian motion**. Built with Streamlit + Plotly
(+ Matplotlib for YouTube-ready exports).

Designed for filming stochastic-calculus / mathematical-finance explainers.

## Features

| Tab | Purpose |
|-----|---------|
| **Random Walk** | Symmetric ±1 walk with seeds, step size, optional draw animation |
| **Scaling Experiment** | Fixed axes, slider in *n*, coupled or independent paths |
| **Scaled vs. Unscaled** | Side-by-side variance explosion vs. controlled scaling |
| **Variance** | Monte Carlo mean/variance vs. theory `Var(B_t)=t` |
| **Central Limit Theorem** | Histogram of `W_n(t)` vs. `N(0,t)` |
| **Animation Studio** | 16:9 dark/light exports (MP4 via FFmpeg, GIF via Pillow) |

## Requirements

- Python **3.11+** recommended (3.9+ works)
- Optional: **FFmpeg** for MP4 export

## Install & run (macOS)

```bash
git clone https://github.com/TheQuantArt/quantArt.git
cd quantArt/random-walk-brownian

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

The app opens in your browser (typically `http://localhost:8501`).

### FFmpeg (MP4 exports)

```bash
brew install ffmpeg
```

GIF export does **not** require FFmpeg.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Filming tips

1. Use **Coupled (shared increments)** in Scaling Experiment / Animation Studio.
2. Keep sidebar **y-limits** fixed across takes.
3. Pause at pedagogical `n` (10 → 100 → 1 000 → 10 000).
4. Narrate **convergence in distribution** (Donsker); coupling is for visual coherence.
5. Prefer the dark theme for compressed video; light theme for slide overlays.

## Project layout

```
app.py                  # Streamlit entry point
rwbm/
  simulation.py         # Random walks, scaling, coupling
  statistics.py         # Variance & CLT helpers
  plots.py              # Plotly figures
  animation.py          # Matplotlib Animation Studio
  math_explanations.py  # Collapsible theory copy
tests/                  # Scaling, variance, reproducibility
requirements.txt
README.md
```

## Mathematical reminder

- Finite *n*: `E[M_k]=0`, `Var(M_k)=k` exactly; on the grid `Var(W_n(k/n))=k/n`.
- Limit: Donsker’s invariance principle — `W_n ⇒ B` in law on path space.
- Independent redraws for each *n* need **not** converge pathwise.
