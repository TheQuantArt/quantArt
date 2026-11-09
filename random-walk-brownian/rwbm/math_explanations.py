"""Collapsible mathematical explanations (Markdown) for the Streamlit UI."""

MATH_MARKDOWN = r"""
### 1. Symmetric random walk

Let $(X_i)_{i \geq 1}$ be i.i.d. with
$$
\mathbb{P}(X_i = +1) = \mathbb{P}(X_i = -1) = \tfrac12.
$$
The **symmetric random walk** is
$$
M_0 = 0, \qquad M_k = X_1 + \cdots + X_k \quad (k \in \mathbb{N}).
$$
This is a discrete-time, integer-valued martingale with independent increments.

---

### 2. Why $\mathrm{Var}(M_k) = k$

Since $\mathbb{E}[X_i] = 0$ and $\mathrm{Var}(X_i) = \mathbb{E}[X_i^2] = 1$,
independence gives
$$
\mathrm{Var}(M_k) = \sum_{i=1}^{k} \mathrm{Var}(X_i) = k.
$$
In particular $\mathbb{E}[M_k] = 0$ and $\mathrm{Var}(M_k) = k$ are
**exact finite-$n$ identities**, not merely asymptotic statements.

---

### 3. Derivation of the scaling factor $1/\sqrt{n}$

Fix a continuous-time horizon $t \in [0, T]$. After $\lfloor n t \rfloor$ steps,
$$
\mathrm{Var}\!\bigl(M_{\lfloor n t \rfloor}\bigr) = \lfloor n t \rfloor \approx n t.
$$
Dividing by $\sqrt{n}$ yields
$$
\mathrm{Var}\!\Bigl(\frac{M_{\lfloor n t \rfloor}}{\sqrt{n}}\Bigr)
= \frac{\lfloor n t \rfloor}{n} \;\xrightarrow[n\to\infty]{}\; t.
$$
Thus the scaled process
$$
W_n(t) := \frac{M_{\lfloor n t \rfloor}}{\sqrt{n}}
$$
has the same **second-order structure** as Brownian motion $B_t$, for which
$\mathrm{Var}(B_t) = t$. Without the $1/\sqrt{n}$ factor, variance would explode
like $n t$.

Linear interpolation between the grid points $k/n$ produces a continuous
polygonal path in $C([0,T])$ (or cadlag in $D([0,T])$).

---

### 4. Independent increments

For integers $0 \leq k < \ell$,
$$
M_\ell - M_k = X_{k+1} + \cdots + X_\ell
$$
is independent of $\sigma(M_0,\ldots,M_k)$ and distributed as $M_{\ell-k}$.

After scaling, increments of $W_n$ over disjoint time intervals become
approximately independent $\mathcal{N}(0, \Delta t)$ as $n \to \infty$ —
another Brownian hallmark.

---

### 5. Connection to the central limit theorem

At a fixed time $t > 0$,
$$
W_n(t) = \frac{1}{\sqrt{n}}\sum_{i=1}^{\lfloor n t \rfloor} X_i
= \sqrt{\frac{\lfloor n t \rfloor}{n}} \cdot
\frac{1}{\sqrt{\lfloor n t \rfloor}}\sum_{i=1}^{\lfloor n t \rfloor} X_i.
$$
The classical CLT implies
$$
W_n(t) \;\xrightarrow{d}\; \mathcal{N}(0, t).
$$
On the exact grid $t = k/n$ one has $\mathrm{Var}(W_n(t)) = t$ **exactly**
for every finite $n$; only the *shape* of the law approaches the Gaussian.

---

### 6. Donsker's invariance principle

Donsker's theorem upgrades the marginal CLT to a **functional** limit theorem:
as random elements of $C([0,T])$ (with the uniform topology, after interpolation)
or $D([0,T])$ (Skorokhod topology),
$$
W_n \;\xrightarrow{d}\; B,
$$
where $B$ is standard Brownian motion (Wiener measure).

**Finite-$n$ vs limit.** For each fixed $n$, $W_n$ is a piecewise-linear
process with finitely many degrees of freedom. Continuity of paths, Gaussian
margins, and independent increments hold **exactly** only in the limit
(except for the exact grid variance identity above).

**Convergence in distribution, not pathwise.** If you redraw an independent
walk for every $n$, individual sample paths need not settle visually —
only their *laws* converge. A **coupled** construction (shared increments
$(X_i)$ across resolutions) produces a coherent visual refinement and is
compatible with almost-sure constructions of Brownian motion, but the theorem
itself is about weak convergence of measures.
"""


DISTINCTION_NOTE = """
> **Filming tip.** When demonstrating convergence with *independent* paths,
> say explicitly that you are illustrating **convergence in distribution**.
> Prefer the **coupled** mode in Scaling Experiment / Animation Studio so
> viewers see a single noise source being refined, not unrelated zigzags.
"""
