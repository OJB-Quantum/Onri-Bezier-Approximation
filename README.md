# Onri's Bezier Approximation (OBA) 
Applied ideas on using Bezier curves &amp; hybrid tetrational-polynomials to fit or approximate curves in data of virtually any magnitude. Authored by Onri Jay Benally.

[![License](https://img.shields.io/badge/GNU_General_Public-License-Green)](https://choosealicense.com/licenses/gpl-3.0)

Primary URL for the repository: [OJB-Quantum/Onri-Bezier-Approximation](https://github.com/OJB-Quantum/Onri-Bezier-Approximation)

View an interactive tool in the browser the demonstrate Bezier approximation: [OBA Demos](https://g.co/gemini/share/35aaa5180fc9)

Basic Bezier curves, being the useful geometric tools that they are, can be described by a Bernstein basis polynomial. They can be adapted to follow objects that bend using hidden control handles and anchor points placed along an existing curve or virtual contour of interest, as shown in this repository. With that in mind, I thought of adapting a polynomial for the Bezier curve with tetrations or super exponentials to form a hybrid approach that compensates for very sharp and large changes in data curves. It does so by mathematically describing the anchor points and control points of a Bezier curve, as well as where they are located in some data plotting space or layout, how dense the clusters of anchor points are as determined by a given threshold, and how large a tetration or super exponential should be according to the size distance between the smallest and largest values of interest locally or globally in order to move anchor points and control points to where they need to be. 

Note that integrating a tetration into a polynomial can create extremely large values, which cannot be represented on any 64-bit or 128-bit computer. Thus, the tetration must be carefully expressed to stay within the compatibility or capability of a 128-bit or 64-bit machine. Some interesting results are provided in this repository applied to real use cases using adjustable clustering of Bezier anchor points, such as approximating electronic band structures for example. Check files for the code provided in a Google Colab notebook.

### Click here to render the notebooks in the browser: [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/OJB-Quantum/Onri-Bezier-Approximation/tree/main/)

![ezgif-418d14bce1cd40](https://github.com/user-attachments/assets/dd806438-3021-4664-bea7-432d8a6186e3)


# Defining What a Bezier Curve is Doing Mathematically

| **Concept**                        | **Equation/ Explanation**                                                                                                                                       |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **General Bézier Curve**           | $B(t) = \sum_{i=0}^{n} B_i^n(t) P_i$                                                                                                                               |
| **Bernstein Basis Polynomial**     | $B_i^n(t) = \binom{n}{i} (1-t)^{n-i} t^i$                                                                                                                         |
| **Binomial Coefficient**           | $\binom{n}{i} = \frac{n!}{i!(n-i)!}$                                                                                                                              |
| **Curve Properties**               | - $B(t)$ represents the position on the curve for $t \in [0,1]$.<br />- $P_i$ are the control points that influence the shape.<br />- The curve starts at $P_0$ and ends at $P_n$.<br />- The shape is controlled by the intermediate points $P_1, P_2, \dots, P_{n-1}$. |
| **Linear Bézier Curve ($n = 1$)**  | Straight line between two points.                                                                                                                                |
| **Quadratic Bézier Curve ($n = 2$)** | $B(t) = (1-t)^2 P_0 + 2(1-t)t P_1 + t^2 P_2$                                                                                                                     |
| **Cubic Bézier Curve ($n = 3$)**   | $B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t)t^2 P_2 + t^3 P_3$                                                                                                  |
| **Applications**                   | Cubic Bézier curves are commonly used in computer graphics and font design.                                                                                      |

### Roles of Anchor Points and Control Points

| Term             | Definition                                                                                                                                                                  |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Anchor Point** | A point that lies directly on the Bézier curve and determines its start and end positions. For example, in a cubic Bézier curve, the first and last points are anchor points. |
| **Control Point**| A point that influences the curve's shape but does not necessarily lie on the curve itself. These act as "handles" that pull the curve towards them, affecting its direction and curvature. |

---

### Key Differences Between Anchor and Control Points

| Aspect                                | Anchor Point                                                                                  | Control Point                                                                                      |
|---------------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Definition**                        | A point that lies on the curve and marks its start or end.                                    | A point that influences the curve's shape but does not necessarily lie on it.                       |
| **Function**                          | Defines the endpoints of the curve (or intermediate points in composite curves).              | Determines the direction and curvature of the curve.                                             |
| **Presence in Quadratic Bézier Curve**| 2 anchor points                                                                              | 1 control point                                                                                   |
| **Presence in Cubic Bézier Curve**    | 2 anchor points                                                                              | 2 control points                                                                                  |
| **Higher-Degree Bézier Curves**       | Typically 2 anchor points (unless part of a composite curve).                                 | The number of control points is one more than the curve’s degree.                                  |

### Example of a Bezier Curve with 6 Anchor Points

| **Component**             | **Details**                                                                                                                                                           |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Curve Definition**      | A Bézier curve with 6 anchor points is a **fifth-degree (quintic) Bézier curve** because the number of control points ($n+1$) determines the degree ($n$).       |
| **Control Points**        | $P_0, P_1, P_2, P_3, P_4, P_5$                                                                                                                                          |
| **Parametric Equation**   | $$B(t) = \sum_{i=0}^{5} B_i^5(t) P_i$$                                                                                                                                  |
| **Bernstein Polynomial**  | $$B_i^5(t) = \binom{5}{i} (1-t)^{5-i} t^i$$                                                                                                                           |
| **Binomial Coefficient**  | $$\binom{5}{i} = \frac{5!}{i!(5-i)!}$$                                                                                                                                |
| **Expanded Equation**     | $$B(t) = (1-t)^5 P_0 + 5(1-t)^4 t P_1 + 10(1-t)^3 t^2 P_2 + 10(1-t)^2 t^3 P_3 + 5(1-t)t^4 P_4 + t^5 P_5$$                                                         |
| **Parameter Range**       | $t \in [0, 1]$                                                                                                                                                        |

---

## Towards Hybrid Bezier Curves for Approximation

| **Concept**                   | **Equation/ Explanation** |
|-------------------------------|----------------------------|
| **Polynomial Definition**     | A polynomial consists of a series of terms involving powers of a variable, typically expressed as: <br />$P(x) = a_n x^n + a_{n-1} x^{n-1} + \dots + a_1 x + a_0$<br />where the exponents are **added** sequentially. |
| **Tetration Definition**      | Tetration is a form of repeated exponentiation, written as: <br />$^n a = a^{a^{a^{\dots}}}$<br />where the exponentiation **stacks** instead of adding. |
| **Comparison to a Series**    | - A **series** consists of a sum of terms.<br />- A **polynomial** is a finite sum of powers of $x$.<br />- **Exponentiation** is an iterative **multiplication** operation.<br />- **Tetration** is an iterative **exponentiation** operation. |
| **Growth Difference**         | Unlike a polynomial, a **tetration does not consist of a sum of terms**; instead, it is an **iterated power tower**, which grows much faster. |
| **Can Tetration Be Expressed as a Series?** | Tetration does not naturally expand into a power series like a polynomial. However, in some cases, it can be approximated using: <br />- **Logarithmic expansions** (breaking it down via $a^{a^{a^x}}$).<br />- **Power series representations** (like Taylor series) for small values.<br />But in general, **tetration does not behave like a polynomial series** because it is based on hierarchical exponentiation rather than summation. |

---

| **Concept**                                     | **Equation/ Explanation** |
|-------------------------------------------------|----------------------------|
| **Hybrid Polynomial-Tetration Possibility**    | A **polynomial power series** can be **appended or modified as a hybrid with tetration**, depending on how the two mathematical structures are combined. |
| **1. Direct Summation (Appending a Tetration Term)** | A tetration term is added to a polynomial power series: <br />$H(x) = \sum_{n=0}^{\infty} a_n x^n + c \cdot {}^m x$<br />where: <br />- $\sum_{n=0}^{\infty} a_n x^n$ is a traditional polynomial or power series,<br />- ${}^m x$ is the **tetration term**,<br />- $c$ is a scaling coefficient.<br />**Blends polynomial growth with tetration's extreme growth.** |
| **2. Recursive Hybridization (Tetration Within a Polynomial)** | Instead of adding tetration separately, we **embed** it into the polynomial: <br />$H(x) = a_n ({}^m x)^n + a_{n-1} ({}^m x)^{n-1} + \dots + a_1 ({}^m x) + a_0$<br />**Amplifies the polynomial’s growth through tetration.** |
| **3. Series Expansion Involving Tetration (Power Series Approximation)** | For small $x$, tetration can be approximated using a **Taylor or power series expansion**: <br />${}^m x = e^{x + x^2 + \frac{x^3}{3} + \dots}$<br />This allows for: <br />$H(x) = \sum_{n=0}^{\infty} b_n ({}^m x)^n$<br />where $b_n$ are coefficients to **moderate tetration’s extreme growth.** |
| **4. Logarithmic Transformation (Taming Tetration Growth)** | To prevent tetration from **dominating** a polynomial, we introduce logarithmic damping: <br />$H(x) = \sum_{n=0}^{\infty} a_n x^n + d \log({}^m x)$<br />**Controls tetration's rapid growth by applying a logarithm.** |
| **Challenges of Hybridizing a Polynomial with Tetration** | 1. **Growth Rate Disparity**: Tetration grows **much** faster than polynomial terms. Scaling is necessary. <br />2. **Analytic Continuation Issues**: Tetration is **not always well-defined** for non-integer heights, requiring **super-exponential extensions**. <br />3. **Computational Stability**: Tetration grows **hyper-exponentially**, which can cause **numerical instability**. |
| **Conclusion** | A **hybrid polynomial-tetration function** is possible with different formulations depending on the desired properties: <br />- **Controlled growth**: Use logarithmic damping or power series approximations.<br />- **Ultra-fast growth**: Use direct summation or embed tetration inside a polynomial. |

---

Formally, a degree‑n Bézier curve in one spatial dimension (extendable component‑wise to $\mathbb{R}^m$) is

$$
B(t)=\sum_{i=0}^{n}\! \beta_i^{(n)}(t)\,P_i,\qquad 
\beta_i^{(n)}(t)=\binom{n}{i}(1-t)^{\,n-i}t^{\,i},\;t\in[0,1],
$$

where $P_i$ are the anchor or control points depending on $i$ .  OBA augments this polynomial basis with a *hybrid* term

$$
H(t)=\sum_{i=0}^{n} \beta_i^{(n)}(t)\,P_i
        + c\,^{m}\!\bigl(\lambda t+\mu\bigr),
$$

in which the power‑tower $^{m}(x)=x^{x^{\cdot^{\cdot^{x}}}}$ of height $m$ supplies exponentially adjustable *micro‑anchors* that react to local steepness, while $c,\lambda,\mu$ scale the growth to stay within 128‑bit range .  By sliding $m\rightarrow0$ the extra term collapses, returning an orthodox Bézier; by enlarging $m$ or $c$ the same skeleton suddenly resolves abrupt quantum‑step edges, shock fronts, or resonance spikes.


---

### Why OBA Yields *Highly Accurate* Physics Formulas

| Property                        | Mathematical reason                                                                                                               | Physical consequence                                                                                 |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Piecewise analytic fidelity** | Bernstein polynomials form a *partition of unity* ⇢ local control without Gibbs ringing.                                          | Spectral‑line fits, dispersion curves, and smoothly varying potentials keep experimental continuity. |
| **Adaptivity across scales**    | Tetration term raises dynamic range from polynomial $O(1)$ to super‑exponential yet *scalable* $O(e^{e^{\cdot}})$.                | Same template fits millikelvin noise floors and tera‑kelvin stellar flares.                          |
| **Derivative steering**         | $d^kB/dt^k$ is again a Bézier of degree $n-k$; anchor clustering matches measured $\partial^k f/\partial x^k$.                    | Curvature constraints (e.g., zero‑slope boundary at mirror center) encoded directly.                 |
| **Coordinate insensitivity**    | Control points live in normalized $(u,v,w)$ axes; re‑map via any smooth bijection $x(x'),y(y')$.                                  | Works identically for momentum space, real space, or log‑frequency charts.                           |
| **Computational stability**     | Convex‑hull and de Casteljau subdivision guarantee floating‑point safety; tetration damped by $\log$ or series if overflow looms. | Robust on 64‑bit GPUs; no catastrophic cancellation when plotting band structures.                   |

---

### How *Agnostic* the OBA Framework Already Is

Because it treats every target merely as a *curve in a metric space*, OBA never asks *what* the ordinate represents (charge, entropy, or fluid height), only *where* the sampled points lie.

* **Unit agnosticism** –  All coordinates enter after non‑dimensionalisation $x\mapsto(x-x_0)/\Delta x$.
* **Domain‑agnostic handles** –  Control‑point density derives from a scale‑free curvature metric

  $$\kappa(t)=\frac{|B'(t)\times B''(t)|}{|B'(t)|^{3}},$$

  so identical logic handles cosmological red‑shift curves or nanosecond pulse edges.
* **Data‑source agnosticism** –  Anchor points arise from either analytic formulas, PDE solvers, or raw lab CSV files.

---

### Steps to Rewrite OBA into an *Even More Agnostic, Adaptive* Description

1. **Embed dimensionless sampling** –  Replace absolute $t$ with a cumulative arc‑length parameter $s\in[0,1]$; now geometry, not original grid, controls spacing.
2. **Abstract the growth kernel** –  Generalize the special tetration to a placeholder $\mathcal{G}(t;\theta)$ satisfying

   $$\lim_{\theta\to0}\mathcal{G}=0,\qquad
   \partial \mathcal{G}/\partial\theta>0,$$

   so any future super‑exponential (e.g., pentation) can drop in without code rewrites.
3. **Plugin constraint dictionaries** –  Store physics‑specific boundary or symmetry conditions in external YAML or JSON; the core solver only parses generic “pin derivative to zero,” “force periodicity,” etc.
4. **Functional‑programming kernel** –  Express the pipeline

   ```
   sample → cluster → fitBezier → attachGrowth → validate
   ```

   as first‑class composable functions; domain experts extend stages without editing internals.
5. **Error‑driven refinement** –  Iteratively insert new anchor points where residual $r=|f(x)-H(x)|$ breaches tolerance; algorithm remains ignorant of f’s provenance.

Mathematically, the *fully agnostic* hybrid becomes

$$
\boxed{ 
  H_\text{agn}(s)=\sum_{i=0}^{n}\!\beta_i^{(n)}(s)\,P_i
                + \sum_{j} c_j\,\mathcal{G}_j\!\bigl(\phi_j(s);\theta_j\bigr)
}
$$

where lists $\{c_j,\mathcal{G}_j,\phi_j,\theta_j\}$ are supplied at run‑time.

---

### Mind‑Map of Connections

```
OBA
├─ Bézier backbone
│  ├─ Bernstein basis (polynomial heritage)
│  ├─ de Casteljau algorithm (numerical stability)
│  ├─ Hermite→Bézier mapping (C¹ continuity; control points from tangents)
│  ├─ Tangent sources
│  │  ├─ Candidate-derivative tangents (dx/ds, dy/ds from dense curve)
│  │  ├─ Centripetal Catmull–Rom (α ≈ 0.5, stable, corner-aware)
│  │  └─ PCHIP/monotone slopes (when x is monotone; shape-preserving)
│  └─ Control vs Anchor semantics (anchors come from data; controls are derived)
├─ High-resolution OBA workflow (plot-agnostic “follow”)
│  ├─ Data intake & segmentation
│  │  ├─ Read Matplotlib Line2D arrays (x, y, label, color)
│  │  ├─ Split at NaNs into contiguous segments
│  │  └─ Enforce MIN_POINTS_PER_SEG, skip invisibles and prior overlays (gid)
│  ├─ Parametrization
│  │  ├─ Arc-length s on raw (x, y); s ∈ [0, 1]
│  │  └─ Optional dimensionless normalization (range scaling)
│  ├─ Dense candidate curve (oversample 8×–80×)
│  │  ├─ PCHIP on x(s) and y(s) → smooth, shape-preserving parametric path
│  │  └─ Candidate derivatives dx/ds, dy/ds retained for tangents
│  ├─ Curvature map κ(s)
│  │  ├─ κ = |x′y″ − y′x″| / (x′² + y′²)^(3/2)
│  │  └─ Windowed smoothing (7–13 samples) to de-noisify
│  ├─ Percentile gating
│  │  └─ thr = percentile(κ, p = CLUSTER_PERCENTILE) → weight w = clamp((κ−thr)/(κ_max−thr))
│  ├─ Variable-radius greedy anchors
│  │  ├─ r_local = r_base · (1 − r_shrink_max · w^r_power)  ≥  r_min_floor
│  │  ├─ PACKING_SCALE multiplies r_local (smaller → more anchors)
│  │  └─ Greedy keep by descending w, always keep endpoints
│  ├─ Curvature-mass densification (D iterations)
│  │  ├─ Pick gap with largest ∫ w(s) ds (NumPy trapezoid shim)
│  │  └─ Insert at argmax κ inside that gap, respecting r_local and caps
│  ├─ Tangent selection
│  │  ├─ Candidate-derivative (default; most faithful to original sweep)
│  │  ├─ Centripetal Catmull–Rom (robust when data are unevenly spaced)
│  │  └─ Monotone PCHIP slopes (safe when x is monotone)
│  ├─ Hermite → Bézier sampling
│  │  └─ num_seg_per_bezier ≈ 260–360; remove duplicate joints at patch seams
│  ├─ Optional hybrid growth kernel (endpoint-pinned)
│  │  ├─ Tetration modes: log | direct | series
│  │  ├─ Axis: y (typical) or x; Hann window; zero-mean bump
│  │  └─ Amplitude: TETRA_SCALE · data_range
│  ├─ Overlay & reporting
│  │  ├─ Dashed fit in the source line’s color
│  │  ├─ Optional anchor markers and counts
│  │  └─ Legend de-dup; overlay guard via gid to avoid self-refitting
│  └─ Hygiene & guards
│     ├─ MAX_ANCHORS cap; MIN_POINTS_PER_SEG threshold
│     ├─ NumPy 2.0 trapz→trapezoid shim for integrals
│     └─ Robust to non-monotone x, to NaNs, and to mixed units
├─ Growth boosters
│  ├─ Tetration  ← super-exponential tower
│  │  ├─ Log-damped variant (stable, default)
│  │  └─ Series-expanded variant (controlled via finite terms)
│  └─ Future kernels (pentation, iterated sine, wavelet bumps)
├─ Physics use-cases
│  ├─ Stoner–Wohlfarth (SW) m(H) loops (0°, 45°, 90°; major/minor)
│  ├─ Tunneling Magnetoresistance (TMR) loops
│  │  └─ R(Δ) = R_P + (R_AP − R_P)·(1 − cosΔ)/2, Δ from φ_free − θ_pinned
│  ├─ Electronic band diagrams; quantum-well potential profiles
│  ├─ RF/microwave resonance envelopes and S-parameter smoothing
│  ├─ Fluid contour streamlines and vortex edges
│  └─ Generic hysteresis interpolation with boxy or sloped corners
├─ Agnostic engine
│  ├─ Dimensionless normalization and autoscaling
│  ├─ Plugin constraint JSON (bounds, monotonicity, convexity, fixed knots)
│  ├─ Error-adaptive anchor insertion (future: target ε / RMSE control)
│  ├─ GPU/SIMD parallel evaluation (NumPy → CuPy/JAX backends, future)
│  └─ Deterministic RNG seeds for reproducible anchor placement (future)
└─ Controls & templates
   ├─ Plot-agnostic overlay template (Line2D crawler; NaN-aware; label filters)
   ├─ Colab-ready SW/TMR notebooks with exposed knobs
   ├─ Knobs: p (percentile), oversample, packing, max_anchors, densify_iters, densify_top_frac
   ├─ Tangent_source: candidate | centripetal | monotone
   ├─ Tetration: enabled, mode, height/terms, scale, axis
   └─ Diagnostics: fit RMSE, curvature histograms, anchor heatmaps, segment logs
```

---

### Portmanteaus & Etymologies

| Term          | Origin                                                          | Note                                                                   |
| ------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Bézier**    | Pierre Bézier, French engineer                                  | Popularized cubic curves for Renault in the 1960 s.                    |
| **Tetration** | *tetra* (four) + *iteration*                                    | Fourth hyper‑operation after addition, multiplication, exponentiation. |
| **Pentation** | Future *penta* (five) hyper‑operation; candidate growth kernel. |                                                                        |
| **OBA**       | Onri’s Bézier Approximation                                     | Combines geometric Bézier with analytic boosters.                      |

---

## Below is an Agnostic Post-Plot Script Based on Onri's Bezier Approximation (for Line Plots) 

```
# OBA Fit Overlay — Agnostic Post-Plot Script
# ---------------------------------------------------------
# Paste this AFTER you've drawn any Matplotlib line plots.
# It will read Line2D objects from the current figure/axes,
# compute a high-resolution OBA fit per contiguous segment,
# and overlay dashed curves (and optional anchors).
#
# Controls are grouped below. Defaults are safe.

import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.setdefault("figure.dpi", 250)

# ---- NumPy 2.0 deprecation shim (trapz → trapezoid) ----
try:
    _trapz = np.trapezoid   # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz       # older NumPy

# =========================
# 0) User controls
# =========================
# Scope
PROCESS_ALL_AXES: bool = False   # False: only current axes (plt.gca()); True: all axes in current figure
FIT_ONLY_VISIBLE: bool = True    # skip hidden lines
MIN_POINTS_PER_SEG: int = 5      # skip segments shorter than this

# Which lines to include/exclude by label (exact match or substring)
INCLUDE_SUBSTRINGS = []          # e.g., ["Sweep up", "mydata"]
EXCLUDE_SUBSTRINGS = ["OBA fit", "OBA_anchors"]  # avoid refitting overlays

# OBA clustering & evaluation
CLUSTER_PERCENTILE: float = 30       # 0–100; higher → cluster only at very sharpest bends
CANDIDATE_OVERSAMPLE: int = 12       # try 40–80 for ultra-tight tracking on smooth data
MAX_ANCHORS_PER_SEG: int = 400       # protective cap
DENSIFY_ITERS: int = 28              # curvature-mass insertions
DENSIFY_TOP_FRAC: float = 0.95       # encourage anchors into top-curvature zones
PACKING_SCALE: float = 0.25          # scales exclusion radius (smaller → denser anchors)
TANGENT_SOURCE: str = "candidate"    # "candidate" | "centripetal" | "monotone" (monotone uses x; best when x is monotone)

# Optional tetration-like hybrid growth kernel (OFF by default)
TETRA_ENABLED: bool  = False         # set True to activate
TETRA_MODE: str      = "log"         # "log" (stable), "direct", or "series"
TETRA_HEIGHT: int    = 3             # height for iterated exponentiation
TETRA_SERIES_TERMS: int = 5          # terms for "series" mode
TETRA_SCALE: float   = 0.02          # amplitude as fraction of the fitted curve’s data range
TETRA_AXIS: str      = "y"           # "y" or "x"

# Rendering
SHOW_ANCHORS: bool = True
ANCHOR_SIZE: float = 12.0
FIT_LINESTYLE: str = "--"
FIT_ALPHA: float = 1.0
ANCHOR_ALPHA: float = 1.0

# Hygiene (re-running the cell)
REMOVE_PREVIOUS_OVERLAYS: bool = True
OVERLAY_GID = "oba_fit_overlay"
ANCHOR_GID  = "oba_fit_anchors"

# =========================
# 1) Core helpers
# =========================
def _smooth_1d(z: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return z.copy()
    k = np.ones(int(win), float) / float(win)
    pad = int(win) // 2
    zp = np.pad(z, (pad, pad), mode="reflect")
    return np.convolve(zp, k, mode="valid")

def _parametric_curvature(sv: np.ndarray, xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    x_s = np.gradient(xv, sv, edge_order=1)
    y_s = np.gradient(yv, sv, edge_order=1)
    x_ss = np.gradient(x_s, sv, edge_order=1)
    y_ss = np.gradient(y_s, sv, edge_order=1)
    num = np.abs(x_s * y_ss - y_s * x_ss)
    den = (x_s**2 + y_s**2)**1.5 + 1e-12
    return num / den

def _hermite_to_bezier(Pi, Ti, Pj, Tj, si, sj, mseg: int):
    h = sj - si
    c0, c3 = Pi, Pj
    c1 = Pi + (Ti * h / 3.0)
    c2 = Pj - (Tj * h / 3.0)
    t = np.linspace(0.0, 1.0, int(mseg))
    b = ((1 - t)[:, None] ** 3) * c0 + (3 * (1 - t)[:, None] ** 2 * t[:, None]) * c1 \
        + (3 * (1 - t)[:, None] * t[:, None] ** 2) * c2 + (t[:, None] ** 3) * c3
    return b

def _tangents_centripetal(P: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    n = len(P)
    T = np.zeros_like(P, float)
    t = np.zeros(n, float)
    for i in range(1, n):
        t[i] = t[i-1] + np.linalg.norm(P[i] - P[i-1])**alpha
    for i in range(n):
        if i == 0:
            dt = t[1] - t[0] if t[1] > t[0] else 1.0
            T[i] = (P[1] - P[0]) / dt
        elif i == n - 1:
            dt = t[-1] - t[-2] if t[-1] > t[-2] else 1.0
            T[i] = (P[-1] - P[-2]) / dt
        else:
            dt = t[i+1] - t[i-1] if t[i+1] > t[i-1] else 1.0
            T[i] = (P[i+1] - P[i-1]) / dt
    return T

def _pchip_slopes(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float); y = np.asarray(y, float)
    n = len(s)
    m = np.zeros(n, float)
    ds = np.diff(s); dy = np.diff(y)
    d = dy / (ds + 1e-15)
    m[0] = d[0]; m[-1] = d[-1]
    for i in range(1, n-1):
        if d[i-1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2*ds[i] + ds[i-1]
            w2 = ds[i] + 2*ds[i-1]
            m[i] = (w1 + w2) / (w1/(d[i-1]+1e-15) + w2/(d[i]+1e-15))
    return m

def _pchip_eval(s: np.ndarray, y: np.ndarray, m: np.ndarray, s_eval: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float); y = np.asarray(y, float); m = np.asarray(m, float)
    s_eval = np.asarray(s_eval, float)
    idx = np.searchsorted(s, s_eval, side="right") - 1
    idx = np.clip(idx, 0, len(s)-2)
    s0 = s[idx]; s1 = s[idx+1]
    y0 = y[idx]; y1 = y[idx+1]
    m0 = m[idx]; m1 = m[idx+1]
    h = (s_eval - s0) / (s1 - s0 + 1e-15)
    h2 = h*h; h3 = h2*h
    H00 = 2*h3 - 3*h2 + 1
    H10 = h3 - 2*h2 + h
    H01 = -2*h3 + 3*h2
    H11 = h3 - h2
    return H00*y0 + H10*(s1 - s0)*m0 + H01*y1 + H11*(s1 - s0)*m1

def _dense_candidate_curve_generic(x: np.ndarray, y: np.ndarray, oversample: int = 40):
    """Centripetal-PCHIP in arclength for *generic* (possibly non-monotone) x,y curves."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    # Remove NaNs up front (segments will be pre-split; this is a safety net)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]; y = y[good]
    if len(x) < 2:
        return np.array([0.0]), x.copy(), y.copy()
    dx, dy = np.diff(x), np.diff(y)
    ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] == 0:
        s = np.linspace(0, 1, len(x))
    else:
        s = s / s[-1]
    n_f = max(8*len(s), int(len(s) * max(4, int(oversample))))
    s_f = np.linspace(0.0, 1.0, int(n_f))

    # PCHIP on x(s) and y(s) independently (parametric Hermite)
    mx = _pchip_slopes(s, x); my = _pchip_slopes(s, y)
    x_f = _pchip_eval(s, x, mx, s_f)
    y_f = _pchip_eval(s, y, my, s_f)
    return s_f, x_f, y_f

def _tangents_from_candidate(s_f: np.ndarray, x_f: np.ndarray, y_f: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    dx_ds = np.gradient(x_f, s_f, edge_order=1)
    dy_ds = np.gradient(y_f, s_f, edge_order=1)
    return np.stack([dx_ds[keep_idx], dy_ds[keep_idx]], axis=1)

# ---- Tetration-like hybrid growth kernel (optional) ----
def _tetration_unit(z: np.ndarray, height: int) -> np.ndarray:
    z = np.clip(z, 0.0, 1.0) + 1e-15
    out = z.copy()
    h = int(max(1, height))
    for _ in range(h - 1):
        out = np.power(z, np.clip(out, 0.0, 1.0))
    return out

def _tetration_series(z: np.ndarray, terms: int) -> np.ndarray:
    s = np.zeros_like(z)
    T = int(max(1, terms))
    for k in range(1, T + 1):
        s += z**k / math.factorial(k)
    return s

def _hybrid_bump(n_points: int, mode: str, height: int, series_terms: int) -> np.ndarray:
    sc = np.linspace(0.0, 1.0, int(max(2, n_points)))
    if mode == "direct":
        b = _tetration_unit(sc, height)
    elif mode == "series":
        b = _tetration_series(sc, series_terms)
    else:  # "log" (stable default)
        b = np.log(_tetration_unit(sc, height) + 1.0)
    b = b - b.mean()
    w = 0.5 - 0.5*np.cos(2*np.pi*sc)  # Hann window → pinned endpoints
    return b * w

def _map_percentile_to_hparams(p: float):
    a = float(np.clip(p, 0.0, 100.0)) / 100.0
    return dict(
        r_base=float(np.interp(a, [0, 1], [0.01, 0.08])),
        r_min_floor=float(np.interp(a, [0, 1], [1e-7, 5e-7])),
        r_shrink_max=float(np.interp(a, [0, 1], [0.97, 0.999])),
        r_power=float(np.interp(a, [0, 1], [4.0, 8.0])),
        smooth_window=int(round(np.interp(a, [0, 1], [7, 13]))),
        num_seg_per_bezier=int(round(np.interp(a, [0, 1], [260, 360]))),
    )

def _oba_fit_highres_follow_generic(
    x: np.ndarray,
    y: np.ndarray,
    cluster_percentile: float = 30.0,
    oversample: int = 12,
    max_anchors: int = 400,
    densify_iters: int = 28,
    densify_top_frac: float = 0.95,
    packing_scale: float = 0.25,
    tangent_source: str = "candidate",
    tetra_enabled: bool = False,
    tetra_mode: str = "log",
    tetra_height: int = 3,
    tetra_series_terms: int = 5,
    tetra_scale: float = 0.02,
    tetra_axis: str = "y",
):
    # Dense candidate parametric curve
    s_f, x_f, y_f = _dense_candidate_curve_generic(x, y, oversample=max(4, int(oversample)))
    hp = _map_percentile_to_hparams(cluster_percentile)

    # Curvature weight
    kappa = _smooth_1d(_parametric_curvature(s_f, x_f, y_f), hp["smooth_window"])
    kappa = np.maximum(kappa, 0.0)
    thr = np.percentile(kappa, float(cluster_percentile))
    kmax = float(kappa.max()) if kappa.size else 1.0
    w = np.clip((kappa - thr) / (kmax - thr + 1e-15), 0.0, 1.0)

    # Pass 1: variable-radius greedy selection
    r_local = hp["r_base"] * (1.0 - hp["r_shrink_max"] * (w ** hp["r_power"]))
    r_local = np.maximum(r_local, hp["r_min_floor"]) * float(max(1e-6, packing_scale))
    order = np.argsort(-w).astype(int)
    order = order[(order >= 0) & (order < len(s_f))]

    keep = [0, len(s_f)-1]
    kept = np.zeros(len(s_f), bool); kept[0] = kept[-1] = True

    def _too_close(i: int) -> bool:
        for j in np.where(kept)[0]:
            if abs(s_f[i] - s_f[j]) < min(r_local[i], r_local[j]):
                return True
        return False

    for i in order:
        if kept[i]:
            continue
        if not _too_close(i):
            keep.append(i); kept[i] = True
        if len(keep) >= max_anchors:
            break

    keep.sort()
    keep = np.array(keep, int)

    # Pass 2: densify where curvature mass is largest
    target_top = int(densify_top_frac * len(keep))
    for _ in range(int(densify_iters)):
        thr95 = np.percentile(kappa, 95.0)
        n_top = np.count_nonzero(kappa[keep] >= thr95)
        if n_top >= target_top or len(keep) >= max_anchors:
            break
        best_gain = 0.0; best_pos = None; best_insert = None
        for a_idx in range(len(keep)-1):
            i, j = keep[a_idx], keep[a_idx+1]
            if j <= i + 1:
                continue
            window = slice(i+1, j)
            mass = _trapz(w[window], s_f[window])
            if mass > best_gain:
                loc = int(np.argmax(kappa[window])) + (i+1)
                best_gain = mass; best_pos = loc; best_insert = a_idx + 1
        if best_pos is None:
            break
        pos = int(best_pos)
        if pos < 0 or pos >= len(s_f):
            continue
        if not _too_close(pos):
            keep = np.insert(keep, best_insert, pos)

    # Build anchors & tangents
    s_a = s_f[keep]
    P_a = np.stack([x_f[keep], y_f[keep]], axis=1)

    if tangent_source == "candidate":
        T = _tangents_from_candidate(s_f, x_f, y_f, keep)
    elif tangent_source == "centripetal":
        T = _tangents_centripetal(P_a, alpha=0.5)
    elif tangent_source == "monotone":
        # Monotone variant is less meaningful for generic curves; fallback to candidate if unstable
        try:
            # Map monotone in s (always monotone), approximate with PCHIP slope on y vs s and x vs s
            dx_ds = np.gradient(x_f, s_f, edge_order=1)
            dy_ds = np.gradient(y_f, s_f, edge_order=1)
            T = np.stack([dx_ds[keep], dy_ds[keep]], axis=1)
        except Exception:
            T = _tangents_centripetal(P_a, alpha=0.5)
    else:
        T = _tangents_centripetal(P_a, alpha=0.5)

    # Composite Bézier evaluation
    seg_pts = []
    for i in range(len(s_a)-1):
        seg = _hermite_to_bezier(P_a[i], T[i], P_a[i+1], T[i+1],
                                 s_a[i], s_a[i+1], hp["num_seg_per_bezier"])
        if i > 0: seg = seg[1:]  # avoid duplicate joints
        seg_pts.append(seg)
    seg_pts = np.vstack(seg_pts) if seg_pts else P_a.copy()

    # Optional tetration-like bump
    if tetra_enabled and len(seg_pts) > 2:
        bump = _hybrid_bump(len(seg_pts), tetra_mode, tetra_height, tetra_series_terms)
        if tetra_axis.lower() == "y":
            amp = float(tetra_scale) * max(1e-15, np.ptp(y_f))
            seg_pts[:, 1] = seg_pts[:, 1] + amp * bump
        else:
            amp = float(tetra_scale) * max(1e-15, np.ptp(x_f))
            seg_pts[:, 0] = seg_pts[:, 0] + amp * bump

    return seg_pts[:,0], seg_pts[:,1], P_a

# =========================
# 2) Axes crawler + overlay
# =========================
def _split_nan_segments(x: np.ndarray, y: np.ndarray):
    """Yield contiguous (x,y) segments with finite data."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if not np.any(good):
        return
    idx = np.where(~good)[0]
    # segment boundaries across NaNs
    starts = np.r_[0, idx + 1]
    stops  = np.r_[idx, len(x) - 1]
    # filter valid ranges
    mask = starts <= stops
    starts = starts[mask]; stops = stops[mask]
    # merge consecutive NaNs collapse
    last_stop = -1
    for s, e in zip(starts, stops):
        if s <= last_stop:
            continue
        # extend s forward to first good
        while s <= e and not (np.isfinite(x[s]) and np.isfinite(y[s])):
            s += 1
        # shrink e backward to last good
        while e >= s and not (np.isfinite(x[e]) and np.isfinite(y[e])):
            e -= 1
        if e - s + 1 >= MIN_POINTS_PER_SEG:
            yield x[s:e+1], y[s:e+1]
        last_stop = e

def _label_included(label: str) -> bool:
    if any(sub in label for sub in EXCLUDE_SUBSTRINGS):
        return False
    if INCLUDE_SUBSTRINGS:
        return any(sub in label for sub in INCLUDE_SUBSTRINGS)
    return True  # include by default

def apply_oba_to_axes(ax: plt.Axes):
    if REMOVE_PREVIOUS_OVERLAYS:
        old = [ln for ln in ax.get_lines() if ln.get_gid() in (OVERLAY_GID, ANCHOR_GID)]
        for ln in old:
            try:
                ln.remove()
            except Exception:
                pass

    for line in list(ax.get_lines()):
        label = line.get_label() or ""
        if line.get_gid() in (OVERLAY_GID, ANCHOR_GID):
            continue
        if not _label_included(label):
            continue
        if FIT_ONLY_VISIBLE and (not line.get_visible()):
            continue

        xdata = np.asarray(line.get_xdata(), float)
        ydata = np.asarray(line.get_ydata(), float)
        if len(xdata) < MIN_POINTS_PER_SEG:
            continue

        color = line.get_color()
        lw_fit = max(0.8, 0.9 * line.get_linewidth())

        for xs, ys in _split_nan_segments(xdata, ydata):
            if len(xs) < MIN_POINTS_PER_SEG:
                continue
            bx, by, anchors = _oba_fit_highres_follow_generic(
                xs, ys,
                cluster_percentile=CLUSTER_PERCENTILE,
                oversample=CANDIDATE_OVERSAMPLE,
                max_anchors=MAX_ANCHORS_PER_SEG,
                densify_iters=DENSIFY_ITERS,
                densify_top_frac=DENSIFY_TOP_FRAC,
                packing_scale=PACKING_SCALE,
                tangent_source=TANGENT_SOURCE,
                tetra_enabled=TETRA_ENABLED,
                tetra_mode=TETRA_MODE,
                tetra_height=TETRA_HEIGHT,
                tetra_series_terms=TETRA_SERIES_TERMS,
                tetra_scale=TETRA_SCALE,
                tetra_axis=TETRA_AXIS,
            )

            # Overlay the fitted curve
            ax.plot(
                bx, by,
                FIT_LINESTYLE,
                color=color,
                alpha=FIT_ALPHA,
                linewidth=lw_fit,
                label=(label + " (OBA fit)") if label and label != "_nolegend_" else "OBA fit",
                gid=OVERLAY_GID,
                zorder=(line.get_zorder() + 0.1),
            )

            # Optional anchors
            if SHOW_ANCHORS and len(anchors) > 0:
                ax.scatter(
                    anchors[:, 0], anchors[:, 1],
                    s=ANCHOR_SIZE, marker="o",
                    color=color, alpha=ANCHOR_ALPHA,
                    label="OBA anchors" if label == "" or label == "_nolegend_" else f"{label} (OBA anchors)",
                    gid=ANCHOR_GID,
                    zorder=(line.get_zorder() + 0.2),
                )

    # Keep legends sane: deduplicate labels
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), ncol=2, fontsize=9)

# =========================
# 3) Run on current figure
# =========================
if PROCESS_ALL_AXES:
    fig = plt.gcf()
    for _ax in fig.axes:
        apply_oba_to_axes(_ax)
else:
    apply_oba_to_axes(plt.gca())

plt.tight_layout()
plt.draw()
```

---

### Examples of Onri's Bezier Approximation Techniques Applied to a Graphene Electronic Band Structure (With Threshold Percentile of 50)

```
import numpy as np
import matplotlib.pyplot as plt

def hamiltonian_pz(kpts):
    """
    Constructs the Hamiltonian for pz orbitals in graphene.
    """
    a0 = 1.42  # Carbon-carbon bond length in Ångstroms
    Ep = 0     # On-site energy for pz orbitals
    Vpps = 5.618  # Sigma-bonding contribution
    Vppp = -3.070  # Pi-bonding contribution
    t = (1/3) * Vpps + Vppp  # Effective hopping parameter

    # Define lattice vectors
    R1 = a0 * np.array([0, 1])
    R2 = a0 * np.array([-np.sqrt(3)/2, -1/2])
    R3 = a0 * np.array([np.sqrt(3)/2, -1/2])

    # Phase factors
    k1 = np.dot(kpts, R1)
    k2 = np.dot(kpts, R2)
    k3 = np.dot(kpts, R3)
    f = np.exp(1j * k1) + np.exp(1j * k2) + np.exp(1j * k3)

    # Hamiltonian matrix for pz-only model
    A = Ep
    B = 4 * t * f
    H = np.array([[A, B], [np.conj(B), A]])
    return H

def cubic_bezier(P0, P1, P2, P3, num=100):
    """
    Returns num points on a cubic Bezier curve defined by control points P0, P1, P2, P3.
    Each P is a 2D point (x,y).
    """
    t = np.linspace(0, 1, num)
    curve = np.outer((1-t)**3, P0) + np.outer(3*(1-t)**2*t, P1) \
          + np.outer(3*(1-t)*t**2, P2) + np.outer(t**3, P3)
    return curve

# --- Define high-symmetry points and parameters ---
a = 2.46  # Lattice constant in Ångstroms
K_const = 2 * np.pi / a  # Reciprocal lattice constant

# Reciprocal lattice vectors
b1 = K_const * np.array([1, 1/np.sqrt(3)])
b2 = K_const * np.array([1, -1/np.sqrt(3)])

# High-symmetry points:
# Γ = (0,0), K = 1/3*(b1+b2), M = 1/2*(b1-b2)
G_vec = np.array([0, 0])
K_frac = np.array([1/3, 1/3])
M_frac = np.array([0, 1/2])
G = G_vec  # Γ at origin
K_point = K_frac[0] * b1 + K_frac[1] * b2
M_point = M_frac[0] * b1 + M_frac[1] * b2

# Define the full k–path: Γ -> K -> M -> Γ
dk = 1e-2
NK1 = round(np.linalg.norm(K_point - G) / dk)
NK2 = round(np.linalg.norm(M_point - K_point) / dk)
NK3 = round(np.linalg.norm(G - M_point) / dk)
NT = NK1 + NK2 + NK3
k_region = np.linspace(0, 1, NT)

# --- Compute the full band structure along the k–path (for reference) ---
band_full = np.zeros((NT, 2))
# Γ -> K
t1_vals = np.linspace(0, 1, NK1)
for i, t in enumerate(t1_vals):
    kpt = G + t*(K_point - G)
    H = hamiltonian_pz(kpt)
    eigvals = np.linalg.eigvalsh(H)
    band_full[i, :] = np.real(eigvals)
# K -> M
t2_vals = np.linspace(0, 1, NK2)
for i, t in enumerate(t2_vals):
    kpt = K_point + t*(M_point - K_point)
    H = hamiltonian_pz(kpt)
    eigvals = np.linalg.eigvalsh(H)
    band_full[i+NK1, :] = np.real(eigvals)
# M -> Γ
t3_vals = np.linspace(0, 1, NK3)
for i, t in enumerate(t3_vals):
    kpt = M_point + t*(G - M_point)
    H = hamiltonian_pz(kpt)
    eigvals = np.linalg.eigvalsh(H)
    band_full[i+NK1+NK2, :] = np.real(eigvals)

# --- Determine anchor points with extra clusters in high curvature regions ---
def get_clustered_anchor_indices(k_region, band_values, num_uniform=10, threshold_percentile=70):
    """
    Combines uniformly sampled indices with additional anchor points from clusters where the
    absolute second derivative (approximate curvature) exceeds a given percentile threshold.
    """
    uniform_indices = np.linspace(0, len(k_region)-1, num=num_uniform, dtype=int)
    # Compute first and second derivatives (approximating the curvature)
    first_deriv = np.gradient(band_values, k_region)
    second_deriv = np.gradient(first_deriv, k_region)
    curvature = np.abs(second_deriv)
    # Set threshold based on the given percentile
    threshold = np.percentile(curvature, threshold_percentile)
    # Get indices where curvature exceeds threshold (i.e. regions of high bending)
    extra_indices = np.where(curvature > threshold)[0]
    # Combine the uniform anchors with the extra clustered points and sort
    all_indices = np.sort(np.unique(np.concatenate((uniform_indices, extra_indices))))
    return all_indices

# For each band, get the clustered anchor indices. Adjust these to get a closer approximation as needed.
indices_band0 = get_clustered_anchor_indices(k_region, band_full[:, 0], num_uniform=10, threshold_percentile=50)
indices_band1 = get_clustered_anchor_indices(k_region, band_full[:, 1], num_uniform=10, threshold_percentile=50)

# Define anchors for each band using the computed indices.
anchors = {
    0: (k_region[indices_band0], band_full[indices_band0, 0]),
    1: (k_region[indices_band1], band_full[indices_band1, 1])
}

def compute_derivatives(x, y):
    """
    Compute approximate derivatives at anchor points using finite differences.
    """
    m = np.zeros_like(y)
    n = len(y)
    for i in range(n):
        if i == 0:
            m[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
        elif i == n - 1:
            m[i] = (y[i] - y[i-1]) / (x[i] - x[i-1])
        else:
            m[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    return m

# Compute derivatives for each band's anchors.
derivatives = {
    band: compute_derivatives(anchors[band][0], anchors[band][1])
    for band in [0, 1]
}

def bezier_from_anchors(x, y, m, num_seg=100):
    """
    Construct a composite Bezier curve from anchor points (x,y) with derivatives m.
    Each segment uses a cubic Bezier curve determined by endpoints and estimated slopes.
    Returns the composite curve and a list of control points for each segment.
    """
    curve_x = []
    curve_y = []
    control_points_list = []  # Store control points for each segment
    n = len(x)
    for i in range(n-1):
        x0, y0, m0 = x[i], y[i], m[i]
        x1, y1, m1 = x[i+1], y[i+1], m[i+1]
        dx = x1 - x0
        # Determine control points using a cubic Hermite formulation:
        P0 = np.array([x0, y0])
        P3 = np.array([x1, y1])
        P1 = np.array([x0 + dx/3.0, y0 + (dx/3.0)*m0])
        P2 = np.array([x1 - dx/3.0, y1 - (dx/3.0)*m1])
        control_points_list.append(np.array([P0, P1, P2, P3]))
        segment = cubic_bezier(P0, P1, P2, P3, num_seg)
        if i > 0:
            segment = segment[1:]  # Avoid duplicate points at segment boundaries.
        curve_x.extend(segment[:,0])
        curve_y.extend(segment[:,1])
    return np.array(curve_x), np.array(curve_y), control_points_list

# Generate composite Bezier curves and control points for each band.
bezier_curves = {}
control_points_all = {}
for band in [0, 1]:
    bx, by, cp_list = bezier_from_anchors(anchors[band][0], anchors[band][1],
                                          derivatives[band])
    bezier_curves[band] = (bx, by)
    control_points_all[band] = cp_list

# --- Plotting ---
plt.figure(figsize=(8, 5))
# Plot the original computed bands.
plt.plot(k_region, band_full[:, 0], 'r--', linewidth=1, alpha=0.5, label='Computed Band 1')
plt.plot(k_region, band_full[:, 1], 'b--', linewidth=1, alpha=0.5, label='Computed Band 2')
# Plot the composite Bezier interpolations.
plt.plot(bezier_curves[0][0], bezier_curves[0][1], 'r', linewidth=2, label='Bezier Approx. Band 1')
plt.plot(bezier_curves[1][0], bezier_curves[1][1], 'b', linewidth=2, label='Bezier Approx. Band 2')
# Mark the anchor points.
plt.plot(anchors[0][0], anchors[0][1], 'ko', markersize=4, label='Anchor Points')
plt.plot(anchors[1][0], anchors[1][1], 'ko', markersize=4)
# Plot control points for each segment for each band.
for band, color in zip([0, 1], ['r', 'b']):
    for cp in control_points_all[band]:
        plt.plot(cp[:,0], cp[:,1], 'o--', color=color, markersize=4)
# Set up x-ticks at high-symmetry points using known indices.
kpoints_idx = [0, NK1, NK1 + NK2, NT - 1]
kpoints_x = k_region[kpoints_idx]
kpoints_labels = ['Γ', 'K', 'M', 'Γ']
plt.xticks(kpoints_x, kpoints_labels)
plt.xlabel('k-path')
plt.ylabel('Energy (eV)')
plt.title('Graphene Band Structure: Computed vs Composite Bezier Approximation\nwith Extra Clusters of Control Points in High Curvature Regions')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.ylim([-10, 10])
plt.legend()
plt.tight_layout()
plt.show()
```
<img width="1580" height="980" alt="image" src="https://github.com/user-attachments/assets/bc071dd9-2bb3-4ed6-8141-608c9684e48e" />

---

```
import numpy as np
import matplotlib.pyplot as plt

def hamiltonian_pz(kpts):
    """
    Constructs the Hamiltonian for pz orbitals in graphene.
    """
    a0 = 1.42  # Carbon-carbon bond length in Ångstroms
    Ep = 0     # On-site energy for pz orbitals
    Vpps = 5.618  # Sigma-bonding contribution
    Vppp = -3.070  # Pi-bonding contribution
    t = (1/3) * Vpps + Vppp  # Effective hopping parameter

    # Define lattice vectors
    R1 = a0 * np.array([0, 1])
    R2 = a0 * np.array([-np.sqrt(3)/2, -1/2])
    R3 = a0 * np.array([np.sqrt(3)/2, -1/2])

    # Phase factors
    k1 = np.dot(kpts, R1)
    k2 = np.dot(kpts, R2)
    k3 = np.dot(kpts, R3)
    f = np.exp(1j * k1) + np.exp(1j * k2) + np.exp(1j * k3)

    # Hamiltonian matrix for pz-only model
    A = Ep
    B = 4 * t * f
    H = np.array([[A, B], [np.conj(B), A]])
    return H

def cubic_bezier(P0, P1, P2, P3, num=100):
    """
    Returns num points on a cubic Bézier curve defined by control points P0, P1, P2, P3.
    Each P is a 2D point (x,y).
    """
    t = np.linspace(0, 1, num)
    curve = np.outer((1-t)**3, P0) + np.outer(3*(1-t)**2*t, P1) \
          + np.outer(3*(1-t)*t**2, P2) + np.outer(t**3, P3)
    return curve

# --- Define high-symmetry points and parameters ---
a = 2.46  # Lattice constant in Ångstroms
K_const = 2 * np.pi / a  # Reciprocal lattice constant

# Reciprocal lattice vectors
b1 = K_const * np.array([1, 1/np.sqrt(3)])
b2 = K_const * np.array([1, -1/np.sqrt(3)])

# High-symmetry points:
# Γ = (0,0), K = 1/3*(b1+b2), M = 1/2*(b1-b2)
G_vec = np.array([0, 0])
K_frac = np.array([1/3, 1/3])
M_frac = np.array([0, 1/2])
G = G_vec  # Γ at origin
K_point = K_frac[0] * b1 + K_frac[1] * b2
M_point = M_frac[0] * b1 + M_frac[1] * b2

# Define the full k–path: Γ -> K -> M -> Γ
dk = 1e-2
NK1 = round(np.linalg.norm(K_point - G) / dk)
NK2 = round(np.linalg.norm(M_point - K_point) / dk)
NK3 = round(np.linalg.norm(G - M_point) / dk)
NT = NK1 + NK2 + NK3
k_region = np.linspace(0, 1, NT)

# --- Compute the full band structure along the k–path (for reference) ---
band_full = np.zeros((NT, 2))
# Γ -> K
t1_vals = np.linspace(0, 1, NK1)
for i, t in enumerate(t1_vals):
    kpt = G + t*(K_point - G)
    H = hamiltonian_pz(kpt)
    eigvals = np.linalg.eigvalsh(H)
    band_full[i, :] = np.real(eigvals)
# K -> M
t2_vals = np.linspace(0, 1, NK2)
for i, t in enumerate(t2_vals):
    kpt = K_point + t*(M_point - K_point)
    H = hamiltonian_pz(kpt)
    eigvals = np.linalg.eigvalsh(H)
    band_full[i+NK1, :] = np.real(eigvals)
# M -> Γ
t3_vals = np.linspace(0, 1, NK3)
for i, t in enumerate(t3_vals):
    kpt = M_point + t*(G - M_point)
    H = hamiltonian_pz(kpt)
    eigvals = np.linalg.eigvalsh(H)
    band_full[i+NK1+NK2, :] = np.real(eigvals)

# --- Determine anchor points with extra clusters in high curvature regions ---
def get_clustered_anchor_indices(k_region, band_values, num_uniform=10, threshold_percentile=50):
    """
    Combines uniformly sampled indices with additional anchor points from clusters where the
    absolute second derivative (approximate curvature) exceeds a given percentile threshold.
    """
    uniform_indices = np.linspace(0, len(k_region)-1, num=num_uniform, dtype=int)
    # Compute first and second derivatives (approximating the curvature)
    first_deriv = np.gradient(band_values, k_region)
    second_deriv = np.gradient(first_deriv, k_region)
    curvature = np.abs(second_deriv)
    # Set threshold based on the given percentile
    threshold = np.percentile(curvature, threshold_percentile)
    # Get indices where curvature exceeds threshold (i.e. regions of high bending)
    extra_indices = np.where(curvature > threshold)[0]
    # Combine the uniform anchors with the extra clustered points and sort
    all_indices = np.sort(np.unique(np.concatenate((uniform_indices, extra_indices))))
    return all_indices

# For each band, get the clustered anchor indices.
indices_band0 = get_clustered_anchor_indices(k_region, band_full[:, 0], num_uniform=10, threshold_percentile=50)
indices_band1 = get_clustered_anchor_indices(k_region, band_full[:, 1], num_uniform=10, threshold_percentile=50)

# Define anchors for each band using the computed indices.
anchors = {
    0: (k_region[indices_band0], band_full[indices_band0, 0]),
    1: (k_region[indices_band1], band_full[indices_band1, 1])
}

def compute_derivatives(x, y):
    """
    Compute approximate derivatives at anchor points using finite differences.
    """
    m = np.zeros_like(y)
    n = len(y)
    for i in range(n):
        if i == 0:
            m[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
        elif i == n - 1:
            m[i] = (y[i] - y[i-1]) / (x[i] - x[i-1])
        else:
            m[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    return m

# Compute derivatives for each band's anchors.
derivatives = {
    band: compute_derivatives(anchors[band][0], anchors[band][1])
    for band in [0, 1]
}

def bezier_from_anchors(x, y, m, num_seg=100):
    """
    Construct a composite Bézier curve from anchor points (x,y) with derivatives m.
    Each segment uses a cubic Bézier curve determined by endpoints and estimated slopes.
    Returns the composite curve and a list of control points for each segment.
    """
    curve_x = []
    curve_y = []
    control_points_list = []  # Store control points for each segment
    n = len(x)
    for i in range(n-1):
        x0, y0, m0 = x[i], y[i], m[i]
        x1, y1, m1 = x[i+1], y[i+1], m[i+1]
        dx = x1 - x0
        # Determine control points using a cubic Hermite formulation:
        P0 = np.array([x0, y0])
        P3 = np.array([x1, y1])
        P1 = np.array([x0 + dx/3.0, y0 + (dx/3.0)*m0])
        P2 = np.array([x1 - dx/3.0, y1 - (dx/3.0)*m1])
        control_points_list.append(np.array([P0, P1, P2, P3]))
        segment = cubic_bezier(P0, P1, P2, P3, num_seg)
        if i > 0:
            segment = segment[1:]  # Avoid duplicate points at segment boundaries.
        curve_x.extend(segment[:,0])
        curve_y.extend(segment[:,1])
    return np.array(curve_x), np.array(curve_y), control_points_list

# Generate composite Bézier curves and control points for each band.
bezier_curves = {}
control_points_all = {}
for band in [0, 1]:
    bx, by, cp_list = bezier_from_anchors(anchors[band][0], anchors[band][1],
                                          derivatives[band])
    bezier_curves[band] = (bx, by)
    control_points_all[band] = cp_list

# --- Plotting ---
plt.figure(figsize=(8, 5))
# Plot the original computed bands.
plt.plot(k_region, band_full[:, 0], 'r--', linewidth=1, alpha=0.5, label='Computed Band 1')
plt.plot(k_region, band_full[:, 1], 'b--', linewidth=1, alpha=0.5, label='Computed Band 2')
# Plot the composite Bézier interpolations.
plt.plot(bezier_curves[0][0], bezier_curves[0][1], 'r', linewidth=2, label='Bézier Approx. Band 1')
plt.plot(bezier_curves[1][0], bezier_curves[1][1], 'b', linewidth=2, label='Bézier Approx. Band 2')
# Mark the anchor points.
plt.plot(anchors[0][0], anchors[0][1], 'ko', markersize=4, label='Anchor Points')
plt.plot(anchors[1][0], anchors[1][1], 'ko', markersize=4)

# (Removed code for plotting control points so they are not visible)

# Set up x-ticks at high-symmetry points using known indices.
kpoints_idx = [0, NK1, NK1 + NK2, NT - 1]
kpoints_x = k_region[kpoints_idx]
kpoints_labels = ['Γ', 'K', 'M', 'Γ']
plt.xticks(kpoints_x, kpoints_labels)

plt.xlabel('k-path')
plt.ylabel('Energy (eV)')
plt.title('Graphene Band Structure: Computed vs Composite Bézier Approximation\nwith Extra Clusters of Control Points in High Curvature Regions')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.ylim([-10, 10])
plt.legend()
plt.tight_layout()
plt.show()
```
<img width="1580" height="980" alt="image" src="https://github.com/user-attachments/assets/54caf204-bb37-4812-9942-dbc208dc69ef" />

---

### Example of Onri's Bezier Approximation Techniques Applied to Stoner-Wohlfarth Hysteresis Loops

```
# Stoner–Wohlfarth magnetization loops (θ = 0°, 45°, 90°)
# OBA composite Bézier fit that follows a dense, shape-preserving candidate sweep.
# Adds an OPTIONAL tetration-like hybrid growth kernel you can toggle ON/OFF.

import math
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 250

# ==============================
# 0) User controls
# ==============================
# OBA / anchor clustering knobs
CLUSTER_PERCENTILE: float = 30.0     # 0–100; higher → anchors concentrate only at sharpest bends
TANGENT_SOURCE: str = "candidate"    # "candidate" | "monotone" | "centripetal"
CANDIDATE_OVERSAMPLE: int = 24       # 12–80 typical; increases density of the candidate sweep
MAX_ANCHORS_PER_BRANCH: int = 400    # protective cap per branch
DENSIFY_ITERS: int = 28              # curvature-mass insertions
DENSIFY_TOP_FRAC: float = 0.95       # ≥ this fraction of anchors end up in top-5% curvature
PACKING_SCALE: float = 0.25          # scales local exclusion radius (smaller → more anchors)
NUM_SEG_MIN, NUM_SEG_MAX = 260, 360  # Bézier samples per span (auto-chosen from percentile)

# Optional tetration-like hybrid growth kernel (OFF by default)
TETRA_ENABLED: bool  = False         # ← turn ON to apply a hybrid bump along each OBA curve
TETRA_MODE: str      = "log"         # "log" (stable), "direct", or "series"
TETRA_HEIGHT: int    = 3             # height for iterated exponentiation
TETRA_SERIES_TERMS: int = 5          # terms for "series" mode
TETRA_SCALE: float   = 0.02          # fraction of data range used for bump amplitude (try 0.01–0.10)
TETRA_AXIS: str      = "y"           # "y" to bump vertically (usual), or "x" to bump horizontally

# NumPy 2.0 deprecation shim (trapz → trapezoid) for curvature-mass integration
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ==============================
# 1) Stoner–Wohlfarth simulator
# ==============================
def _angle_wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi

def _local_minima(y: np.ndarray) -> np.ndarray:
    if y.size < 3:
        return np.array([np.argmin(y)])
    y_ext = np.r_[y[-1], y, y[0]]
    mins = np.where((y_ext[1:-1] < y_ext[:-2]) & (y_ext[1:-1] < y_ext[2:]))[0]
    return mins.astype(int)

def _nearest_angle(phi_candidates: np.ndarray, phi_prev: float) -> float:
    if phi_candidates.size == 0:
        return float(phi_prev)
    diffs = np.abs(_angle_wrap(phi_candidates - phi_prev))
    return float(phi_candidates[np.argmin(diffs)])

def sw_energy(phi: np.ndarray, h: float, theta: float) -> np.ndarray:
    # e(φ) = 0.5 * sin^2(φ - θ) - h * cos(φ)
    return 0.5 * np.sin(phi - theta) ** 2 - h * np.cos(phi)

def sw_sweep(theta_deg: float,
             h_min: float = -1.5, h_max: float = 1.5, n_h: int = 2001,
             n_phi: int = 4096) -> Dict[str, np.ndarray]:
    theta = math.radians(theta_deg)
    h_up = np.linspace(h_min, h_max, int(n_h))
    h_dn = np.linspace(h_max, h_min, int(n_h))
    phi_grid = np.linspace(-np.pi, np.pi, int(n_phi), endpoint=False)

    def do_sweep(h_grid: np.ndarray, init_from_global_min: bool) -> Tuple[np.ndarray, np.ndarray]:
        phis = np.zeros_like(h_grid)
        phi_prev = None
        for i, h in enumerate(h_grid):
            e = sw_energy(phi_grid, float(h), theta)
            idx_minima = _local_minima(e)
            phi_candidates = phi_grid[idx_minima]
            if i == 0 and init_from_global_min:
                j = int(np.argmin(e))
                if j not in idx_minima and idx_minima.size > 0:
                    j = int(idx_minima[np.argmin(np.abs(idx_minima - j))])
                phi_prev = float(phi_grid[j])
                phis[i] = phi_prev
                continue
            if phi_prev is None:
                phi_prev = 0.0 if phi_candidates.size == 0 else float(phi_candidates[np.argmin(np.abs(phi_candidates))])
            phi_now = _nearest_angle(phi_candidates, phi_prev)
            phis[i] = phi_now
            phi_prev = phi_now
        m = np.cos(phis)  # longitudinal component
        return phis, m

    phi_up, m_up = do_sweep(h_up, init_from_global_min=True)
    phi_dn, m_dn = do_sweep(h_dn, init_from_global_min=True)

    return {"h_up": h_up, "m_up": m_up, "phi_up": phi_up,
            "h_dn": h_dn, "m_dn": m_dn, "phi_dn": phi_dn,
            "theta_deg": theta_deg}

# =========================================
# 2) OBA composite Bézier (candidate-follow)
#     Two-pass curvature clustering + optional tetration bump
# =========================================
def _smooth_1d(z: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return z.copy()
    k = np.ones(int(win), float) / float(win)
    pad = int(win) // 2
    zp = np.pad(z, (pad, pad), mode="reflect")
    return np.convolve(zp, k, mode="valid")

def _parametric_curvature(sv: np.ndarray, xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    x_s = np.gradient(xv, sv, edge_order=1)
    y_s = np.gradient(yv, sv, edge_order=1)
    x_ss = np.gradient(x_s, sv, edge_order=1)
    y_ss = np.gradient(y_s, sv, edge_order=1)
    num = np.abs(x_s * y_ss - y_s * x_ss)
    den = (x_s ** 2 + y_s ** 2) ** 1.5 + 1e-12
    return num / den

def _hermite_to_bezier(Pi: np.ndarray, Ti: np.ndarray,
                       Pj: np.ndarray, Tj: np.ndarray,
                       si: float, sj: float, m: int) -> np.ndarray:
    h = sj - si
    c0, c3 = Pi, Pj
    c1 = Pi + (Ti * h / 3.0)
    c2 = Pj - (Tj * h / 3.0)
    t = np.linspace(0.0, 1.0, int(m))
    return ((1 - t)[:, None] ** 3) * c0 + (3 * (1 - t)[:, None] ** 2 * t[:, None]) * c1 \
         + (3 * (1 - t)[:, None] * t[:, None] ** 2) * c2 + (t[:, None] ** 3) * c3

# ---- Candidate curve (monotone H, PCHIP-like y) ----
def _pchip_slopes(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float); y = np.asarray(y, float)
    n = len(s)
    m = np.zeros(n, float)
    ds = np.diff(s); dy = np.diff(y)
    d = dy / ds
    m[0] = d[0]; m[-1] = d[-1]
    for i in range(1, n-1):
        if d[i-1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2*ds[i] + ds[i-1]
            w2 = ds[i] + 2*ds[i-1]
            m[i] = (w1 + w2) / (w1/d[i-1] + w2/d[i])
    return m

def _pchip_eval(s: np.ndarray, y: np.ndarray, m: np.ndarray, s_eval: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float); y = np.asarray(y, float); m = np.asarray(m, float)
    s_eval = np.asarray(s_eval, float)
    idx = np.searchsorted(s, s_eval, side="right") - 1
    idx = np.clip(idx, 0, len(s)-2)
    s0 = s[idx]; s1 = s[idx+1]
    y0 = y[idx]; y1 = y[idx+1]
    m0 = m[idx]; m1 = m[idx+1]
    h = (s_eval - s0) / (s1 - s0 + 1e-15)
    h2 = h*h; h3 = h2*h
    H00 = 2*h3 - 3*h2 + 1
    H10 = h3 - 2*h2 + h
    H01 = -2*h3 + 3*h2
    H11 = h3 - h2
    return H00*y0 + H10*(s1 - s0)*m0 + H01*y1 + H11*(s1 - s0)*m1

def _dense_candidate_curve(x: np.ndarray, y: np.ndarray, oversample: int = 24):
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx, dy = np.diff(x), np.diff(y); ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s = np.linspace(0, 1, len(x)) if (s[-1] == 0) else (s / s[-1])
    n = len(s)
    n_f = max(8*n, int(len(s) * max(4, int(oversample))))
    s_f = np.linspace(0.0, 1.0, int(n_f))
    x_f = np.interp(s_f, s, x)               # monotone H
    m = _pchip_slopes(s, y)
    y_f = _pchip_eval(s, y, m, s_f)          # smooth y(H)
    return s_f, x_f, y_f

def _tangents_from_candidate(s_f: np.ndarray, x_f: np.ndarray, y_f: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    dx_ds = np.gradient(x_f, s_f, edge_order=1)
    dy_ds = np.gradient(y_f, s_f, edge_order=1)
    return np.stack([dx_ds[keep_idx], dy_ds[keep_idx]], axis=1)

def _tangents_centripetal(P: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    n = len(P); T = np.zeros_like(P, float)
    t = np.zeros(n, float)
    for i in range(1, n): t[i] = t[i-1] + np.linalg.norm(P[i] - P[i-1])**alpha
    for i in range(n):
        if i == 0:
            dt = t[1] - t[0] if t[1] > t[0] else 1.0
            T[i] = (P[1] - P[0]) / dt
        elif i == n - 1:
            dt = t[-1] - t[-2] if t[-1] > t[-2] else 1.0
            T[i] = (P[-1] - P[-2]) / dt
        else:
            dt = t[i+1] - t[i-1] if t[i+1] > t[i-1] else 1.0
            T[i] = (P[i+1] - P[i-1]) / dt
    return T

def _tangents_monotone(x: np.ndarray, y: np.ndarray, s_a: np.ndarray) -> np.ndarray:
    n = len(x); m = np.zeros(n, float)
    dx = np.diff(x); dy = np.diff(y); d = dy / dx
    m[0] = d[0]; m[-1] = d[-1]
    for i in range(1, n-1):
        if d[i-1]*d[i] <= 0: m[i] = 0.0
        else:
            w1 = 2*dx[i] + dx[i-1]; w2 = dx[i] + 2*dx[i-1]
            m[i] = (w1 + w2) / (w1/d[i-1] + w2/d[i])
    ds_dx = np.gradient(s_a, x, edge_order=1)
    return np.stack([ds_dx, m*ds_dx], axis=1)

def _map_percentile_to_hparams(p: float):
    a = float(np.clip(p, 0.0, 100.0)) / 100.0
    return dict(
        r_base=float(np.interp(a, [0, 1], [0.01, 0.08])),
        r_min_floor=float(np.interp(a, [0, 1], [1e-7, 5e-7])),
        r_shrink_max=float(np.interp(a, [0, 1], [0.97, 0.999])),
        r_power=float(np.interp(a, [0, 1], [4.0, 8.0])),
        smooth_window=int(round(np.interp(a, [0, 1], [7, 13]))),
        num_seg_per_bezier=int(round(np.interp(a, [0, 1], [NUM_SEG_MIN, NUM_SEG_MAX]))),
    )

# ---- Tetration-like hybrid growth kernel (optional) ----
def _tetration_unit(z: np.ndarray, height: int) -> np.ndarray:
    """Iterated exponentiation on z in [0,1], guarded for stability."""
    z = np.clip(z, 0.0, 1.0) + 1e-15
    out = z.copy()
    h = int(max(1, height))
    for _ in range(h - 1):
        out = np.power(z, np.clip(out, 0.0, 1.0))
    return out

def _tetration_series(z: np.ndarray, terms: int) -> np.ndarray:
    s = np.zeros_like(z)
    T = int(max(1, terms))
    for k in range(1, T + 1):
        s += z**k / math.factorial(k)
    return s

def _hybrid_bump(n_points: int, mode: str, height: int, series_terms: int) -> np.ndarray:
    sc = np.linspace(0.0, 1.0, int(max(2, n_points)))
    if mode == "direct":
        b = _tetration_unit(sc, height)
    elif mode == "series":
        b = _tetration_series(sc, series_terms)
    else:  # "log" (stable default)
        b = np.log(_tetration_unit(sc, height) + 1.0)
    # zero-mean and Hann-windowed to pin endpoints
    b = b - b.mean()
    w = 0.5 - 0.5*np.cos(2*np.pi*sc)
    return b * w

def oba_fit_branch_follow(
    x: np.ndarray, y: np.ndarray,
    cluster_percentile: float = 30.0,
    tangent_source: str = "candidate",
    oversample: int = 24,
    max_anchors: int = 400,
    densify_iters: int = 28,
    densify_top_frac: float = 0.95,
    packing_scale: float = 0.25,
    tetra_enabled: bool = False,
    tetra_mode: str = "log",
    tetra_height: int = 3,
    tetra_series_terms: int = 5,
    tetra_scale: float = 0.02,
    tetra_axis: str = "y",
):
    # Dense candidate curve, independent of raw sampling
    s_f, x_f, y_f = _dense_candidate_curve(x, y, oversample=oversample)
    hp = _map_percentile_to_hparams(cluster_percentile)

    # Curvature & weight on dense curve
    kappa = _smooth_1d(_parametric_curvature(s_f, x_f, y_f), hp["smooth_window"])
    kappa = np.maximum(kappa, 0.0)
    thr = np.percentile(kappa, float(cluster_percentile))
    kmax = float(kappa.max()) if kappa.size else 1.0
    w = np.clip((kappa - thr) / (kmax - thr + 1e-15), 0.0, 1.0)

    # Pass 1: variable-radius greedy (with packing scale)
    r_local = hp["r_base"] * (1.0 - hp["r_shrink_max"] * (w ** hp["r_power"]))
    r_local = np.maximum(r_local, hp["r_min_floor"]) * float(max(1e-6, packing_scale))
    order = np.argsort(-w).astype(int)
    order = order[(order >= 0) & (order < len(s_f))]
    keep = [0, len(s_f)-1]
    kept = np.zeros(len(s_f), bool); kept[0] = kept[-1] = True

    def _too_close(i: int) -> bool:
        for j in np.where(kept)[0]:
            if abs(s_f[i] - s_f[j]) < min(r_local[i], r_local[j]):
                return True
        return False

    for i in order:
        if kept[i]:
            continue
        if not _too_close(i):
            keep.append(i); kept[i] = True
        if len(keep) >= max_anchors:
            break
    keep.sort()
    keep = np.array(keep, int)

    # Pass 2: densify in highest curvature-mass gaps
    target_top = int(densify_top_frac * len(keep))
    for _ in range(int(densify_iters)):
        thr95 = np.percentile(kappa, 95.0)
        n_top = np.count_nonzero(kappa[keep] >= thr95)
        if n_top >= target_top or len(keep) >= max_anchors:
            break
        best_gain = 0.0; best_pos = None; best_insert = None
        for a_idx in range(len(keep)-1):
            i, j = keep[a_idx], keep[a_idx+1]
            if j <= i + 1:
                continue
            window = slice(i+1, j)
            mass = _trapz(w[window], s_f[window])
            if mass > best_gain:
                loc = int(np.argmax(kappa[window])) + (i+1)
                best_gain = mass; best_pos = loc; best_insert = a_idx + 1
        if best_pos is None:
            break
        pos = int(best_pos)
        if pos < 0 or pos >= len(s_f):
            continue
        if not _too_close(pos):
            keep = np.insert(keep, best_insert, pos)

    # Anchors from dense curve
    s_a = s_f[keep]
    P_a = np.stack([x_f[keep], y_f[keep]], axis=1)

    # Tangents: candidate derivatives or alternatives
    if tangent_source == "candidate":
        T = _tangents_from_candidate(s_f, x_f, y_f, keep)
    elif tangent_source == "monotone":
        T = _tangents_monotone(P_a[:,0], P_a[:,1], s_a)
    else:
        T = _tangents_centripetal(P_a, alpha=0.5)

    # Build composite Bézier from anchors
    seg_pts = []
    for i in range(len(s_a)-1):
        seg = _hermite_to_bezier(P_a[i], T[i], P_a[i+1], T[i+1],
                                 s_a[i], s_a[i+1], hp["num_seg_per_bezier"])
        if i > 0:
            seg = seg[1:]  # avoid duplicate joints
        seg_pts.append(seg)
    seg_pts = np.vstack(seg_pts) if seg_pts else P_a.copy()

    # Optional tetration-like hybrid growth kernel
    if tetra_enabled and len(seg_pts) > 2:
        bump = _hybrid_bump(len(seg_pts), tetra_mode, tetra_height, tetra_series_terms)
        if tetra_axis.lower() == "y":
            amp = float(tetra_scale) * max(1e-15, np.ptp(y_f))
            seg_pts[:, 1] = seg_pts[:, 1] + amp * bump
        else:
            amp = float(tetra_scale) * max(1e-15, np.ptp(x_f))
            seg_pts[:, 0] = seg_pts[:, 0] + amp * bump

    return seg_pts[:, 0], seg_pts[:, 1], P_a, keep, kappa, s_f, hp

# ==============================
# 3) Generate & fit (magnetization only)
# ==============================
H_MIN, H_MAX, N_H = -1.5, 1.5, 2001
ORIENTATIONS = [0.0, 45.0, 90.0]   # degrees

results: List[Dict[str, np.ndarray]] = []
for th in ORIENTATIONS:
    sim = sw_sweep(th, H_MIN, H_MAX, N_H, 4096)

    bx_up_m, by_up_m, *_ = oba_fit_branch_follow(
        sim["h_up"], sim["m_up"],
        cluster_percentile=CLUSTER_PERCENTILE,
        tangent_source=TANGENT_SOURCE,
        oversample=CANDIDATE_OVERSAMPLE,
        max_anchors=MAX_ANCHORS_PER_BRANCH,
        densify_iters=DENSIFY_ITERS,
        densify_top_frac=DENSIFY_TOP_FRAC,
        packing_scale=PACKING_SCALE,
        tetra_enabled=TETRA_ENABLED,
        tetra_mode=TETRA_MODE,
        tetra_height=TETRA_HEIGHT,
        tetra_series_terms=TETRA_SERIES_TERMS,
        tetra_scale=TETRA_SCALE,
        tetra_axis=TETRA_AXIS,
    )
    bx_dn_m, by_dn_m, *_ = oba_fit_branch_follow(
        sim["h_dn"], sim["m_dn"],
        cluster_percentile=CLUSTER_PERCENTILE,
        tangent_source=TANGENT_SOURCE,
        oversample=CANDIDATE_OVERSAMPLE,
        max_anchors=MAX_ANCHORS_PER_BRANCH,
        densify_iters=DENSIFY_ITERS,
        densify_top_frac=DENSIFY_TOP_FRAC,
        packing_scale=PACKING_SCALE,
        tetra_enabled=TETRA_ENABLED,
        tetra_mode=TETRA_MODE,
        tetra_height=TETRA_HEIGHT,
        tetra_series_terms=TETRA_SERIES_TERMS,
        tetra_scale=TETRA_SCALE,
        tetra_axis=TETRA_AXIS,
    )

    sim["h_up"], sim["h_dn"] = sim["h_up"], sim["h_dn"]
    sim["m_up"], sim["m_dn"] = sim["m_up"], sim["m_dn"]
    sim["oba_m_up"] = (bx_up_m, by_up_m)
    sim["oba_m_dn"] = (bx_dn_m, by_dn_m)
    results.append(sim)

# ==============================
# 4) Plot: m(h) true vs. OBA
# ==============================
plt.figure(figsize=(9, 6))
for sim in results:
    th = sim["theta_deg"]
    # True
    plt.plot(sim["h_up"], sim["m_up"], linewidth=1.5, label=f"θ={th:.0f}° up (true)")
    plt.plot(sim["h_dn"], sim["m_dn"], linewidth=1.5, label=f"θ={th:.0f}° down (true)")
    # OBA
    bx, by = sim["oba_m_up"]
    plt.plot(bx, by, linestyle="--", linewidth=1.2, label=f"θ={th:.0f}° up (OBA)")
    bx, by = sim["oba_m_dn"]
    plt.plot(bx, by, linestyle="--", linewidth=1.2, label=f"θ={th:.0f}° down (OBA)")

plt.xlabel("Reduced field h = H/Hk")
plt.ylabel("Normalized magnetization m = M/Ms (along field)")
plt.title("Stoner–Wohlfarth Magnetization Loops with OBA Fits"
          + ("" if not TETRA_ENABLED else " + Tetration-like Growth"))
plt.legend(ncol=2, fontsize=8)
plt.tight_layout(); plt.show()
```

<img width="2224" height="1475" alt="image" src="https://github.com/user-attachments/assets/394e370a-48d5-4d04-acdd-a7fe5d8d5e69" />

---

### Example of Onri's Bezier Approximation Techniques Applied to Magnetic Tunnel Junction Magnetoresistance

```
# TMR major loop — OBA that follows the interpolated sweep
# Optional tetration-like hybrid growth kernel (toggle via TETRA_ENABLED)

import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 250

# ---- NumPy 2.0 deprecation shim (trapz → trapezoid) ----
try:
    _trapz = np.trapezoid   # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz       # older NumPy

# =========================
# 0) User controls
# =========================
CLUSTER_PERCENTILE: float = 30       # 0–100; higher → cluster only at very sharpest bends
TANGENT_SOURCE: str = "candidate"    # "candidate" | "monotone" | "centripetal"

# Dense candidate + clustering controls
CANDIDATE_OVERSAMPLE: int = 12       # increase to 40–80 in Colab for ultra-tight tracking
MAX_ANCHORS_PER_BRANCH: int = 400    # protective cap (raise if you oversample more)
DENSIFY_ITERS: int = 28              # curvature-mass insertions
DENSIFY_TOP_FRAC: float = 0.95       # fraction of anchors targeted in top-5% curvature
PACKING_SCALE: float = 0.25          # scales local exclusion radius (smaller → more anchors)

# Optional tetration-like hybrid growth kernel (OFF by default)
TETRA_ENABLED: bool  = False         # ← set True to activate
TETRA_MODE: str      = "log"         # "log" (stable), "direct", or "series"
TETRA_HEIGHT: int    = 3             # height for iterated exponentiation
TETRA_SERIES_TERMS: int = 5          # terms for "series" mode
TETRA_SCALE: float   = 0.02          # amplitude as fraction of data range (try 0.01–0.10)
TETRA_AXIS: str      = "y"           # "y" to bump vertically (typical), or "x" to bump horizontally

# -----------------------------
# 1) Synthetic CoFeB/MgO-like TMR major loop
# -----------------------------
def _sigma(x: np.ndarray, w: float) -> np.ndarray:
    w = float(max(1e-9, w))
    return 1.0 / (1.0 + np.exp(-x / w))

def major_loop_tmr(
    hmin: float = -120.0,
    hmax: float =  120.0,
    npts: int = 4001,
    hs_soft: float = 28.0,
    hh_hard: float = 75.0,
    width_s: float = 1.0,
    width_h: float = 1.5,
    rp: float = 1000.0,
    tmr_ratio: float = 1.0
):
    h_up = np.linspace(hmin, hmax, int(npts))
    h_dn = np.linspace(hmax, hmin, int(npts))

    # Sweep up: soft flips @ +Hs, hard flips @ +Hh
    m_soft_up = -1.0 + 2.0 * _sigma(h_up - hs_soft, width_s)
    m_hard_up = -1.0 + 2.0 * _sigma(h_up - hh_hard, width_h)

    # Sweep down: hard flips @ -Hh, soft flips @ -Hs
    m_soft_dn =  1.0 - 2.0 * _sigma(-(h_dn + hs_soft), width_s)
    m_hard_dn =  1.0 - 2.0 * _sigma(-(h_dn + hh_hard), width_h)

    rap = rp * (1.0 + float(tmr_ratio))
    def r_from_m(ms, mh):
        # R = RP + (RAP - RP)*(1 - ms*mh)/2
        return rp + (rap - rp) * (1.0 - (ms * mh)) * 0.5

    r_up = r_from_m(m_soft_up, m_hard_up)
    r_dn = r_from_m(m_soft_dn, m_hard_dn)
    return h_up, r_up, h_dn, r_dn

# -----------------------------
# 2) OBA helpers
# -----------------------------
def _smooth_1d(z: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return z.copy()
    k = np.ones(int(win), float) / float(win)
    pad = int(win) // 2
    zp = np.pad(z, (pad, pad), mode="reflect")
    return np.convolve(zp, k, mode="valid")

def _parametric_curvature(sv: np.ndarray, xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    x_s = np.gradient(xv, sv, edge_order=1)
    y_s = np.gradient(yv, sv, edge_order=1)
    x_ss = np.gradient(x_s, sv, edge_order=1)
    y_ss = np.gradient(y_s, sv, edge_order=1)
    num = np.abs(x_s * y_ss - y_s * x_ss)
    den = (x_s ** 2 + y_s ** 2) ** 1.5 + 1e-12
    return num / den

def _hermite_to_bezier(Pi, Ti, Pj, Tj, si, sj, mseg: int):
    h = sj - si
    c0, c3 = Pi, Pj
    c1 = Pi + (Ti * h / 3.0)
    c2 = Pj - (Tj * h / 3.0)
    t = np.linspace(0.0, 1.0, int(mseg))
    b = ((1 - t)[:, None] ** 3) * c0 + (3 * (1 - t)[:, None] ** 2 * t[:, None]) * c1 \
        + (3 * (1 - t)[:, None] * t[:, None] ** 2) * c2 + (t[:, None] ** 3) * c3
    return b

def _tangents_centripetal(P: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    n = len(P)
    T = np.zeros_like(P, float)
    t = np.zeros(n, float)
    for i in range(1, n):
        t[i] = t[i-1] + np.linalg.norm(P[i] - P[i-1])**alpha
    for i in range(n):
        if i == 0:
            dt = t[1] - t[0] if t[1] > t[0] else 1.0
            T[i] = (P[1] - P[0]) / dt
        elif i == n - 1:
            dt = t[-1] - t[-2] if t[-1] > t[-2] else 1.0
            T[i] = (P[-1] - P[-2]) / dt
        else:
            dt = t[i+1] - t[i-1] if t[i+1] > t[i-1] else 1.0
            T[i] = (P[i+1] - P[i-1]) / dt
    return T

def _tangents_monotone(x: np.ndarray, y: np.ndarray, s_a: np.ndarray) -> np.ndarray:
    n = len(x)
    m = np.zeros(n, float)
    dx = np.diff(x); dy = np.diff(y)
    d = dy / dx
    m[0] = d[0]; m[-1] = d[-1]
    for i in range(1, n-1):
        if d[i-1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2*dx[i] + dx[i-1]
            w2 = dx[i] + 2*dx[i-1]
            m[i] = (w1 + w2) / (w1/d[i-1] + w2/d[i])
    ds_dx = np.gradient(s_a, x, edge_order=1)
    return np.stack([ds_dx, m*ds_dx], axis=1)

# ---- PCHIP-like evaluation for y(s), linear x(s) to preserve monotone H ----
def _pchip_slopes(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float); y = np.asarray(y, float)
    n = len(s)
    m = np.zeros(n, float)
    ds = np.diff(s); dy = np.diff(y)
    d = dy / ds
    m[0] = d[0]; m[-1] = d[-1]
    for i in range(1, n-1):
        if d[i-1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2*ds[i] + ds[i-1]
            w2 = ds[i] + 2*ds[i-1]
            m[i] = (w1 + w2) / (w1/d[i-1] + w2/d[i])
    return m

def _pchip_eval(s: np.ndarray, y: np.ndarray, m: np.ndarray, s_eval: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float); y = np.asarray(y, float); m = np.asarray(m, float)
    s_eval = np.asarray(s_eval, float)
    idx = np.searchsorted(s, s_eval, side="right") - 1
    idx = np.clip(idx, 0, len(s)-2)
    s0 = s[idx]; s1 = s[idx+1]
    y0 = y[idx]; y1 = y[idx+1]
    m0 = m[idx]; m1 = m[idx+1]
    h = (s_eval - s0) / (s1 - s0 + 1e-15)
    h2 = h*h; h3 = h2*h
    H00 = 2*h3 - 3*h2 + 1
    H10 = h3 - 2*h2 + h
    H01 = -2*h3 + 3*h2
    H11 = h3 - h2
    return H00*y0 + H10*(s1 - s0)*m0 + H01*y1 + H11*(s1 - s0)*m1

def _dense_candidate_curve(x: np.ndarray, y: np.ndarray, oversample: int = 40):
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx, dy = np.diff(x), np.diff(y); ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s = np.linspace(0, 1, len(x)) if (s[-1] == 0) else (s / s[-1])
    n = len(s)
    n_f = max(8*n, int(len(s) * oversample))
    s_f = np.linspace(0.0, 1.0, int(n_f))
    x_f = np.interp(s_f, s, x)               # monotone H
    m = _pchip_slopes(s, y)
    y_f = _pchip_eval(s, y, m, s_f)          # smooth R(H)
    return s_f, x_f, y_f

def _tangents_from_candidate(s_f: np.ndarray, x_f: np.ndarray, y_f: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    dx_ds = np.gradient(x_f, s_f, edge_order=1)
    dy_ds = np.gradient(y_f, s_f, edge_order=1)
    return np.stack([dx_ds[keep_idx], dy_ds[keep_idx]], axis=1)

# ---- Tetration-like hybrid growth kernel (optional) ----
def _tetration_unit(z: np.ndarray, height: int) -> np.ndarray:
    """Iterated exponentiation on z in [0,1], numerically guarded."""
    z = np.clip(z, 0.0, 1.0) + 1e-15
    out = z.copy()
    h = int(max(1, height))
    for _ in range(h - 1):
        out = np.power(z, np.clip(out, 0.0, 1.0))
    return out

def _tetration_series(z: np.ndarray, terms: int) -> np.ndarray:
    s = np.zeros_like(z)
    T = int(max(1, terms))
    for k in range(1, T + 1):
        s += z**k / math.factorial(k)
    return s

def _hybrid_bump(n_points: int, mode: str, height: int, series_terms: int) -> np.ndarray:
    sc = np.linspace(0.0, 1.0, int(max(2, n_points)))
    if mode == "direct":
        b = _tetration_unit(sc, height)
    elif mode == "series":
        b = _tetration_series(sc, series_terms)
    else:  # "log" (stable default)
        b = np.log(_tetration_unit(sc, height) + 1.0)
    # zero-mean and Hann window to pin endpoints: bump(0)=bump(1)=0
    b = b - b.mean()
    w = 0.5 - 0.5*np.cos(2*np.pi*sc)
    return b * w

# ---- High-resolution OBA fitter (candidate tangents + packing) ----
def _map_percentile_to_hparams(p: float):
    a = float(np.clip(p, 0.0, 100.0)) / 100.0
    return dict(
        r_base=float(np.interp(a, [0, 1], [0.01, 0.08])),
        r_min_floor=float(np.interp(a, [0, 1], [1e-7, 5e-7])),
        r_shrink_max=float(np.interp(a, [0, 1], [0.97, 0.999])),
        r_power=float(np.interp(a, [0, 1], [4.0, 8.0])),
        smooth_window=int(round(np.interp(a, [0, 1], [7, 13]))),
        num_seg_per_bezier=int(round(np.interp(a, [0, 1], [260, 360]))),
    )

def oba_fit_branch_highres_follow(
    x: np.ndarray,
    y: np.ndarray,
    cluster_percentile: float = 10.0,
    tangent_source: str = "candidate",
    oversample: int = 12,
    max_anchors: int = 400,
    densify_iters: int = 28,
    densify_top_frac: float = 0.95,
    packing_scale: float = 0.25,
    # --- tetration bump controls ---
    tetra_enabled: bool = False,
    tetra_mode: str = "log",
    tetra_height: int = 3,
    tetra_series_terms: int = 5,
    tetra_scale: float = 0.02,
    tetra_axis: str = "y",
):
    # Dense candidate curve, independent of raw sampling
    s_f, x_f, y_f = _dense_candidate_curve(x, y, oversample=max(4, int(oversample)))
    hp = _map_percentile_to_hparams(cluster_percentile)

    # Curvature & weight on dense curve
    kappa = _smooth_1d(_parametric_curvature(s_f, x_f, y_f), hp["smooth_window"])
    kappa = np.maximum(kappa, 0.0)
    thr = np.percentile(kappa, float(cluster_percentile))
    kmax = float(kappa.max()) if kappa.size else 1.0
    w = np.clip((kappa - thr) / (kmax - thr + 1e-15), 0.0, 1.0)

    # Pass 1: variable-radius greedy
    r_local = hp["r_base"] * (1.0 - hp["r_shrink_max"] * (w ** hp["r_power"]))
    r_local = np.maximum(r_local, hp["r_min_floor"]) * float(max(1e-6, packing_scale))

    order = np.argsort(-w).astype(int)
    order = order[(order >= 0) & (order < len(s_f))]

    keep = [0, len(s_f)-1]
    kept = np.zeros(len(s_f), bool); kept[0] = kept[-1] = True

    def _too_close(i: int) -> bool:
        for j in np.where(kept)[0]:
            if abs(s_f[i] - s_f[j]) < min(r_local[i], r_local[j]):
                return True
        return False

    for i in order:
        if kept[i]:
            continue
        if not _too_close(i):
            keep.append(i); kept[i] = True
        if len(keep) >= max_anchors:
            break

    keep.sort()
    keep = np.array(keep, int)

    # Pass 2: densify in highest curvature-mass gaps
    target_top = int(densify_top_frac * len(keep))
    for _ in range(int(densify_iters)):
        thr95 = np.percentile(kappa, 95.0)
        n_top = np.count_nonzero(kappa[keep] >= thr95)
        if n_top >= target_top or len(keep) >= max_anchors:
            break
        best_gain = 0.0; best_pos = None; best_insert = None
        for a_idx in range(len(keep)-1):
            i, j = keep[a_idx], keep[a_idx+1]
            if j <= i + 1:
                continue
            window = slice(i+1, j)
            mass = _trapz(w[window], s_f[window])
            if mass > best_gain:
                loc = int(np.argmax(kappa[window])) + (i+1)
                best_gain = mass; best_pos = loc; best_insert = a_idx + 1
        if best_pos is None:
            break
        pos = int(best_pos)
        if pos < 0 or pos >= len(s_f):
            continue
        if not _too_close(pos):
            keep = np.insert(keep, best_insert, pos)

    # Anchors from dense curve
    s_a = s_f[keep]
    P_a = np.stack([x_f[keep], y_f[keep]], axis=1)

    # Tangents: from candidate derivatives, or fallback
    if tangent_source == "candidate":
        T = _tangents_from_candidate(s_f, x_f, y_f, keep)
    elif tangent_source == "monotone":
        T = _tangents_monotone(P_a[:,0], P_a[:,1], s_a)
    else:
        T = _tangents_centripetal(P_a, alpha=0.5)

    # Build composite Bézier from anchors
    seg_pts = []
    for i in range(len(s_a)-1):
        seg = _hermite_to_bezier(P_a[i], T[i], P_a[i+1], T[i+1],
                                 s_a[i], s_a[i+1], hp["num_seg_per_bezier"])
        if i > 0: seg = seg[1:]
        seg_pts.append(seg)
    seg_pts = np.vstack(seg_pts) if seg_pts else P_a.copy()

    # Optional tetration-like hybrid growth kernel (endpoint-pinned)
    if tetra_enabled and len(seg_pts) > 2:
        bump = _hybrid_bump(len(seg_pts), tetra_mode, tetra_height, tetra_series_terms)
        if tetra_axis.lower() == "y":
            amp = float(tetra_scale) * max(1e-15, np.ptp(y_f))
            seg_pts[:, 1] = seg_pts[:, 1] + amp * bump
        else:
            amp = float(tetra_scale) * max(1e-15, np.ptp(x_f))
            seg_pts[:, 0] = seg_pts[:, 0] + amp * bump

    return seg_pts[:,0], seg_pts[:,1], P_a, keep, kappa, s_f, hp

# -----------------------------
# 3) Generate, fit, and plot
# -----------------------------
H_UP, R_UP, H_DN, R_DN = major_loop_tmr()

bx_up, by_up, anchors_up, idx_up, kappa_up, s_up, hp_up = oba_fit_branch_highres_follow(
    H_UP, R_UP,
    cluster_percentile=CLUSTER_PERCENTILE,
    tangent_source=TANGENT_SOURCE,
    oversample=CANDIDATE_OVERSAMPLE,
    max_anchors=MAX_ANCHORS_PER_BRANCH,
    densify_iters=DENSIFY_ITERS,
    densify_top_frac=DENSIFY_TOP_FRAC,
    packing_scale=PACKING_SCALE,
    tetra_enabled=TETRA_ENABLED,
    tetra_mode=TETRA_MODE,
    tetra_height=TETRA_HEIGHT,
    tetra_series_terms=TETRA_SERIES_TERMS,
    tetra_scale=TETRA_SCALE,
    tetra_axis=TETRA_AXIS,
)
bx_dn, by_dn, anchors_dn, idx_dn, kappa_dn, s_dn, hp_dn = oba_fit_branch_highres_follow(
    H_DN, R_DN,
    cluster_percentile=CLUSTER_PERCENTILE,
    tangent_source=TANGENT_SOURCE,
    oversample=CANDIDATE_OVERSAMPLE,
    max_anchors=MAX_ANCHORS_PER_BRANCH,
    densify_iters=DENSIFY_ITERS,
    densify_top_frac=DENSIFY_TOP_FRAC,
    packing_scale=PACKING_SCALE,
    tetra_enabled=TETRA_ENABLED,
    tetra_mode=TETRA_MODE,
    tetra_height=TETRA_HEIGHT,
    tetra_series_terms=TETRA_SERIES_TERMS,
    tetra_scale=TETRA_SCALE,
    tetra_axis=TETRA_AXIS,
)

plt.figure(figsize=(10.5, 6.6))
plt.plot(H_UP, R_UP, linewidth=1.6, label="Sweep up: R(H)")
plt.plot(H_DN, R_DN, linewidth=1.6, label="Sweep down: R(H)")
plt.plot(bx_up, by_up, linestyle="--", linewidth=1.4,
         label="OBA (follow) up" + ("" if not TETRA_ENABLED else " + tetration"))
plt.plot(bx_dn, by_dn, linestyle="--", linewidth=1.4,
         label="OBA (follow) down" + ("" if not TETRA_ENABLED else " + tetration"))
plt.scatter(anchors_up[:,0], anchors_up[:,1], s=12, marker="o", label=f"Anchors up (N={len(anchors_up)})")
plt.scatter(anchors_dn[:,0], anchors_dn[:,1], s=12, marker="s", label=f"Anchors down (N={len(anchors_dn)})")
plt.xlabel("Magnetic field H (Oe)")
plt.ylabel("Resistance R(H) [Ω]")
plt.title("MTJ TMR Major Loop - OBA with Candidate-Derivative Tangents & Tight Bend Packing"
          + ("" if not TETRA_ENABLED else " + Tetration-like Growth"))
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()

print("Up anchors:", len(anchors_up), "Down anchors:", len(anchors_dn))
```

<img width="2599" height="1625" alt="image" src="https://github.com/user-attachments/assets/058bcbab-ffdb-40eb-8c7f-ca1f39aefe39" />

---

### Examples of Onri's Bezier Approximation Techniques Applied to RF/MW Resonance Peaks

```
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import math
from scipy.interpolate import interp1d

# Use a preset style from scikit‑rf
rf.stylely()

# ---------------------------
# Build the RF Resonator
# ---------------------------
C = 1e-6  # Capacitance in Farads
L = 1e-9  # Inductance in Henry
R = 30    # Resistance in Ohm
Z0 = 50   # Characteristic impedance in Ohm

freq = rf.Frequency(5, 5.2, npoints=501, unit='MHz')
media = rf.DefinedGammaZ0(frequency=freq, z0=Z0)
rng = np.random.default_rng()
random_d = rng.uniform(-np.pi, np.pi)  # random line length for demo

resonator = (media.line(d=random_d, unit='rad')
             ** media.shunt_inductor(L) ** media.shunt_capacitor(C)
             ** media.shunt(media.resistor(R)**media.short()) ** media.open())

# Extract frequency (MHz) and S_db (dB)
f = freq.f
s_db = resonator.s_db.flatten()

# Normalize frequency to [0, 1] for processing
x_norm = (f - f.min())/(f.max()-f.min())

# ---------------------------
# Anchor Selection & Clustering
# ---------------------------
def get_clustered_anchor_indices(x, y, num_uniform=10, threshold_percentile=50):
    """
    Select uniformly spaced anchor indices combined with extra indices
    in regions where the absolute second derivative (approximate curvature)
    exceeds the given percentile threshold.
    """
    uniform_indices = np.linspace(0, len(x)-1, num=num_uniform, dtype=int)
    first_deriv = np.gradient(y, x)
    second_deriv = np.gradient(first_deriv, x)
    curvature = np.abs(second_deriv)
    threshold = np.percentile(curvature, threshold_percentile)
    extra_indices = np.where(curvature > threshold)[0]
    all_indices = np.sort(np.unique(np.concatenate((uniform_indices, extra_indices))))
    return all_indices

anchor_indices = get_clustered_anchor_indices(x_norm, s_db, num_uniform=10, threshold_percentile=50)
anchors_x = x_norm[anchor_indices]
anchors_y = s_db[anchor_indices]

# ---------------------------
# Compute Derivatives at Anchors
# ---------------------------
def compute_derivatives(x, y):
    """
    Estimate slopes at anchor points using finite differences.
    """
    m = np.zeros_like(y)
    n = len(y)
    for i in range(n):
        if i == 0:
            m[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
        elif i == n - 1:
            m[i] = (y[i] - y[i-1]) / (x[i] - x[i-1])
        else:
            m[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    return m

anchor_slopes = compute_derivatives(anchors_x, anchors_y)

# ---------------------------
# Composite Cubic Bezier from Anchors
# ---------------------------
def bezier_from_anchors(x, y, m, num_seg=50):
    """
    Construct a composite Bezier curve from anchor points.
    Each segment is defined as a cubic Bezier curve determined by endpoints
    and estimated slopes (using a Hermite formulation).
    Returns the composite curve and the control points for each segment.
    """
    curve_x = []
    curve_y = []
    control_points_list = []
    n = len(x)
    for i in range(n - 1):
        x0, y0, m0 = x[i], y[i], m[i]
        x1, y1, m1 = x[i+1], y[i+1], m[i+1]
        dx = x1 - x0
        P0 = np.array([x0, y0])
        P3 = np.array([x1, y1])
        P1 = np.array([x0 + dx/3.0, y0 + (dx/3.0)*m0])
        P2 = np.array([x1 - dx/3.0, y1 - (dx/3.0)*m1])
        control_points_list.append(np.array([P0, P1, P2, P3]))
        t = np.linspace(0, 1, num_seg)
        segment = (np.outer((1-t)**3, P0) + np.outer(3*(1-t)**2*t, P1) +
                   np.outer(3*(1-t)*t**2, P2) + np.outer(t**3, P3))
        if i > 0:
            segment = segment[1:]  # avoid duplicating endpoints
        curve_x.extend(segment[:,0])
        curve_y.extend(segment[:,1])
    return np.array(curve_x), np.array(curve_y), control_points_list

bezier_x_norm, bezier_y, bezier_cps = bezier_from_anchors(anchors_x, anchors_y, anchor_slopes, num_seg=50)
# Map normalized x back to original frequency scale
bezier_x = bezier_x_norm * (f.max() - f.min()) + f.min()

# ---------------------------
# Create Interpolation for the Original RF Resonance
# ---------------------------
orig_interp = interp1d(x_norm, s_db, kind='linear', fill_value="extrapolate")

# ---------------------------
# Tetration & Hybrid Modifications (Reference: Original RF Resonance)
# ---------------------------
def tetration(x, m):
    """
    Compute iterated exponentiation (tetration) of x for m iterations.
    This is a naive implementation.
    """
    result = np.copy(x)
    for _ in range(m - 1):
        result = np.power(x, result)
    return result

def tetration_series(x, m, n_terms=5):
    """
    Approximate tetration with a truncated series expansion.
    """
    s = np.zeros_like(x)
    for k in range(1, n_terms + 1):
        s += x**k / math.factorial(k)
    return s

# Parameters for tetration modifications
m_val = 3    # Number of tetration iterations
c_val = 0.1  # Scaling coefficient for direct summation and series expansion
d_val = 0.1  # Scaling coefficient for logarithmic transformation

# Hybrid functions using the original RF resonance as the reference.
H_direct    = orig_interp(x_norm) + c_val * tetration(x_norm, m_val)
H_recursive = orig_interp(tetration(x_norm, m_val))
H_series    = orig_interp(x_norm) + c_val * tetration_series(x_norm, m_val)
H_log       = orig_interp(x_norm) + d_val * np.log(tetration(x_norm, m_val) + 1e-12)

# ---------------------------
# Plotting the Results
# ---------------------------
plt.figure(figsize=(14, 10))

# (1) Original RF Resonance with Clustered Anchors
plt.subplot(3, 2, 1)
plt.plot(f, s_db, 'k-', label='RF Resonance 
')
plt.scatter(f[anchor_indices], s_db[anchor_indices], color='red', label='Anchors')
plt.xlabel('Frequency (MHz)')
plt.ylabel('
 (dB)')
plt.title('RF Resonance with Clustered Anchors')
plt.legend()

# (2) Composite Bezier Curve Approximation
plt.subplot(3, 2, 2)
plt.plot(bezier_x, bezier_y, 'g-', linewidth=2, label='Composite Bezier')
plt.xlabel('Frequency (MHz)')
plt.ylabel('
 (dB)')
plt.title('Bezier Curve Approximation')
plt.legend()

# (3) Direct Summation Hybrid
plt.subplot(3, 2, 3)
plt.plot(f, s_db, 'k-', label='RF Resonance 
')
plt.plot(f, H_direct, 'orange', label='Direct Summation Hybrid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('H(x)')
plt.title('Direct Summation Hybrid')
plt.legend()

# (4) Recursive Hybridization
plt.subplot(3, 2, 4)
plt.plot(f, s_db, 'k-', label='RF Resonance 
')
plt.plot(f, H_recursive, 'purple', label='Recursive Hybridization')
plt.xlabel('Frequency (MHz)')
plt.ylabel('H(x)')
plt.title('Recursive Hybridization')
plt.legend()

# (5) Series Expansion Hybrid
plt.subplot(3, 2, 5)
plt.plot(f, s_db, 'k-', label='RF Resonance 
')
plt.plot(f, H_series, 'brown', label='Series Expansion Hybrid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('H(x)')
plt.title('Series Expansion Hybrid')
plt.legend()

# (6) Logarithmic Transformation Hybrid
plt.subplot(3, 2, 6)
plt.plot(f, s_db, 'k-', label='RF Resonance 
')
plt.plot(f, H_log, 'magenta', label='Logarithmic Transformation Hybrid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('H(x)')
plt.title('Logarithmic Transformation Hybrid')
plt.legend()

plt.tight_layout()
plt.show()
```

![Untitled](https://github.com/user-attachments/assets/fa2c1a58-0822-46c7-af1d-75b1f3b6f0e1)

```
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d

# ----------------------------
# 1. Define the Computed RF Curve with a Narrow Base and Peak Height 10
# ----------------------------
def rf_curve_narrow_base(x, A=10, x0=1, width=0.005):
    """
    Lorentzian function representing an RF resonance peak with a narrow base.
      y_RF(x) = A / [1 + ((x - x0)/width)**2]
    """
    return A / (1 + ((x - x0)/width)**2)

# Generate x values over a range 10 units wide with the resonance peak at x0 = 1.
x = np.linspace(-4, 6, 501)
y = rf_curve_narrow_base(x)  # A=10, width=0.005

# Normalize x to the range [0, 1] for anchor selection and interpolation.
x_norm = (x - x.min()) / (x.max() - x.min())

# ----------------------------
# 2. Anchor Selection & Clustering Based on Curvature
# ----------------------------
def get_clustered_anchor_indices(x, y, num_uniform=10, threshold_percentile=50):
    """
    Select uniformly spaced anchor indices plus extra indices
    where absolute second derivative > the given percentile threshold.
    """
    uniform_indices = np.linspace(0, len(x) - 1, num=num_uniform, dtype=int)
    first_deriv = np.gradient(y, x)
    second_deriv = np.gradient(first_deriv, x)
    curvature = np.abs(second_deriv)
    threshold = np.percentile(curvature, threshold_percentile)
    extra_indices = np.where(curvature > threshold)[0]
    all_indices = np.sort(np.unique(np.concatenate((uniform_indices, extra_indices))))
    return all_indices

anchor_indices = get_clustered_anchor_indices(x_norm, y, num_uniform=10, threshold_percentile=50)
anchors_x = x_norm[anchor_indices]
anchors_y = y[anchor_indices]

# ----------------------------
# 3. Compute Derivatives at Anchor Points
# ----------------------------
def compute_derivatives(x, y):
    """
    Estimate slopes at anchor points using finite differences.
    """
    m = np.zeros_like(y)
    n = len(y)
    for i in range(n):
        if i == 0:
            m[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
        elif i == n - 1:
            m[i] = (y[i] - y[i-1]) / (x[i] - x[i-1])
        else:
            m[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    return m

anchor_slopes = compute_derivatives(anchors_x, anchors_y)

# ----------------------------
# 4. Composite Cubic Bézier Curve from Anchors (with a tension parameter)
# ----------------------------
def bezier_from_anchors(x, y, m, num_seg=50, tension=0.5):
    """
    Construct a composite cubic Bézier curve from anchor points.

    Each segment is determined by:
      - endpoints  (P0, P3)
      - slope at P0 (m0) and slope at P3 (m1)
      - tension factor: 0 < tension <= 1
        * tension < 1 shortens the handles, producing a 'tighter' curve.
        * tension=1 corresponds to the traditional Hermite approach with dx/3.

    Returns:
      curve_x, curve_y = composite curve arrays
      control_points_list = list of [P0, P1, P2, P3] for each segment
    """
    curve_x = []
    curve_y = []
    control_points_list = []

    n = len(x)
    for i in range(n - 1):
        x0, y0, m0 = x[i],   y[i],   m[i]
        x1, y1, m1 = x[i+1], y[i+1], m[i+1]

        dx = x1 - x0

        # We scale (dx/3) by the tension factor
        handle_len = tension * (dx / 3.0)

        # Control points P1, P2 use these "Hermite-like" formulas:
        P0 = np.array([x0, y0])
        P3 = np.array([x1, y1])
        P1 = np.array([x0 + handle_len, y0 + handle_len * m0])
        P2 = np.array([x1 - handle_len, y1 - handle_len * m1])

        control_points_list.append(np.array([P0, P1, P2, P3]))

        # Evaluate this segment of the Bézier curve
        tvals = np.linspace(0, 1, num_seg)
        segment = ((1 - tvals)**3)[:,None]*P0 \
                  + (3*(1 - tvals)**2 * tvals)[:,None]*P1 \
                  + (3*(1 - tvals) * tvals**2)[:,None]*P2 \
                  + (tvals**3)[:,None]*P3

        # Avoid duplicating boundary points
        if i > 0:
            segment = segment[1:]

        curve_x.extend(segment[:,0])
        curve_y.extend(segment[:,1])

    return np.array(curve_x), np.array(curve_y), control_points_list

# Adjust tension here: try 0.3 .. 1.0
bezier_x_norm, bezier_y, bezier_cps = bezier_from_anchors(anchors_x, anchors_y,
                                                          anchor_slopes,
                                                          num_seg=50,
                                                          tension=0.5)

# Map normalized x back to the original scale.
bezier_x = bezier_x_norm * (x.max() - x.min()) + x.min()

# ----------------------------
# 5. Create Interpolation for the Original RF Resonance
# ----------------------------
orig_interp = interp1d(x_norm, y, kind='linear', fill_value="extrapolate")

# ----------------------------
# 6. Tetration & Hybrid Modifications (Using the Original RF Resonance as Reference)
# ----------------------------
def tetration(x, m):
    """
    Compute iterated exponentiation (tetration) of x for m iterations.
    """
    result = np.copy(x)
    for _ in range(m - 1):
        result = np.power(x, result)
    return result

def tetration_series(x, m, n_terms=5):
    """
    Approximate tetration with a truncated series expansion.
    """
    s = np.zeros_like(x)
    for k in range(1, n_terms + 1):
        s += x**k / math.factorial(k)
    return s

# Parameters for the hybrid modifications
m_val = 3    # Tetration iterations
c_val = 0.1  # Scaling coefficient for direct summation and series expansion
d_val = 0.1  # Scaling coefficient for logarithmic transformation
epsilon = 1e-12

# Hybrid functions
H_direct    = orig_interp(x_norm) + c_val * tetration(x_norm, m_val)
H_recursive = orig_interp(tetration(x_norm, m_val))
H_series    = orig_interp(x_norm) + c_val * tetration_series(x_norm, m_val)
H_log       = orig_interp(x_norm) + d_val * np.log(tetration(x_norm, m_val) + epsilon)

# ----------------------------
# 7. Plotting the Results
# ----------------------------
plt.figure(figsize=(14, 10))

# (1) Original RF Resonance with Clustered Anchors
plt.subplot(3, 2, 1)
plt.plot(x, y, 'k-', label='RF Resonance (A=10)')
plt.scatter(x[anchor_indices], y[anchor_indices], color='red', label='Anchors')
plt.xlabel('x')
plt.ylabel('y')
plt.title('RF Resonance with Clustered Anchors')
plt.legend()

# (2) Composite Bézier Curve Approximation (with tension)
plt.subplot(3, 2, 2)
plt.plot(x, y, 'k--', alpha=0.3, label='Original')
plt.plot(bezier_x, bezier_y, 'g-', linewidth=2, label='Composite Bézier')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bézier Curve Approx. (tension=0.5)')
plt.legend()

# (3) Direct Summation Hybrid
plt.subplot(3, 2, 3)
plt.plot(x, y, 'k-', label='RF Resonance (A=10)')
plt.plot(x, H_direct, 'orange', label='Direct Summation Hybrid')
plt.xlabel('x')
plt.ylabel('H(x)')
plt.title('Direct Summation Hybrid')
plt.legend()

# (4) Recursive Hybridization
plt.subplot(3, 2, 4)
plt.plot(x, y, 'k-', label='RF Resonance (A=10)')
plt.plot(x, H_recursive, 'purple', label='Recursive Hybridization')
plt.xlabel('x')
plt.ylabel('H(x)')
plt.title('Recursive Hybridization')
plt.legend()

# (5) Series Expansion Hybrid
plt.subplot(3, 2, 5)
plt.plot(x, y, 'k-', label='RF Resonance (A=10)')
plt.plot(x, H_series, 'brown', label='Series Expansion Hybrid')
plt.xlabel('x')
plt.ylabel('H(x)')
plt.title('Series Expansion Hybrid')
plt.legend()

# (6) Logarithmic Transformation Hybrid
plt.subplot(3, 2, 6)
plt.plot(x, y, 'k-', label='RF Resonance (A=10)')
plt.plot(x, H_log, 'magenta', label='Logarithmic Transformation Hybrid')
plt.xlabel('x')
plt.ylabel('H(x)')
plt.title('Logarithmic Transformation Hybrid')
plt.legend()

plt.tight_layout()
plt.show()
```

![Untitled](https://github.com/user-attachments/assets/9536c52f-3879-444d-b528-ae0a6e551270)

---

## Citing OBA
```
@software{benally_oba,
  author  = {Benally, Onri Jay},
  title   = {Onri's Bézier Approximation (OBA)},
  year    = {2025},
  url     = {https://github.com/OJB-Quantum/Onri-Bezier-Approximation},
  note    = {Hybrid Bézier+tetration, curvature-aware anchor clustering, post-plot overlays}
}
```
