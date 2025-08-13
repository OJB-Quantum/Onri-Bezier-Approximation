# Bezier-Approximation-Plus - Onri's Bezier Approximation (OBA) 
Applied ideas on using Bezier curves &amp; hybrid tetrational-polynomials to fit or approximate curves in data of virtually any magnitude. Authored by Onri Jay Benally.

[![License](https://img.shields.io/badge/Creative_Commons-License-green)](https://choosealicense.com/licenses/gpl-3.0)

Primary URL for the repository: [OJB-Quantum/Bezier-Approximation-Plus](https://github.com/OJB-Quantum/Bezier-Approximation-Plus)

View an interactive tool in the browser the demonstrate Bezier approximation: [OBA Demos](https://g.co/gemini/share/35aaa5180fc9)

Basic Bezier curves, being the useful geometric tools that they are, can be described by a Bernstein basis polynomial. They can be adapted to follow objects that bend using hidden control handles and anchor points placed along an existing curve or virtual contour of interest, as shown in this repository. With that in mind, I thought of adapting a polynomial for the Bezier curve with tetrations or super exponentials to form a hybrid approach that compensates for very sharp and large changes in data curves. It does so by mathematically describing the anchor points and control points of a Bezier curve, as well as where they are located in some data plotting space or layout, how dense the clusters of anchor points are as determined by a given threshold, and how large a tetration or super exponential should be according to the size distance between the smallest and largest values of interest locally or globally in order to move anchor points and control points to where they need to be. 

Note that integrating a tetration into a polynomial can create extremely large values, which cannot be represented on any 64-bit or 128-bit computer. Thus, the tetration must be carefully expressed to stay within the compatibility or capability of a 128-bit or 64-bit machine. Some interesting results are provided in this repository applied to real use cases using adjustable clustering of Bezier anchor points, such as approximating electronic band structures for example. Check files for the code provided in a Google Colab notebook.

### Click here to render the notebooks in the browser: [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/OJB-Quantum/Bezier-Approximation-Plus/tree/main/)

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
│  └─ Control vs Anchor semantics
├─ Growth boosters
│  ├─ Tetration  ← super‑exponential tower
│  │  ├─ Log‑damped variant
│  │  └─ Series‑expanded variant
│  └─ Future kernels (pentation, iterated sine)
├─ Physics use‑cases
│  ├─ Electronic band diagrams
│  ├─ RF/ microwave resonance envelopes
│  ├─ Fluid contour streamlines
│  └─ Quantum‑well potential profiles
└─ Agnostic engine
   ├─ Dimensionless normalisation
   ├─ Plugin constraint JSON
   ├─ Error‑adaptive anchor insertion
   └─ GPU/ SIMD parallel evaluation
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
