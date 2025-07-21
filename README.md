# Bezier-Approximation-Plus - Onri's Bezier Approximation (OBA)
Applied ideas on using Bezier curves &amp; hybrid tetrational-polynomials to fit or approximate curves in data of virtually any magnitude. Authored by Onri Jay Benally.

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

### Why OBA yields *highly accurate* physics formulas

| Property                        | Mathematical reason                                                                                                               | Physical consequence                                                                                 |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Piecewise analytic fidelity** | Bernstein polynomials form a *partition of unity* ⇢ local control without Gibbs ringing.                                          | Spectral‑line fits, dispersion curves, and smoothly varying potentials keep experimental continuity. |
| **Adaptivity across scales**    | Tetration term raises dynamic range from polynomial $O(1)$ to super‑exponential yet *scalable* $O(e^{e^{\cdot}})$.                | Same template fits millikelvin noise floors and tera‑kelvin stellar flares.                          |
| **Derivative steering**         | $d^kB/dt^k$ is again a Bézier of degree $n-k$; anchor clustering matches measured $\partial^k f/\partial x^k$.                    | Curvature constraints (e.g., zero‑slope boundary at mirror center) encoded directly.                 |
| **Coordinate insensitivity**    | Control points live in normalized $(u,v,w)$ axes; re‑map via any smooth bijection $x(x'),y(y')$.                                  | Works identically for momentum space, real space, or log‑frequency charts.                           |
| **Computational stability**     | Convex‑hull and de Casteljau subdivision guarantee floating‑point safety; tetration damped by $\log$ or series if overflow looms. | Robust on 64‑bit GPUs; no catastrophic cancellation when plotting band structures.                   |

---

### How *agnostic* the OBA framework already is

Because it treats every target merely as a *curve in a metric space*, OBA never asks *what* the ordinate represents—charge, entropy, or fluid height—only *where* the sampled points lie.

* **Unit agnosticism** –  All coordinates enter after non‑dimensionalisation $x\mapsto(x-x_0)/\Delta x$.
* **Domain‑agnostic handles** –  Control‑point density derives from a scale‑free curvature metric

  $$\kappa(t)=\frac{|B'(t)\times B''(t)|}{|B'(t)|^{3}},$$

  so identical logic handles cosmological red‑shift curves or nanosecond pulse edges.
* **Data‑source agnosticism** –  Anchor points arise from either analytic formulas, PDE solvers, or raw lab CSV files.

---

### Steps to rewrite OBA into an *even more agnostic, adaptive* description

1. **Embed dimensionless sampling** –  Replace absolute $t$ with a cumulative arc‑length parameter $s\in[0,1]$; now geometry, not original grid, controls spacing.
2. **Abstract the growth kernel** –  Generalise the special tetration to a placeholder $\mathcal{G}(t;\theta)$ satisfying

   $$
   \lim_{\theta\to0}\mathcal{G}=0,\qquad
   \partial \mathcal{G}/\partial\theta>0,
   $$

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

### Mind‑map of connections

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

### Portmanteaus and etymologies worth noting

| Term          | Origin                                                          | Note                                                                   |
| ------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Bézier**    | Pierre Bézier, French engineer                                  | Popularized cubic curves for Renault in the 1960 s.                    |
| **Tetration** | *tetra* (four) + *iteration*                                    | Fourth hyper‑operation after addition, multiplication, exponentiation. |
| **Pentation** | Future *penta* (five) hyper‑operation; candidate growth kernel. |                                                                        |
| **OBA**       | Onri’s Bézier Approximation                                     | Combines geometric Bézier with analytic boosters.                      |


---

### Examples of the Hybrid Bezier Approximation Applied to RF/MW Resonance Peaks

![Untitled](https://github.com/user-attachments/assets/fa2c1a58-0822-46c7-af1d-75b1f3b6f0e1)

![Untitled](https://github.com/user-attachments/assets/9536c52f-3879-444d-b528-ae0a6e551270)

