# Bezier-Approximation-Plus
Applied ideas on using Bezier curves &amp; hybrid tetrations to approximate &amp; fit curves in data. Authored by Onri Jay Benally.

Bezier curves, being the useful geometric tools that they are, can be described by a Bernstein basis polynomial. They can be modified to follow objects that bend using hidden control handles and anchor points placed on an existing curve of interest. With that in mind, I thought of the introducing the polynomial for the Bezier curve with a tetration to form a hybrid approach which may compensate for very sharp and large changes in curve data as well. Some interesting results are provided in this repository. 

![ezgif-418d14bce1cd40](https://github.com/user-attachments/assets/dd806438-3021-4664-bea7-432d8a6186e3)


# Defining What a Bezier Curve is Doing Mathematically

| **Concept**                        | **Equation / Explanation**                                                                                                                                       |
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

| **Concept**                   | **Equation / Explanation** |
|-------------------------------|----------------------------|
| **Polynomial Definition**     | A polynomial consists of a series of terms involving powers of a variable, typically expressed as: <br />$P(x) = a_n x^n + a_{n-1} x^{n-1} + \dots + a_1 x + a_0$<br />where the exponents are **added** sequentially. |
| **Tetration Definition**      | Tetration is a form of repeated exponentiation, written as: <br />$^n a = a^{a^{a^{\dots}}}$<br />where the exponentiation **stacks** instead of adding. |
| **Comparison to a Series**    | - A **series** consists of a sum of terms.<br />- A **polynomial** is a finite sum of powers of $x$.<br />- **Exponentiation** is an iterative **multiplication** operation.<br />- **Tetration** is an iterative **exponentiation** operation. |
| **Growth Difference**         | Unlike a polynomial, a **tetration does not consist of a sum of terms**; instead, it is an **iterated power tower**, which grows much faster. |
| **Can Tetration Be Expressed as a Series?** | Tetration does not naturally expand into a power series like a polynomial. However, in some cases, it can be approximated using: <br />- **Logarithmic expansions** (breaking it down via $a^{a^{a^x}}$).<br />- **Power series representations** (like Taylor series) for small values.<br />But in general, **tetration does not behave like a polynomial series** because it is based on hierarchical exponentiation rather than summation. |

---

| **Concept**                                     | **Equation / Explanation** |
|-------------------------------------------------|----------------------------|
| **Hybrid Polynomial-Tetration Possibility**    | A **polynomial power series** can be **appended or modified as a hybrid with tetration**, depending on how the two mathematical structures are combined. |
| **1. Direct Summation (Appending a Tetration Term)** | A tetration term is added to a polynomial power series: <br />$H(x) = \sum_{n=0}^{\infty} a_n x^n + c \cdot {}^m x$<br />where: <br />- $\sum_{n=0}^{\infty} a_n x^n$ is a traditional polynomial or power series,<br />- ${}^m x$ is the **tetration term**,<br />- $c$ is a scaling coefficient.<br />**Blends polynomial growth with tetration's extreme growth.** |
| **2. Recursive Hybridization (Tetration Within a Polynomial)** | Instead of adding tetration separately, we **embed** it into the polynomial: <br />$H(x) = a_n ({}^m x)^n + a_{n-1} ({}^m x)^{n-1} + \dots + a_1 ({}^m x) + a_0$<br />**Amplifies the polynomial’s growth through tetration.** |
| **3. Series Expansion Involving Tetration (Power Series Approximation)** | For small $x$, tetration can be approximated using a **Taylor or power series expansion**: <br />${}^m x = e^{x + x^2 + \frac{x^3}{3} + \dots}$<br />This allows for: <br />$H(x) = \sum_{n=0}^{\infty} b_n ({}^m x)^n$<br />where $b_n$ are coefficients to **moderate tetration’s extreme growth.** |
| **4. Logarithmic Transformation (Taming Tetration Growth)** | To prevent tetration from **dominating** a polynomial, we introduce logarithmic damping: <br />$H(x) = \sum_{n=0}^{\infty} a_n x^n + d \log({}^m x)$<br />**Controls tetration's rapid growth by applying a logarithm.** |
| **Challenges of Hybridizing a Polynomial with Tetration** | 1. **Growth Rate Disparity**: Tetration grows **much** faster than polynomial terms. Scaling is necessary. <br />2. **Analytic Continuation Issues**: Tetration is **not always well-defined** for non-integer heights, requiring **super-exponential extensions**. <br />3. **Computational Stability**: Tetration grows **hyper-exponentially**, which can cause **numerical instability**. |
| **Conclusion** | A **hybrid polynomial-tetration function** is possible with different formulations depending on the desired properties: <br />- **Controlled growth**: Use logarithmic damping or power series approximations.<br />- **Ultra-fast growth**: Use direct summation or embed tetration inside a polynomial. |

