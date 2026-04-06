"""
kernel_theory.py
----------------
All theory cards render as self-contained HTML/KaTeX components.

Key design decision: NO fixed heights anywhere.
After KaTeX renders, a ResizeObserver measures the card's actual
pixel height and posts it to the parent frame so Streamlit resizes
the iframe to exactly fit — content is never clipped.

Formulas are passed via JSON-encoded JS variables so backslashes
and quotes in LaTeX strings are always safe.
Inline $...$ math uses KaTeX auto-render.

The polynomial section is fully dynamic: every formula, example,
and dimension count is computed from the `degree` parameter.
"""
from __future__ import annotations
import json
import math
from itertools import combinations_with_replacement
import streamlit as st
import streamlit.components.v1 as components


_ACCENT = {
    "blue":   "#38bdf8",
    "purple": "#a855f7",
    "green":  "#4ade80",
    "red":    "#f87171",
}

# ─────────────────────────────────────────────────────────────
# Auto-sizing JS snippet — measures rendered height and resizes iframe
# ─────────────────────────────────────────────────────────────
# We post the scrollHeight of <body> upward after a short settling
# delay (fonts + KaTeX rendering). A ResizeObserver fires again if
# the DOM grows further (e.g. after auto-render inserts math nodes).
_AUTORESIZE_JS = """
function postHeight() {
  var h = document.body.scrollHeight + 8;
  window.parent.postMessage({type:"streamlit:setFrameHeight", height:h}, "*");
}
// Fire once immediately, then again after KaTeX settles
setTimeout(postHeight, 50);
setTimeout(postHeight, 300);
setTimeout(postHeight, 800);
var ro = new ResizeObserver(postHeight);
ro.observe(document.body);
"""


# ─────────────────────────────────────────────────────────────
# Polynomial helpers — all derived from degree, never hardcoded
# ─────────────────────────────────────────────────────────────

def _poly_dim(degree: int, n_input: int = 2) -> int:
    """Number of monomials up to `degree` in `n_input` variables (with bias)."""
    return sum(math.comb(n_input + d - 1, d) for d in range(degree + 1))


def _poly_monomials(degree: int, n_input: int = 2) -> list[str]:
    """
    LaTeX terms for the normalised feature map φ(x) so that
    φ(x)·φ(x') = (x·x' + 1)^degree exactly.
    """
    vars_ = [f"x_{i+1}" for i in range(n_input)]
    terms: list[str] = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(n_input), d):
            counts = [combo.count(i) for i in range(n_input)]
            multinomial = math.factorial(d)
            for c in counts:
                multinomial //= math.factorial(c)
            coeff = math.sqrt(multinomial)

            if d == 0:
                monomial = "1"
            else:
                parts = []
                for i in range(n_input):
                    if counts[i] == 1:
                        parts.append(vars_[i])
                    elif counts[i] > 1:
                        parts.append(f"{vars_[i]}^{{{counts[i]}}}")
                monomial = "".join(parts)

            if coeff == 1.0:
                terms.append(monomial)
            elif coeff == int(coeff):
                terms.append(f"{int(coeff)}\\,{monomial}")
            else:
                sq = int(round(coeff ** 2))
                terms.append(f"\\sqrt{{{sq}}}\\,{monomial}")
    return terms


def _poly_feature_map_latex(degree: int) -> str:
    terms = _poly_monomials(degree)
    inner = ",\\;".join(terms)
    return rf"\phi(x_1,x_2) = \bigl[{inner}\bigr]^\top"


def _poly_interaction_examples(degree: int) -> str:
    lines = []
    for d in range(2, degree + 1):
        cross_terms = []
        for combo in combinations_with_replacement(range(2), d):
            counts = [combo.count(i) for i in range(2)]
            if all(c < d for c in counts):
                parts = []
                for i, c in enumerate(counts):
                    if c == 1:
                        parts.append(f"x_{i+1}")
                    elif c > 1:
                        parts.append(f"x_{i+1}^{c}")
                cross_terms.append("".join(parts))
        if cross_terms:
            terms_str = ", ".join(f"${t}$" for t in cross_terms)
            lines.append(f"Degree {d} adds: {terms_str}")
    return " — ".join(lines) if lines else ""


def _poly_numeric_example(degree: int) -> str:
    x, xp = [1, 2], [2, 1]
    dot = sum(a * b for a, b in zip(x, xp))
    k_val = (dot + 1) ** degree
    dim = _poly_dim(degree)
    return (
        f"For $x = [1,2]$ and $x' = [2,1]$: "
        f"$x^Tx' = {dot}$, so "
        f"$K(x,x') = ({dot}+1)^{{{degree}}} = {k_val}$. "
        f"Computing $\\phi(x)^T\\phi(x')$ in all {dim} dimensions gives the same result — "
        f"the kernel shortcut saves all that work."
    )


# ─────────────────────────────────────────────────────────────
# Core card — NO fixed height, auto-resizes to content
# ─────────────────────────────────────────────────────────────

def _katex_card(
    *,
    concept: str,
    formulas: list[str] | None = None,
    intuition: str | None = None,
    accent: str = "blue",
) -> None:
    """
    Render a self-contained KaTeX theory card.
    The iframe height is set automatically by measuring the rendered DOM —
    content is never clipped regardless of text length or formula count.
    """
    color = _ACCENT.get(accent, _ACCENT["blue"])
    formulas_json = json.dumps(formulas or [])

    formula_section = ""
    if formulas:
        formula_section = """
        <div class="section formula-section">
          <div class="label">Formula</div>
          <div id="formula-rows" class="formula-rows"></div>
        </div>"""

    intuition_section = ""
    if intuition:
        intuition_section = f"""
        <div class="section intuition-section">
          <div class="label">Intuition</div>
          <div id="intuition-body" class="body">{intuition}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"
        onload="window._katexReady=true"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
        onload="window._arReady=true"></script>
<style>
  html,body{{margin:0;padding:0;box-sizing:border-box}}
  body{{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    background:transparent;
    padding:4px 2px 10px;
    color:#e2e8f0;
    /* Let the body grow to content — never clip */
    overflow:hidden;
  }}
  *{{box-sizing:border-box}}
  .card{{
    border:1.5px solid {color};
    border-radius:16px;
    background:rgba(255,255,255,0.02);
  }}
  .section{{padding:14px 20px}}
  .section+.section{{border-top:1px solid rgba(255,255,255,0.08)}}
  .label{{
    font-size:.66rem;letter-spacing:.1em;font-weight:700;
    text-transform:uppercase;color:#64748b;margin-bottom:8px;
  }}
  .body{{font-size:.94rem;line-height:1.65;color:#cbd5e1}}
  .concept-section .body{{color:#e2e8f0}}
  .formula-section{{background:rgba(0,0,0,0.2)}}
  .formula-rows{{
    display:flex;flex-direction:column;
    align-items:center;gap:12px;padding:8px 0 6px;
  }}
  .formula-row{{color:#f1f5f9}}
  .formula-row .katex{{color:#f1f5f9;font-size:1.05rem}}
  .intuition-section .body{{color:#94a3b8}}
</style>
</head>
<body>
<div class="card">
  <div class="section concept-section">
    <div class="label">Concept</div>
    <div class="body">{concept}</div>
  </div>
  {formula_section}
  {intuition_section}
</div>

<script>
var FORMULAS = {formulas_json};
var AR_OPTS = {{
  delimiters:[
    {{left:"$$",right:"$$",display:true}},
    {{left:"$",right:"$",display:false}},
    {{left:"\\\\(",right:"\\\\)",display:false}},
    {{left:"\\\\[",right:"\\\\]",display:true}}
  ],
  throwOnError:false
}};

function postHeight(){{
  var h = document.body.scrollHeight + 8;
  window.parent.postMessage({{type:"streamlit:setFrameHeight",height:h}},"*");
}}

function renderAll(){{
  /* Display formulas */
  var cont = document.getElementById("formula-rows");
  if(cont){{
    FORMULAS.forEach(function(latex){{
      var d = document.createElement("div");
      d.className = "formula-row";
      cont.appendChild(d);
      try{{ katex.render(latex, d, {{displayMode:true, throwOnError:false}}); }}
      catch(e){{ d.textContent = latex; }}
    }});
  }}
  /* Inline math in concept + intuition */
  [document.querySelector(".concept-section .body"),
   document.getElementById("intuition-body")].forEach(function(el){{
    if(el && window.renderMathInElement)
      renderMathInElement(el, AR_OPTS);
  }});
  /* Resize after render settles */
  postHeight();
  setTimeout(postHeight, 150);
  setTimeout(postHeight, 500);
}}

/* Auto-resize on any DOM change */
var ro = new ResizeObserver(postHeight);
ro.observe(document.body);

/* Wait for both KaTeX scripts then render */
(function wait(n){{
  if(window._katexReady && window._arReady){{ renderAll(); }}
  else if(n < 80){{ setTimeout(function(){{wait(n+1);}}, 80); }}
}})(0);
</script>
</body>
</html>"""

    # Start with a generous initial height — the postMessage will shrink/grow it
    # to the exact content height once KaTeX finishes rendering.
    components.html(html, height=480, scrolling=False)


# ─────────────────────────────────────────────────────────────
# Kernel choice card — also auto-sizing
# ─────────────────────────────────────────────────────────────

def _kernel_choice_card(*, name, color, formula, use_when, avoid):
    bullets = "".join(f"<li>{w}</li>" for w in use_when)
    formula_json = json.dumps(formula)
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"
        onload="window._kr=true"></script>
<style>
  html,body{{margin:0;padding:0;box-sizing:border-box;overflow:hidden}}
  body{{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    background:transparent;padding:2px 2px 8px;
  }}
  *{{box-sizing:border-box}}
  .card{{
    border:1.5px solid {color};border-radius:14px;
    background:rgba(255,255,255,0.02);
    display:grid;grid-template-columns:1fr 1fr;
  }}
  .left{{
    padding:14px 16px;border-right:1px solid rgba(255,255,255,0.08);
    display:flex;flex-direction:column;gap:8px;
  }}
  .right{{padding:14px 16px}}
  .name{{font-size:1rem;font-weight:800;color:{color}}}
  #fw{{display:flex;justify-content:center;color:#f1f5f9}}
  #fw .katex{{color:#f1f5f9;font-size:.92rem}}
  .label{{
    font-size:.65rem;letter-spacing:.1em;font-weight:700;
    text-transform:uppercase;color:#64748b;margin-bottom:6px;
  }}
  ul{{padding-left:16px;color:#cbd5e1;font-size:.88rem;line-height:1.6}}
  .avoid{{font-size:.82rem;color:#f87171;margin-top:8px}}
</style>
</head>
<body>
<div class="card">
  <div class="left">
    <div class="name">{name}</div>
    <div id="fw"></div>
  </div>
  <div class="right">
    <div class="label">Use when</div>
    <ul>{bullets}</ul>
    <div class="avoid">⚠ Avoid: {avoid}</div>
  </div>
</div>
<script>
function postHeight(){{
  var h = document.body.scrollHeight + 8;
  window.parent.postMessage({{type:"streamlit:setFrameHeight",height:h}},"*");
}}
var ro = new ResizeObserver(postHeight);
ro.observe(document.body);
(function wait(n){{
  if(window._kr){{
    katex.render({formula_json}, document.getElementById("fw"),
      {{displayMode:true, throwOnError:false}});
    postHeight();
    setTimeout(postHeight, 200);
  }} else if(n < 50){{ setTimeout(function(){{wait(n+1);}}, 80); }}
}})(0);
</script>
</body>
</html>"""
    components.html(html, height=200, scrolling=False)


# ─────────────────────────────────────────────────────────────
# Public expander render functions
# ─────────────────────────────────────────────────────────────

def render_linear_separability_theory():
    with st.expander("📖 Linear separability — theory & math", expanded=False):
        _katex_card(
            concept="A dataset is <strong>linearly separable</strong> if a hyperplane exists that perfectly divides every class. In 2D this is a straight line; in higher dimensions it is a flat surface.",
            formulas=[r"w^T x + b = 0", r"y_i\,(w^T x_i + b)>0\quad\forall\,i"],
            intuition="Concentric circles are the classic counter-example: the inner class is surrounded by the outer class, so no straight line can ever split them. This is the problem the kernel trick solves.",
            accent="blue",
        )


def render_feature_map_theory():
    with st.expander("📖 Feature map Φ(x) — theory & math", expanded=False):
        _katex_card(
            concept="The kernel trick maps each input from a low-dimensional space into a richer feature space where a <strong>linear</strong> hyperplane may now separate the classes — without ever computing the mapping explicitly.",
            formulas=[r"\phi:\mathbb{R}^d\to\mathbb{R}^D\quad(D\gg d)", r"K(x,\,x')=\phi(x)^T\phi(x')"],
            intuition="For concentric circles in 2D, adding a third coordinate $z = x_1^2 + x_2^2$ lifts the inner circle to a lower height and the outer ring higher. A flat horizontal plane can then separate them cleanly.",
            accent="purple",
        )


def render_infinite_dimension_theory():
    with st.expander("📖 Why some kernels map to infinite dimensions", expanded=False):
        _katex_card(
            concept="The RBF / Gaussian kernel corresponds to an <strong>infinitely large</strong> feature space — yet we never materialise that space. The kernel function computes the inner product directly, bypassing the need to enumerate every dimension.",
            formulas=[r"K(x,\,x')=\exp\!\left(-\gamma\,\|x-x'\|^2\right)"],
            intuition="Expanding $\\exp(-\\gamma\\|x-x'\\|^2)$ as an infinite Taylor series gives one term per polynomial degree — the kernel sums them all in a single evaluation.",
            accent="green",
        )


def render_mercer_theory():
    with st.expander("📖 Mercer's theorem — validity of kernels", expanded=False):
        _katex_card(
            concept="Not every function $K(x,x')$ is a valid kernel. Mercer's theorem states $K$ is valid if and only if the Gram matrix it produces is <strong>positive semidefinite</strong> for any finite set of inputs.",
            formulas=[r"G_{ij}=K(x_i,\,x_j)", r"G\succeq 0"],
            intuition="If the kernel violates this — as Sigmoid sometimes does — the optimisation may be non-convex and the model can behave unpredictably. Linear, Polynomial, and RBF always satisfy Mercer's theorem.",
            accent="blue",
        )


def render_kernel_choice_theory():
    with st.expander("📖 When to use each kernel", expanded=False):
        kernels = [
            {"name": "Linear",        "color": "#38bdf8",
             "formula": r"K(x,x')=x^Tx'",
             "use_when": ["Boundary already close to linear", "High-dimensional sparse data (e.g. text)", "You need interpretability"],
             "avoid": "Classes in rings, spirals, or XOR patterns."},
            {"name": "Polynomial",    "color": "#a855f7",
             "formula": r"K(x,x')=(x^Tx'+c)^d",
             "use_when": ["Feature interactions carry signal (e.g. XOR)", "Curved but structured boundaries", "Image or gene-expression data"],
             "avoid": "High degrees overfit — start with d = 2 or 3."},
            {"name": "RBF / Gaussian","color": "#4ade80",
             "formula": r"K(x,x')=\exp(-\gamma\|x-x'\|^2)",
             "use_when": ["Highly nonlinear boundaries", "Unknown feature interactions", "General-purpose default choice"],
             "avoid": "Very large datasets — kernel matrix becomes expensive."},
            {"name": "Sigmoid",       "color": "#f87171",
             "formula": r"K(x,x')=\tanh(\alpha x^Tx'+c)",
             "use_when": ["Teaching comparison to neural networks", "Mild nonlinearity with neural-net intuition"],
             "avoid": "Not always Mercer-valid. RBF almost always outperforms it."},
        ]
        for k in kernels:
            _kernel_choice_card(**k)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


def render_kernel_pca_bridge_theory():
    with st.expander("📖 Kernel PCA vs standard PCA — theory & math", expanded=False):
        _katex_card(
            concept="Standard PCA finds the directions of <strong>maximum linear variance</strong> in the original feature space. Kernel PCA first applies the kernel trick to implicitly lift the data into a richer space, then performs PCA there.",
            formulas=[r"\tilde{K}_{ij}=K(x_i,x_j)-\tfrac{1}{n}\sum_k K(x_k,x_j)-\tfrac{1}{n}\sum_k K(x_i,x_k)+\tfrac{1}{n^2}\sum_{k,l}K(x_k,x_l)"],
            intuition="If data lies on a curved manifold — like a Swiss roll — standard PCA will cut through it rather than follow it. Kernel PCA with RBF can unfold the manifold because it already sees the geometry in a transformed space.",
            accent="green",
        )


# ─────────────────────────────────────────────────────────────
# Kernel-specific Step 3  (fully dynamic for Polynomial)
# ─────────────────────────────────────────────────────────────

def render_kernel_specific_theory(kernel_name: str, gamma: float = 0.5, degree: int = 2) -> None:

    # ── Linear ──────────────────────────────────────────────
    if kernel_name == "Linear":
        st.markdown("#### How the linear kernel works")
        _katex_card(
            concept="<strong>Linear kernel</strong> computes the standard dot product between inputs with no transformation. Training an SVM with this kernel is equivalent to fitting a standard maximum-margin linear classifier on the original features.",
            formulas=[r"K(x,\,x')=x^Tx'"],
            intuition="When data is already linearly separable, adding nonlinear complexity only hurts generalisation. The linear kernel maximises the margin of the straight-line boundary — simple, fast, and interpretable.",
            accent="blue",
        )
        st.markdown("#### Maximum margin — why SVM is special")
        _katex_card(
            concept="Among all hyperplanes that separate the classes, SVM finds the one with the <strong>largest margin</strong> — the gap between the two closest points from each class (the support vectors).",
            formulas=[r"\text{margin}=\frac{2}{\|w\|}", r"\min_{w,b}\;\tfrac{1}{2}\|w\|^2\quad\text{s.t.}\;y_i(w^Tx_i+b)\geq 1"],
            intuition="A wider margin means the classifier is more confident and generalises better to unseen data. Only the support vectors define the boundary — all other training points are irrelevant.",
            accent="blue",
        )

    # ── Polynomial (fully dynamic) ───────────────────────────
    elif kernel_name == "Polynomial":
        dim = _poly_dim(degree)
        feature_map_latex = _poly_feature_map_latex(degree)
        numeric_example = _poly_numeric_example(degree)
        interaction_str = _poly_interaction_examples(degree)

        # Per-degree plain-English breakdown of feature space contents
        degree_descriptions = []
        for d in range(1, degree + 1):
            n_terms = math.comb(2 + d - 1, d)
            if d == 1:
                degree_descriptions.append(f"degree 1 ({n_terms} terms: $x_1, x_2$)")
            elif d == 2:
                degree_descriptions.append(f"degree 2 ({n_terms} terms: $x_1^2, x_1x_2, x_2^2$)")
            elif d == 3:
                degree_descriptions.append(f"degree 3 ({n_terms} terms: $x_1^3, x_1^2x_2, x_1x_2^2, x_2^3$)")
            else:
                degree_descriptions.append(f"degree {d} ({n_terms} terms)")
        degree_list = "; ".join(degree_descriptions)

        st.markdown(f"#### Degree-{degree} polynomial kernel: what it computes")
        _katex_card(
            concept=(
                f"The degree-{degree} polynomial kernel maps 2 input features into a "
                f"<strong>{dim}-dimensional</strong> feature space. "
                f"It includes: {degree_list}; plus the bias term (1 dimension). "
                f"The kernel evaluates this entire inner product in a single scalar operation."
            ),
            formulas=[
                rf"K(x,\,x') = (x^Tx' + 1)^{{{degree}}}",
                feature_map_latex,
            ],
            intuition=numeric_example,
            accent="purple",
        )

        st.markdown("#### Why XOR needs the cross-term $x_1 x_2$")
        _katex_card(
            concept=(
                "XOR labels are determined entirely by whether $x_1$ and $x_2$ have the <strong>same sign</strong>. "
                "Neither $x_1$ alone nor $x_2$ alone carries that information — only their product does. "
                "A degree-1 (linear) kernel has no cross-terms and cannot separate XOR. "
                "A degree-2 or higher kernel implicitly includes $x_1 x_2$, which is exactly what is needed."
            ),
            formulas=[
                r"x_1 x_2 > 0 \;\Rightarrow\; \text{same sign (class 0)}",
                r"x_1 x_2 < 0 \;\Rightarrow\; \text{different sign (class 1)}",
            ],
            intuition=(
                f"Your current degree {degree} provides the cross-term $x_1 x_2$ automatically."
                + (f" {interaction_str}." if interaction_str else "")
                + " Higher degrees add further cross-terms, giving the boundary more flexibility but also more risk of overfitting on small datasets."
            ),
            accent="purple",
        )

        st.markdown("#### The kernel trick: why we never compute φ(x) explicitly")
        _katex_card(
            concept=(
                f"Instead of computing the {dim}-dimensional vector $\\phi(x)$ and taking its dot product, "
                f"the kernel evaluates $(x^Tx' + 1)^{{{degree}}}$ directly — "
                f"a single scalar operation regardless of whether $\\phi$ has {dim} dimensions or a million."
            ),
            formulas=[
                rf"(x^Tx' + 1)^{{{degree}}} = \phi(x)^T\phi(x') \quad \text{{({dim}\text{{ dims, never computed}})}}"
            ],
            intuition=(
                f"Computing $\\phi(x)^T\\phi(x')$ directly would require {dim} multiplications. "
                f"The kernel evaluates $(x^Tx'+1)^{{{degree}}}$ in just {degree + 1} multiplications via the binomial expansion. "
                "This efficiency gap grows dramatically with degree and input dimension — it is what makes kernel SVMs practical."
            ),
            accent="purple",
        )

        if degree >= 4:
            st.warning(
                f"⚠️ **Degree {degree} is high.** With {dim} implicit dimensions from just 2 input features, "
                "the boundary may become overly complex and overfit the XOR dataset. "
                "Try degree 2 or 3 for a cleaner, more generalisable boundary."
            )
        elif degree == 3:
            st.info(
                f"ℹ️ **Degree 3** gives {dim} features. Cubic cross-terms add flexibility beyond the essential "
                "$x_1 x_2$ term, which can help with noisy data but may overfit very small datasets."
            )

    # ── RBF ─────────────────────────────────────────────────
    elif kernel_name == "RBF / Gaussian":
        sim_1unit = round(math.exp(-gamma), 3)
        sim_half  = round(math.exp(-gamma * 0.25), 3)
        st.markdown("#### How the RBF kernel measures similarity")
        _katex_card(
            concept="<strong>RBF kernel</strong> measures similarity between two points purely by their Euclidean distance. It decays exponentially — nearby points are very similar (close to 1), distant points are nearly orthogonal (close to 0). $\\gamma$ controls how quickly similarity drops off.",
            formulas=[rf"K(x,\,x')=\exp\!\left(-{gamma:.2f}\,\|x-x'\|^2\right)"],
            intuition=(
                f"With $\\gamma={gamma:.2f}$: two points 0.5 units apart have similarity $\\approx {sim_half}$; "
                f"two points 1 unit apart have similarity $\\approx {sim_1unit}$. "
                "As $\\gamma\\to\\infty$ the kernel approaches a nearest-neighbour classifier. "
                "As $\\gamma\\to 0$ it becomes nearly constant and the boundary flattens."
            ),
            accent="green",
        )
        st.markdown("#### Connection to infinite dimensions")
        _katex_card(
            concept="RBF corresponds to an <strong>infinite-dimensional</strong> implicit feature map. Expanding the exponential via Taylor series reveals it simultaneously computes polynomial features of every degree.",
            formulas=[r"\exp(-\gamma\|x\|^2)\exp(-\gamma\|x'\|^2)\sum_{n=0}^{\infty}\frac{(2\gamma\,x^Tx')^n}{n!}"],
            intuition="This is why RBF is such a powerful general-purpose kernel — it never commits to a fixed polynomial degree. $\\gamma$ implicitly weights how much each degree contributes to the similarity score.",
            accent="green",
        )

    # ── Sigmoid ──────────────────────────────────────────────
    elif kernel_name == "Sigmoid":
        st.markdown("#### How the sigmoid kernel works")
        _katex_card(
            concept="<strong>Sigmoid kernel</strong> is inspired by neural network activation functions. With appropriate $\\alpha$ and $c$ it can mimic a two-layer neural network — but unlike RBF and Polynomial, it is <em>not always a valid kernel</em> (the Gram matrix is not always positive semidefinite).",
            formulas=[r"K(x,\,x')=\tanh(\alpha\,x^Tx'+c)"],
            intuition="The sigmoid kernel is included mainly for conceptual comparison with neural networks. In practice, RBF almost always produces better and more stable boundaries. Use it to draw the connection between SVMs and shallow networks.",
            accent="red",
        )
        st.markdown("#### Why is it not always a valid kernel?")
        _katex_card(
            concept="For the Gram matrix to be positive semidefinite we need $K(x,x')=\\phi(x)^T\\phi(x')$ for some $\\phi$. The $\\tanh$ function only satisfies this for restricted parameter ranges.",
            formulas=[r"G\succeq 0\iff\alpha>0,\;c\leq 0\quad\text{(sufficient condition)}"],
            intuition="When the Gram matrix has negative eigenvalues, the SVM quadratic program may not have a unique solution — producing unstable or poor decision boundaries. This is why the sigmoid boundary often looks worse than RBF on the same data.",
            accent="red",
        )