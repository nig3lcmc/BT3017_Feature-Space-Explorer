from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.llm.client import is_ollama_running
from src.llm.tutor import ask_tutor
from src.content.kernel_theory import (
    render_linear_separability_theory,
    render_feature_map_theory,
    render_infinite_dimension_theory,
    render_mercer_theory,
    render_kernel_choice_theory,
    render_kernel_pca_bridge_theory,
    render_kernel_specific_theory,
)

import json


# =========================================================
# Session state
# =========================================================
def init_kernel_state():
    defaults = {
        "kernel_original_explanation": None,
        "kernel_original_state": None,
        "kernel_lifted_explanation": None,
        "kernel_lifted_state": None,
        "kernel_boundary_explanation": None,
        "kernel_boundary_state": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =========================================================
# Dataset generators
# =========================================================
def generate_xor_dataset(n_samples: int = 300, noise: float = 0.08, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-1, 1, size=(n_samples, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X = X + rng.normal(0, noise, size=X.shape)
    return X, y


def generate_teaching_dataset(kernel_name: str, n_samples: int, noise: float, random_state: int = 42):
    if kernel_name == "Linear":
        X, y = make_blobs(
            n_samples=n_samples, centers=[(-1.6, -0.8), (1.4, 0.9)],
            cluster_std=0.75 + noise, random_state=random_state,
        )
        dataset_name = "blobs"
    elif kernel_name == "Polynomial":
        X, y = generate_xor_dataset(n_samples=n_samples, noise=max(0.03, noise), random_state=random_state)
        dataset_name = "xor"
    elif kernel_name == "RBF / Gaussian":
        X, y = make_circles(n_samples=n_samples, factor=0.45, noise=noise, random_state=random_state)
        dataset_name = "circles"
    else:
        X, y = make_moons(n_samples=n_samples, noise=max(0.08, noise), random_state=random_state)
        dataset_name = "moons"

    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["target"] = y.astype(str)
    df["r_squared"] = df["x1"] ** 2 + df["x2"] ** 2
    return dataset_name, df


# =========================================================
# Plot helpers
# =========================================================
def make_original_scatter(df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.scatter(df, x="x1", y="x2", color="target", title=title)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=50, b=0), legend_title_text="Class")
    return fig


def make_linear_boundary_line_plot(X: np.ndarray, y: np.ndarray, model, title: str) -> go.Figure:
    df_plot = pd.DataFrame(X, columns=["x1", "x2"])
    df_plot["target"] = y.astype(str)
    fig = px.scatter(df_plot, x="x1", y="x2", color="target", title=title)

    w = model.coef_[0]
    b = model.intercept_[0]
    x_min = float(df_plot["x1"].min()) - 0.15
    x_max = float(df_plot["x1"].max()) + 0.15
    y_min = float(df_plot["x2"].min()) - 0.15
    y_max = float(df_plot["x2"].max()) + 0.15
    x_vals = np.linspace(x_min, x_max, 300)

    if abs(w[1]) > 1e-8:
        y_vals = -(w[0] * x_vals + b) / w[1]
        # Clamp to plot range so the line doesn't shoot off screen
        in_range = (y_vals >= y_min - 0.5) & (y_vals <= y_max + 0.5)
        fig.add_trace(go.Scatter(
            x=x_vals[in_range], y=y_vals[in_range], mode="lines",
            name="Linear boundary", line=dict(width=3, dash="dash", color="#f87171"),
        ))
    elif abs(w[0]) > 1e-8:
        x_boundary = -b / w[0]
        fig.add_trace(go.Scatter(
            x=[x_boundary, x_boundary], y=[y_min, y_max], mode="lines",
            name="Linear boundary", line=dict(width=3, dash="dash", color="#f87171"),
        ))

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=50, b=0), legend_title_text="")
    return fig


def make_kernel_decision_boundary(
    X: np.ndarray, y: np.ndarray,
    kernel: str, gamma: float, degree: int, title: str,
) -> go.Figure:
    kernel_map = {"Linear": "linear", "Polynomial": "poly", "RBF / Gaussian": "rbf", "Sigmoid": "sigmoid"}
    svm = SVC(kernel=kernel_map.get(kernel, "rbf"), gamma=gamma, degree=degree, C=5.0)
    svm.fit(X, y)

    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 300),
        y=np.linspace(y_min, y_max, 300),
        z=Z, showscale=False,
        colorscale=[[0, "rgba(99,102,241,0.18)"], [1, "rgba(251,146,60,0.18)"]],
        contours=dict(showlines=False),
        name="Decision region",
    ))

    df_plot = pd.DataFrame(X, columns=["x1", "x2"])
    df_plot["target"] = y.astype(str)
    for label, color in [("0", "#818cf8"), ("1", "#fb923c")]:
        subset = df_plot[df_plot["target"] == label]
        fig.add_trace(go.Scatter(
            x=subset["x1"], y=subset["x2"], mode="markers",
            name=f"Class {label}",
            marker=dict(size=5, color=color, line=dict(width=0.5, color="white")),
        ))

    sv = svm.support_vectors_
    fig.add_trace(go.Scatter(
        x=sv[:, 0], y=sv[:, 1], mode="markers", name="Support vectors",
        marker=dict(size=9, color="rgba(0,0,0,0)", symbol="circle",
                    line=dict(width=2, color="white")),
    ))

    fig.update_layout(title=title, height=420, margin=dict(l=0, r=0, t=50, b=0),
                      legend_title_text="", xaxis_range=[x_min, x_max], yaxis_range=[y_min, y_max])
    return fig


def make_3d_lifted_plot(
    df: pd.DataFrame, title: str, camera_eye: dict,
    show_plane: bool = False, plane_z: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    for target in sorted(df["target"].unique()):
        subset = df[df["target"] == target]
        fig.add_trace(go.Scatter3d(
            x=subset["x1"], y=subset["x2"], z=subset["r_squared"],
            mode="markers", name=f"Class {target}", marker=dict(size=3),
        ))

    if show_plane and plane_z is not None:
        x_range = np.linspace(df["x1"].min() - 0.1, df["x1"].max() + 0.1, 20)
        y_range = np.linspace(df["x2"].min() - 0.1, df["x2"].max() + 0.1, 20)
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.full_like(xx, plane_z, dtype=float)
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.32, showscale=False, name="Separating plane"))

    fig.update_layout(title=title, height=480, margin=dict(l=0, r=0, t=50, b=0),
                      scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="r²",
                                 camera=dict(eye=camera_eye)))
    return fig


def make_radial_view(df: pd.DataFrame, title: str) -> go.Figure:
    rng = np.random.default_rng(42)
    radial_df = pd.DataFrame({
        "r_squared": df["r_squared"],
        "target": df["target"],
        "jitter": rng.normal(0, 0.035, size=len(df)),
    })
    fig = px.scatter(radial_df, x="r_squared", y="jitter", color="target", title=title)
    fig.update_yaxes(showticklabels=False, title_text="")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=50, b=0), legend_title_text="")
    return fig


def make_similarity_curve_plot(df: pd.DataFrame, gamma: float, title: str) -> go.Figure:
    r = np.sqrt(df["x1"] ** 2 + df["x2"] ** 2)
    sim = np.exp(-gamma * r**2)
    sim_df = pd.DataFrame({"radius": r, "similarity": sim, "target": df["target"]})
    fig = px.scatter(sim_df, x="radius", y="similarity", color="target", title=title)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=50, b=0), legend_title_text="")
    return fig


def make_interaction_feature_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Plot x1*x2 (the interaction term the polynomial kernel implicitly creates)
    vs x1 — this clearly shows why XOR becomes linearly separable.
    The two classes separate on the x1*x2 axis alone.
    """
    interaction_df = pd.DataFrame({
        "x1": df["x1"],
        "x1·x2": df["x1"] * df["x2"],
        "target": df["target"],
    })
    fig = px.scatter(
        interaction_df, x="x1", y="x1·x2", color="target",
        title="Lifted Feature: x₁ · x₂ (the interaction term)",
        labels={"x1·x2": "x₁ · x₂ (interaction term)"},
    )
    # Add a horizontal separator at y=0 to show the classes split there
    fig.add_hline(
        y=0, line_dash="dash", line_color="#4ade80", line_width=2,
        annotation_text="y = 0 separates the classes",
        annotation_position="top right",
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=50, b=0), legend_title_text="Class")
    return fig


# =========================================================
# Kernel header card (KaTeX component)
# =========================================================
KERNEL_INFO = {
    "Linear": {
        "icon": "📐", "color": "#38bdf8",
        "latex": r"K(x,\, x') = x^T x'",
        "tag": "no lifting",
        "desc": "No implicit lifting — uses the raw dot product. Best when classes are already close to linearly separable.",
        "intuition": "Two vectors are similar when they point in the same direction. No new features are created.",
        "when_to_use": "Text classification, high-dimensional sparse features, interpretable models.",
        "dataset_note": "We use <strong>blobs</strong> — two well-separated clusters — where a straight line suffices.",
    },
    "Polynomial": {
        "icon": "🔢", "color": "#a855f7",
        "latex": r"K(x,\, x') = (x^T x' + 1)^d",
        "tag": "interaction terms",
        "desc": "Captures feature interactions and curved boundaries by implicitly computing cross-terms.",
        "intuition": "A degree-2 kernel implicitly creates x₁², x₂², and x₁x₂ — without computing them explicitly.",
        "when_to_use": "XOR-style patterns, NLP, gene expression — wherever feature products carry signal.",
        "dataset_note": "We use <strong>XOR</strong> — neither x₁ nor x₂ alone predicts the class, but their product does.",
    },
    "RBF / Gaussian": {
        "icon": "🔵", "color": "#4ade80",
        "latex": r"K(x,\, x') = \exp\!\left(-\gamma\,\|x - x'\|^2\right)",
        "tag": "∞-dim implicit map",
        "desc": "Measures similarity via Gaussian decay of Euclidean distance. Corresponds to an ∞-dimensional implicit map.",
        "intuition": "Two points are 'similar' if they are close. Similarity drops smoothly to zero as distance grows.",
        "when_to_use": "Your default nonlinear kernel. Image classification, bioinformatics, regression.",
        "dataset_note": "We use <strong>concentric circles</strong> — perfectly suited to radial (distance-based) similarity.",
    },
    "Sigmoid": {
        "icon": "〰️", "color": "#f87171",
        "latex": r"K(x,\, x') = \tanh(\alpha\, x^T x' + c)",
        "tag": "use with caution",
        "desc": "Neural-network-style activation applied as a kernel. Not always a valid kernel (Mercer's theorem).",
        "intuition": "Mimics a single-layer neural network. Can create soft nonlinear decisions but can be unstable.",
        "when_to_use": "Teaching comparison to neural networks. Rarely preferred over RBF in practice.",
        "dataset_note": "We use <strong>moons</strong> to contrast sigmoid's softer effect against RBF's cleaner separation.",
    },
}


def render_kernel_header(kernel_name: str, gamma: float, degree: int) -> None:
    info = KERNEL_INFO[kernel_name]
    color = info["color"]

    if kernel_name == "Polynomial":
        latex_str = rf"K(x,\, x') = (x^T x' + 1)^{{{degree}}}"
    elif kernel_name == "RBF / Gaussian":
        latex_str = rf"K(x,\, x') = \exp\!\left(-{gamma:.2f}\,\|x - x'\|^2\right)"
    else:
        latex_str = info["latex"]

    latex_json = json.dumps(latex_str)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"
        onload="window._kr=true"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        background:transparent;padding:4px 2px 10px}}
  .card{{border:1.5px solid {color};border-radius:18px;overflow:hidden;
         background:rgba(255,255,255,0.02)}}
  .header{{display:flex;align-items:center;gap:12px;padding:18px 22px 14px;
           border-bottom:1px solid rgba(255,255,255,0.08)}}
  .icon{{font-size:1.8rem;line-height:1}}
  .title{{font-size:1.45rem;font-weight:800;color:{color}}}
  .tag{{margin-left:auto;border:1.5px solid {color};border-radius:999px;
        padding:3px 13px;color:{color};font-size:.78rem;font-weight:700;white-space:nowrap}}
  .desc{{padding:12px 22px;font-size:.93rem;color:#94a3b8;line-height:1.6;
         border-bottom:1px solid rgba(255,255,255,0.08)}}
  .formula-section{{padding:10px 22px 14px;background:rgba(0,0,0,0.2);
                    border-bottom:1px solid rgba(255,255,255,0.08)}}
  .label{{font-size:.65rem;letter-spacing:.1em;font-weight:700;text-transform:uppercase;
          color:#64748b;margin-bottom:8px}}
  #formula{{display:flex;justify-content:center;color:#f1f5f9}}
  #formula .katex{{color:#f1f5f9;font-size:1.15rem}}
  .grid{{display:grid;grid-template-columns:1fr 1fr}}
  .grid-cell{{padding:12px 18px;border-right:1px solid rgba(255,255,255,0.08)}}
  .grid-cell:last-child{{border-right:none}}
  .grid-cell .body{{font-size:.88rem;color:#cbd5e1;line-height:1.55}}
  .dataset-note{{padding:10px 22px 14px;font-size:.83rem;color:#64748b;
                 border-top:1px solid rgba(255,255,255,0.08)}}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <span class="icon">{info['icon']}</span>
    <span class="title">{kernel_name} Kernel</span>
    <span class="tag">{info['tag']}</span>
  </div>
  <div class="desc">{info['desc']}</div>
  <div class="formula-section">
    <div class="label">Formula</div>
    <div id="formula"></div>
  </div>
  <div class="grid">
    <div class="grid-cell">
      <div class="label">Intuition</div>
      <div class="body">{info['intuition']}</div>
    </div>
    <div class="grid-cell">
      <div class="label">When to use</div>
      <div class="body">{info['when_to_use']}</div>
    </div>
  </div>
  <div class="dataset-note">📌 Dataset: {info['dataset_note']}</div>
</div>
<script>
(function wait(n){{
  if(window._kr){{
    katex.render({latex_json},document.getElementById("formula"),
      {{displayMode:true,throwOnError:false}});
  }}else if(n<50){{setTimeout(function(){{wait(n+1)}},80)}}
}})(0);
</script>
</body>
</html>"""
    components.html(html, height=328, scrolling=False)


# =========================================================
# AI explanation helper
# =========================================================
def maybe_render_ai_explanation(*, button_key, state_key, state_snapshot_key, snapshot, question):
    if st.button("🧠 Ask AI to explain this", key=button_key):
        if not is_ollama_running():
            st.warning("Ollama is not running. Start it with `ollama serve` in your terminal.")
        else:
            with st.spinner("Generating explanation..."):
                try:
                    explanation = ask_tutor(question=question, topic="kernel trick",
                                            chat_history=[], model="mistral")
                    st.session_state[state_key] = explanation
                    st.session_state[state_snapshot_key] = snapshot
                except Exception as exc:
                    st.session_state[state_key] = f"Error: {exc}"
                    st.session_state[state_snapshot_key] = snapshot

    explanation = st.session_state.get(state_key)
    saved_snapshot = st.session_state.get(state_snapshot_key)
    if explanation:
        is_stale = saved_snapshot != snapshot
        if is_stale:
            st.warning("⚠️ Parameters changed — click again to refresh.")
        with st.expander("AI Explanation", expanded=not is_stale):
            st.write(explanation)


# =========================================================
# Step progress bar
# =========================================================
def render_progress(active: int):
    steps = ["🔍 Step 1: The Problem", "✨ Step 2: Apply the Kernel", "🧠 Step 3: Understand Why"]
    cols = st.columns(3)
    for i, (col, label) in enumerate(zip(cols, steps)):
        with col:
            is_active = i == active
            bg = "#4ade80" if is_active else "#1e293b"
            fg = "#0f172a" if is_active else "#475569"
            st.markdown(
                f'<div style="background:{bg};border-radius:10px;padding:9px 12px;text-align:center;'
                f'color:{fg};font-weight:{"700" if is_active else "500"};font-size:.85rem;">{label}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# =========================================================
# Page
# =========================================================
st.title("🌀 Kernel Trick")
st.write(
    "Kernels implicitly map data into higher dimensions — turning unseparable problems into separable ones without ever computing the transformation. "
)

init_kernel_state()

with st.expander("🧩 Why do we need kernels?", expanded=False):
    st.markdown(
        r"""
Most real-world data is **not linearly separable**. A straight line may work for simple blobs,  
but it fails on patterns like **XOR**, **circles**, and **moons**.

A **kernel** helps by comparing points **as if they had been mapped into a richer feature space**,  
without explicitly computing that transformation.

The key identity is:

$$
K(x, x') = \phi(x)^T \phi(x')
$$

This lets SVMs learn:
- curved boundaries
- feature interactions
- higher-dimensional structure

while still training from pairwise similarities alone.
"""
    )

init_kernel_state()

# ── Kernel selector ───────────────────────────────────────
st.markdown("### Choose a kernel to study")
kernel_name = st.radio(
    "kernel", options=list(KERNEL_INFO.keys()), horizontal=True,
    label_visibility="collapsed",
    format_func=lambda k: f"{KERNEL_INFO[k]['icon']} {k}",
)

st.divider()

# ── Dataset controls ──────────────────────────────────────
with st.expander("⚙️ Dataset controls", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        n_samples = st.slider("Samples", min_value=120, max_value=1000, value=300, step=20)
    with c2:
        noise = st.slider("Noise level", min_value=0.00, max_value=0.30, value=0.08, step=0.01)

    gamma = 0.50
    degree = 2
    if kernel_name == "RBF / Gaussian":
        gamma = st.slider("γ (gamma) — controls locality", min_value=0.05, max_value=2.0, value=0.50, step=0.05)
        st.caption("High γ → tighter, more complex boundary. Low γ → smoother, more global.")
    elif kernel_name == "Polynomial":
        degree = st.slider("Polynomial degree", min_value=2, max_value=6, value=2, step=1)
        # Import helper locally so caption is always correct for this degree
        from src.content.kernel_theory import _poly_dim
        _dim = _poly_dim(degree)
        st.caption(
            f"Degree {degree}: maps 2 input features into a {_dim}-dimensional space. "
            f"Includes all monomials $x_1^a x_2^b$ where $a+b \\leq {degree}$, plus the bias term."
        )

dataset_name, df = generate_teaching_dataset(kernel_name=kernel_name, n_samples=n_samples, noise=noise)

st.session_state["current_page"] = "kernel trick"
st.session_state["kernel_context"] = {
    "kernel_name": kernel_name, "dataset_name": dataset_name,
    "n_samples": n_samples, "noise": noise, "gamma": gamma, "degree": degree,
}

X = df[["x1", "x2"]].values
y = df["target"].astype(int).values

# Bug fix: use lbfgs solver which is more stable at large n_samples,
# and increase max_iter to ensure convergence regardless of sample count.
linear_model = LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0)
linear_model.fit(X, y)

# ── Kernel header card ────────────────────────────────────
render_kernel_header(kernel_name, gamma, degree)

# =========================================================
# 3-step tabs
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["🔍 Step 1: The Problem", "✨ Step 2: Apply the Kernel", "🧠 Step 3: Understand Why"]
)

# ──────────────────────────────────────────────────────────
# TAB 1 — The Problem
# ──────────────────────────────────────────────────────────
with tab1:
    render_progress(active=0)

    st.markdown(
        f"### Can a straight line separate this data?\n\n"
        f"The **{dataset_name}** dataset is chosen because it highlights exactly what the "
        f"**{kernel_name} kernel** solves. A logistic regression (linear model) can only draw "
        f"a **straight boundary** — watch what happens:"
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(make_original_scatter(df, f"Raw Data: {dataset_name.title()}"),
                        use_container_width=True)
    with col_right:
        st.plotly_chart(make_linear_boundary_line_plot(X, y, linear_model, "Best Straight-Line Boundary"),
                        use_container_width=True)

    render_linear_separability_theory()

    if kernel_name == "Linear":
        st.success("✅ **Linear kernel works here.** The blobs are well-separated — a straight line does the job. No nonlinear transformation needed.")
    else:
        st.error(f"❌ **Linear classifier fails here.** The {dataset_name} structure cannot be separated by a straight line. This is exactly the problem the **{kernel_name} kernel** solves.")

    maybe_render_ai_explanation(
        button_key="explain_tab1",
        state_key="kernel_original_explanation",
        state_snapshot_key="kernel_original_state",
        snapshot={"kernel_name": kernel_name, "dataset_name": dataset_name,
                  "n_samples": n_samples, "noise": noise},
        question=(
            f"We're teaching undergraduates about the kernel trick using the {dataset_name} dataset "
            f"and the {kernel_name} kernel. Explain why a linear classifier "
            f"{'succeeds' if kernel_name == 'Linear' else 'fails'} on this data, "
            f"and what intuition that builds for why a kernel is needed."
        ),
    )
    st.info("👉 Move to **Step 2** to see the kernel boundary in action.")


# ──────────────────────────────────────────────────────────
# TAB 2 — Apply the Kernel
# ──────────────────────────────────────────────────────────
with tab2:
    render_progress(active=1)

    st.markdown(
        f"### The {kernel_name} kernel finds a boundary a straight line cannot.\n\n"
        f"An SVM with the **{kernel_name} kernel** is trained on the same data. "
        f"The shaded regions show which class the model predicts across the whole space."
    )

    st.plotly_chart(
        make_kernel_decision_boundary(X, y, kernel_name, gamma=gamma, degree=degree,
                                      title=f"{kernel_name} Kernel SVM — Decision Boundary"),
        use_container_width=True,
    )
    st.caption("Circled points are **support vectors** — the training points closest to the boundary that define where it sits.")

    # ── Kernel-specific lifting / insight visuals ──────────
    if kernel_name == "RBF / Gaussian":
        st.markdown("---")
        st.markdown("### Why does RBF work? Adding a dimension.")
        st.markdown(
            r"For circles, a useful illustrative lift is: $r^2 = x_1^2 + x_2^2$. "
            "After lifting, the inner and outer classes sit at **different heights** — "
            "a flat plane can cut them apart."
        )
        c1, c2 = st.columns(2)
        class_means = df.groupby("target")["r_squared"].mean().sort_values()
        plane_z = float((class_means.iloc[0] + class_means.iloc[-1]) / 2.0)
        with c1:
            st.plotly_chart(
                make_3d_lifted_plot(df, "Top View — Lifted Space",
                                    camera_eye={"x": 0.0, "y": 0.0, "z": 2.35}),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                make_3d_lifted_plot(df, "Side View — Plane Separates the Classes",
                                    camera_eye={"x": 2.0, "y": 0.2, "z": 0.25},
                                    show_plane=True, plane_z=plane_z),
                use_container_width=True,
            )
        st.plotly_chart(make_radial_view(df, "Collapsed to 1D: Radial Feature"), use_container_width=True)
        st.plotly_chart(
            make_similarity_curve_plot(df, gamma, f"RBF Similarity vs Radius (γ = {gamma:.2f})"),
            use_container_width=True,
        )
        st.caption(
            f"With γ = {gamma:.2f}, similarity drops quickly with distance. "
            "Try increasing γ — notice how the boundary tightens around each cluster."
        )

    elif kernel_name == "Polynomial":
        st.markdown("---")
        st.markdown(
            "### What feature does the polynomial kernel create that unlocks this?\n\n"
            "The key for XOR is the **interaction term** $x_1 \\cdot x_2$. "
            "This term is **positive** when both inputs have the same sign (quadrants 1 & 3) "
            "and **negative** when they differ (quadrants 2 & 4) — matching the XOR pattern exactly.\n\n"
            "A single horizontal line at $x_1 \\cdot x_2 = 0$ separates the two classes perfectly:"
        )
        st.plotly_chart(make_interaction_feature_scatter(df), use_container_width=True)
        st.caption(
            "The green dashed line at y = 0 perfectly separates the classes. "
            "This is the implicit feature that the polynomial kernel discovers — without you ever computing it explicitly."
        )

    elif kernel_name == "Linear":
        st.markdown("---")
        st.success(
            "**No lifting needed!** The Linear kernel uses the original dot product. "
            "The SVM finds the same straight-line boundary as logistic regression, "
            "but with **maximum margin** — that's SVM's contribution even without a kernel trick."
        )

    elif kernel_name == "Sigmoid":
        st.markdown("---")
        st.markdown(
            "### Sigmoid: a neural-network-inspired kernel\n\n"
            "The sigmoid kernel mimics a single-layer neural network activation. "
            "It can create soft nonlinear regions — but unlike RBF it is **not always a valid kernel** "
            "(does not always satisfy Mercer's theorem)."
        )
        st.warning(
            "📌 **Try switching to RBF** and compare boundary quality on the same moons data. "
            "RBF almost always wins — Sigmoid is shown here for conceptual contrast."
        )

    maybe_render_ai_explanation(
        button_key="explain_tab2",
        state_key="kernel_lifted_explanation",
        state_snapshot_key="kernel_lifted_state",
        snapshot={"kernel_name": kernel_name, "dataset_name": dataset_name,
                  "n_samples": n_samples, "noise": noise, "gamma": gamma, "degree": degree},
        question=(
            f"Explain the {kernel_name} SVM decision boundary and any lifting visualisations "
            f"for the {dataset_name} dataset. Cover: what the coloured regions mean, "
            f"what support vectors are, and how the kernel achieves this without "
            f"explicitly computing new features."
        ),
    )
    st.info("👉 Move to **Step 3** to understand the math behind this.")


# ──────────────────────────────────────────────────────────
# TAB 3 — Understand Why (kernel-specific)
# ──────────────────────────────────────────────────────────
with tab3:
    render_progress(active=2)

    st.markdown(f"### The math behind the **{kernel_name} kernel**")
    st.caption("This section is tailored to the kernel you selected above.")

    # ── Kernel-specific deep-dive ──────────────────────────
    render_kernel_specific_theory(kernel_name, gamma=gamma, degree=degree)

    st.markdown("---")
    st.markdown("### All kernels at a glance")

    cheat = pd.DataFrame({
        "Kernel":        ["Linear", "Polynomial", "RBF / Gaussian", "Sigmoid"],
        "Implicit dims": ["d (original)", "O(dᵏ)", "∞", "depends"],
        "Best for":      ["Linearly separable", "Feature interactions", "General nonlinear", "Teaching only"],
        "Key parameter": ["—", "degree d", "γ (gamma)", "α, c"],
        "Mercer valid?": ["✅", "✅", "✅", "⚠️ Not always"],
    })
    st.dataframe(cheat, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### General theory")
    render_feature_map_theory()
    render_infinite_dimension_theory()
    render_mercer_theory()
    render_kernel_choice_theory()

    st.markdown("---")
    st.markdown("### Bridge to Kernel PCA")
    st.write(
        "The kernel trick extends beyond classification. Applied to PCA it lets us find "
        "**nonlinear** principal components — capturing variance along curved manifolds."
    )

    b1, b2, b3 = st.columns(3)
    for col, icon, color, title, desc in [
        (b1, "🧊", "#38bdf8", "Standard PCA", "Linear structure only"),
        (b2, "🧬", "#a855f7", "Apply Kernel Trick", "Implicit map φ(x)"),
        (b3, "📈", "#4ade80", "Kernel PCA", "PCA in implicit feature space"),
    ]:
        with col:
            col.markdown(
                f'<div style="border:1px solid {color}55;border-radius:16px;padding:16px;'
                f'background:rgba(255,255,255,0.02);text-align:center;min-height:120px;">'
                f'<div style="font-size:1.7rem;">{icon}</div>'
                f'<div style="font-size:1.05rem;font-weight:700;color:{color};margin-top:8px;">{title}</div>'
                f'<div style="color:#94a3b8;font-size:.87rem;margin-top:5px;">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    render_kernel_pca_bridge_theory()

    maybe_render_ai_explanation(
        button_key="explain_tab3",
        state_key="kernel_boundary_explanation",
        state_snapshot_key="kernel_boundary_state",
        snapshot={"kernel_name": kernel_name, "gamma": gamma, "degree": degree},
        question=(
            f"Summarise the key theoretical ideas behind the {kernel_name} kernel for undergraduate "
            f"students: the feature map intuition, Mercer's theorem, and when to choose this kernel "
            f"over alternatives. Keep it clear and concrete."
        ),
    )

    st.markdown(
        '<div style="border:1px solid rgba(74,222,128,0.45);border-radius:16px;padding:20px;'
        'background:rgba(74,222,128,0.04);margin-top:20px;">'
        '<div style="font-size:1.4rem;font-weight:800;color:#4ade80;">Ready for PCA? 📉</div>'
        '<div style="color:#cbd5e1;margin-top:8px;font-size:.97rem;">'
        'The next module shows how principal components reduce dimensionality — '
        'and how the kernel trick extends that to nonlinear structure.</div></div>',
        unsafe_allow_html=True,
    )