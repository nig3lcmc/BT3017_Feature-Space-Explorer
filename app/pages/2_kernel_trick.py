from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.kernel.generators import generate_kernel_dataset
from src.kernel.mappings import apply_polynomial_mapping
from src.llm.tutor import ask_tutor
from src.llm.client import is_ollama_running


st.title("🌀 Kernel Trick Visualizer")
st.write("See how nonlinear data can become easier to separate after feature transformation.")

# -------------------------
# Inputs
# -------------------------
dataset_name = st.selectbox(
    "Choose a synthetic dataset",
    options=["moons", "circles"],
    index=0,
)

n_samples = st.slider(
    "Number of samples",
    min_value=100,
    max_value=1000,
    value=300,
    step=50,
)

noise = st.slider(
    "Noise level",
    min_value=0.0,
    max_value=0.3,
    value=0.1,
    step=0.01,
)

kernel_data = generate_kernel_dataset(
    dataset_name=dataset_name,
    n_samples=n_samples,
    noise=noise,
)

# -------------------------
# SAVE CONTEXT FOR AI TUTOR (DO NOT CHANGE)
# -------------------------
st.session_state["current_page"] = "kernel trick"

st.session_state["kernel_context"] = {
    "dataset_name": dataset_name,
    "n_samples": n_samples,
    "noise": noise,
    "mapping": "polynomial",
}

# -------------------------
# Session state (inline explanations)
# -------------------------
if "kernel_original_explanation" not in st.session_state:
    st.session_state["kernel_original_explanation"] = None
if "kernel_original_state" not in st.session_state:
    st.session_state["kernel_original_state"] = None

if "kernel_projected_explanation" not in st.session_state:
    st.session_state["kernel_projected_explanation"] = None
if "kernel_projected_state" not in st.session_state:
    st.session_state["kernel_projected_state"] = None

# -------------------------
# ORIGINAL SPACE
# -------------------------
raw_df = kernel_data.features_df.copy()
raw_df["target"] = kernel_data.target_series.values.astype(str)

st.markdown("### Original 2D Feature Space")

raw_fig = px.scatter(
    raw_df,
    x="x1",
    y="x2",
    color="target",
    title="Original Input Space",
)
st.plotly_chart(raw_fig, use_container_width=True)

current_original_state = {
    "dataset_name": dataset_name,
    "n_samples": n_samples,
    "noise": noise,
}

if st.button("🧠 Explain this chart", key="explain_kernel_original_inline"):

    if not is_ollama_running():
        st.warning("Ollama is not running.")
    else:
        # ✅ keep tutor compatibility
        st.session_state["kernel_chart_context"] = {
            "chart_name": "original_input_space",
            "dataset_name": dataset_name,
            "n_samples": n_samples,
            "noise": noise,
        }

        question = (
            f"Explain this original 2D input space for the '{dataset_name}' dataset. "
            f"The number of samples is {n_samples} and the noise level is {noise}. "
            f"Explain why the classes may be difficult to separate and what noise changes visually."
        )

        with st.spinner("Generating explanation..."):
            try:
                explanation = ask_tutor(
                    question=question,
                    topic="kernel trick",
                    chat_history=[],
                    model="mistral",
                )
                st.session_state["kernel_original_explanation"] = explanation
                st.session_state["kernel_original_state"] = current_original_state
            except Exception as e:
                st.session_state["kernel_original_explanation"] = f"Error: {e}"
                st.session_state["kernel_original_state"] = current_original_state

if st.session_state["kernel_original_explanation"]:
    saved_state = st.session_state["kernel_original_state"]
    is_stale = saved_state != current_original_state

    if is_stale:
        st.warning("⚠️ Chart changed. Refresh explanation.")

    with st.expander("Chart explanation", expanded=not is_stale):
        st.write(st.session_state["kernel_original_explanation"])


# -------------------------
# MAPPING + PCA PROJECTION
# -------------------------
mapped_df = apply_polynomial_mapping(kernel_data.features_df)
mapped_with_target = mapped_df.copy()
mapped_with_target["target"] = kernel_data.target_series.values.astype(str)

st.markdown("### Transformed Feature Preview")
st.dataframe(mapped_with_target.head(10), use_container_width=True)

st.markdown("### Why a projection is shown below")
st.markdown(
    """
    After polynomial mapping, the data is no longer just 2-dimensional.
    To visualize the transformed representation, we project it back into 2 dimensions using PCA.
    This gives an intuition for how the transformed space may become easier to separate.
    """
)

scaler = StandardScaler()
mapped_scaled = scaler.fit_transform(mapped_df)

pca = PCA(n_components=2)
mapped_projected = pca.fit_transform(mapped_scaled)

projected_df = pd.DataFrame(mapped_projected, columns=["PC1", "PC2"])
projected_df["target"] = kernel_data.target_series.values.astype(str)

st.markdown("### 2D Projection of Transformed Feature Space")

projected_fig = px.scatter(
    projected_df,
    x="PC1",
    y="PC2",
    color="target",
    title="Projected View of Transformed Feature Space",
)
st.plotly_chart(projected_fig, use_container_width=True)

current_projected_state = {
    "dataset_name": dataset_name,
    "n_samples": n_samples,
    "noise": noise,
    "mapping": "polynomial",
}

if st.button("🧠 Explain this chart", key="explain_kernel_projected_inline"):

    if not is_ollama_running():
        st.warning("Ollama is not running.")
    else:
        # ✅ keep tutor compatibility
        st.session_state["kernel_chart_context"] = {
            "chart_name": "projected_transformed_space",
            "dataset_name": dataset_name,
            "n_samples": n_samples,
            "noise": noise,
            "mapping": "polynomial",
        }

        question = (
            f"Explain this projected transformed feature space for the '{dataset_name}' dataset. "
            f"The number of samples is {n_samples}, the noise level is {noise}, and the mapping is polynomial. "
            f"Explain what this suggests about how polynomial mapping changes the data representation."
        )

        with st.spinner("Generating explanation..."):
            try:
                explanation = ask_tutor(
                    question=question,
                    topic="kernel trick",
                    chat_history=[],
                    model="mistral",
                )
                st.session_state["kernel_projected_explanation"] = explanation
                st.session_state["kernel_projected_state"] = current_projected_state
            except Exception as e:
                st.session_state["kernel_projected_explanation"] = f"Error: {e}"
                st.session_state["kernel_projected_state"] = current_projected_state

if st.session_state["kernel_projected_explanation"]:
    saved_state = st.session_state["kernel_projected_state"]
    is_stale = saved_state != current_projected_state

    if is_stale:
        st.warning("⚠️ Chart changed. Refresh explanation.")

    with st.expander("Chart explanation", expanded=not is_stale):
        st.write(st.session_state["kernel_projected_explanation"])


# -------------------------
# INTUITION
# -------------------------
st.markdown("### Intuition")
st.markdown(
    """
    In the original input space, the classes may not be linearly separable.
    By mapping the data into a richer feature space using polynomial terms,
    the structure of the data changes. This is the core intuition behind the kernel trick:
    rather than solving the problem in the original space, we transform the representation
    so that simpler decision boundaries become possible.
    """
)