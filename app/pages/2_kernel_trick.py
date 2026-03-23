from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.kernel.generators import generate_kernel_dataset
from src.kernel.mappings import apply_polynomial_mapping
from src.llm.tutor import ask_tutor
from src.llm.client import is_ollama_running
from components.charts import (
    make_decision_boundary_plot,
    make_kernel_boundary_plot,
)


st.title("🌀 Kernel Trick Visualizer")
st.write("See how nonlinear data can become easier to separate after feature transformation.")

# -------------------------
# Inputs
# -------------------------
# initialize in case we are in upload mode and synthetic sliders are skipped
n_samples = None
noise = None
x_feature = "x1"
y_feature = "x2"
target_column = None

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    help="Upload a dataset with at least two numeric columns; this will override synthetic sample options.",
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read uploaded CSV: {e}")
        st.stop()

    numeric_cols = uploaded_df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Uploaded dataset must have at least two numeric columns.")
        st.stop()

    x_feature = st.selectbox("Choose X feature", options=numeric_cols, index=0)
    y_feature = st.selectbox(
        "Choose Y feature",
        options=[c for c in numeric_cols if c != x_feature],
        index=0,
    )

    categorical_candidates = [
        c
        for c in uploaded_df.columns
        if c not in [x_feature, y_feature] and uploaded_df[c].nunique() <= 20
    ]
    if categorical_candidates:
        target_column = st.selectbox("Choose target column", options=categorical_candidates)
        target_series = uploaded_df[target_column].astype(str)
    else:
        st.info(
            "No suitable target column found; using numeric discretized target from the first numeric column."
        )
        target_series = pd.qcut(uploaded_df[numeric_cols[0]], q=2, labels=False, duplicates="drop").astype(str)

    kernel_features_df = pd.DataFrame({
        "x1": uploaded_df[x_feature],
        "x2": uploaded_df[y_feature],
    })
    dataset_name = f"uploaded:{uploaded_file.name}"

else:
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

    kernel_features_df = kernel_data.features_df.copy()
    target_series = kernel_data.target_series

raw_df = kernel_features_df.copy()
raw_df["target"] = target_series.values.astype(str)

st.markdown("### Original 2D Feature Space")

# -------------------------
# SAVE CONTEXT FOR AI TUTOR
# -------------------------
st.session_state["current_page"] = "kernel trick"

st.session_state["kernel_context"] = {
    "dataset_name": dataset_name,
    "n_samples": n_samples,
    "noise": noise,
    "x_feature": x_feature,
    "y_feature": y_feature,
    "target_column": target_column,
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
raw_df = kernel_features_df.copy()
raw_df["target"] = target_series.values.astype(str)

st.markdown("### Original 2D Feature Space")

raw_fig = px.scatter(
    raw_df,
    x="x1",
    y="x2",
    color="target",
    title="Original Input Space",
)
st.plotly_chart(raw_fig, use_container_width=True)

# Linear boundary in original space
X_original = kernel_features_df[["x1", "x2"]].values
y = target_series.values

linear_model_original = LogisticRegression()
linear_model_original.fit(X_original, y)

boundary_fig_original = make_decision_boundary_plot(
    linear_model_original,
    X_original,
    y,
    title="Linear Decision Boundary in Original Space",
)
st.plotly_chart(boundary_fig_original, use_container_width=True)

current_original_state = {
    "dataset_name": dataset_name,
    "n_samples": n_samples,
    "noise": noise,
}

if st.button("🧠 Explain this chart", key="explain_kernel_original_inline"):

    if not is_ollama_running():
        st.warning("Ollama is not running.")
    else:
        st.session_state["kernel_chart_context"] = {
            "chart_name": "original_input_space",
            "dataset_name": dataset_name,
            "n_samples": n_samples,
            "noise": noise,
        }

        question = (
            f"Explain this original 2D input space and the linear decision boundary for the "
            f"'{dataset_name}' dataset. The number of samples is {n_samples} and the noise level is {noise}. "
            f"Explain why a straight-line separator struggles here and what noise changes visually."
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
mapped_df = apply_polynomial_mapping(kernel_features_df)
mapped_with_target = mapped_df.copy()
mapped_with_target["target"] = target_series.values.astype(str)

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
projected_df["target"] = target_series.values.astype(str)

st.markdown("### 2D Projection of Transformed Feature Space")

projected_fig = px.scatter(
    projected_df,
    x="PC1",
    y="PC2",
    color="target",
    title="Projected View of Transformed Feature Space",
)
st.plotly_chart(projected_fig, use_container_width=True)

# Boundary after mapping, shown back in original space
linear_model_mapped = LogisticRegression()
linear_model_mapped.fit(mapped_df.values, y)

kernel_boundary_fig = make_kernel_boundary_plot(
    linear_model_mapped,
    raw_df,
    apply_polynomial_mapping,
    title="Decision Boundary After Polynomial Mapping",
)
st.plotly_chart(kernel_boundary_fig, use_container_width=True)

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
        st.session_state["kernel_chart_context"] = {
            "chart_name": "projected_transformed_space",
            "dataset_name": dataset_name,
            "n_samples": n_samples,
            "noise": noise,
            "mapping": "polynomial",
        }

        question = (
            f"Explain this transformed representation and the nonlinear decision boundary for the "
            f"'{dataset_name}' dataset. The number of samples is {n_samples}, the noise level is {noise}, "
            f"and the mapping is polynomial. Explain why mapping into richer features can help separation."
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

st.markdown("### Comparing the boundaries")
st.markdown(
    """
    In the original input space, a linear classifier can only draw a straight boundary, which often struggles on
    datasets like moons or circles. After polynomial mapping, the classifier is still linear in the transformed
    feature space, but this corresponds to a curved boundary when viewed back in the original space.
    """
)

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