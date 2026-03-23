from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.sample_datasets import (
    get_available_datasets,
    load_sample_dataset,
    load_uploaded_dataset,
)
from src.pca.pipeline import run_pca
from src.llm.tutor import ask_tutor
from src.llm.client import is_ollama_running


st.title("📉 PCA Explorer")
st.write("Reduce dimensions and inspect explained variance.")

# -------------------------
# Dataset selection
# -------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    help="If you upload a dataset, it will be used instead of sample datasets.",
)

if uploaded_file is not None:
    try:
        df = load_uploaded_dataset(uploaded_file)
        dataset_name = f"uploaded:{uploaded_file.name}"
        st.success(f"Loaded uploaded dataset: {uploaded_file.name}")
    except Exception as exc:
        st.error(str(exc))
        st.stop()
else:
    dataset_name = st.selectbox(
        "Choose a sample dataset",
        options=get_available_datasets(),
        index=0,
        key="pca_dataset",
    )
    df = load_sample_dataset(dataset_name)

numeric_df = df.select_dtypes(include=["number"]).drop(columns=["target"], errors="ignore")
max_components = max(1, numeric_df.shape[1])

n_components = st.slider(
    "Number of principal components",
    min_value=1,
    max_value=max_components,
    value=min(2, max_components),
)

pca_result = run_pca(df, n_components=n_components, exclude_cols=["target"])

# -------------------------
# Save context for tutor
# -------------------------
st.session_state["current_page"] = "pca"
st.session_state["pca_context"] = {
    "dataset_name": dataset_name,
    "n_components": n_components,
    "explained_variance_ratio": pca_result.explained_variance_ratio,
    "cumulative_explained_variance": pca_result.cumulative_explained_variance,
}

# -------------------------
# Session state for inline explanations
# -------------------------
if "pca_variance_explanation" not in st.session_state:
    st.session_state["pca_variance_explanation"] = None
if "pca_variance_state" not in st.session_state:
    st.session_state["pca_variance_state"] = None

if "pca_scatter_explanation" not in st.session_state:
    st.session_state["pca_scatter_explanation"] = None
if "pca_scatter_state" not in st.session_state:
    st.session_state["pca_scatter_state"] = None

# -------------------------
# Dataset preview
# -------------------------
st.markdown("### Transformed dataset preview")
preview_df = pca_result.transformed_df.copy()

if "target_name" in df.columns:
    preview_df["target_name"] = df["target_name"].values
elif "target" in df.columns:
    preview_df["target"] = df["target"].values

st.dataframe(preview_df.head(10), use_container_width=True)

# -------------------------
# Explained variance
# -------------------------
st.markdown("### Explained variance")

variance_df = pd.DataFrame(
    {
        "component": [f"PC{i + 1}" for i in range(len(pca_result.explained_variance_ratio))],
        "explained_variance_ratio": pca_result.explained_variance_ratio,
        "cumulative_explained_variance": pca_result.cumulative_explained_variance,
    }
)

st.dataframe(variance_df, use_container_width=True)

fig = px.bar(
    variance_df,
    x="component",
    y="explained_variance_ratio",
    title="Explained Variance Ratio by Principal Component",
)

st.plotly_chart(fig, use_container_width=True)

current_variance_state = {
    "dataset_name": dataset_name,
    "n_components": n_components,
    "explained_variance_ratio": tuple(pca_result.explained_variance_ratio),
    "cumulative_explained_variance": tuple(pca_result.cumulative_explained_variance),
}

if st.button("🧠 Explain this chart", key="explain_pca_variance_inline"):
    if not is_ollama_running():
        st.warning("Ollama is not running.")
    else:
        question = (
            f"Explain this PCA explained variance chart for dataset '{dataset_name}'. "
            f"The user selected {n_components} components. "
            f"Explain what explained variance and cumulative variance mean, "
            f"and how to interpret whether the chosen number of components is sufficient."
        )

        with st.spinner("Generating explanation..."):
            try:
                explanation = ask_tutor(
                    question=question,
                    topic="pca",
                    chat_history=[],
                    model="mistral",
                )
                st.session_state["pca_variance_explanation"] = explanation
                st.session_state["pca_variance_state"] = current_variance_state
            except Exception as e:
                st.session_state["pca_variance_explanation"] = f"Error: {e}"
                st.session_state["pca_variance_state"] = current_variance_state

if st.session_state["pca_variance_explanation"]:
    saved_variance_state = st.session_state["pca_variance_state"]
    variance_is_stale = saved_variance_state != current_variance_state

    if variance_is_stale:
        st.warning("⚠️ Chart changed. Refresh explanation.")

    with st.expander("Chart explanation", expanded=not variance_is_stale):
        st.write(st.session_state["pca_variance_explanation"])

# -------------------------
# PCA scatter
# -------------------------
if n_components >= 2:
    scatter_df = pca_result.transformed_df.copy()

    if "target_name" in df.columns:
        scatter_df["class"] = df["target_name"].values
        color_col = "class"
    elif "target" in df.columns:
        scatter_df["class"] = df["target"].astype(str).values
        color_col = "class"
    else:
        color_col = None

    scatter_fig = px.scatter(
        scatter_df,
        x="PC1",
        y="PC2",
        color=color_col,
        title="Projection onto First Two Principal Components",
    )

    st.plotly_chart(scatter_fig, use_container_width=True)

    current_scatter_state = {
        "dataset_name": dataset_name,
        "n_components": n_components,
        "explained_variance_ratio": tuple(pca_result.explained_variance_ratio),
        "has_target_name": "target_name" in df.columns,
        "has_target": "target" in df.columns,
    }

    if st.button("🧠 Explain this chart", key="explain_pca_scatter_inline"):
        if not is_ollama_running():
            st.warning("Ollama is not running.")
        else:
            question = (
                f"Explain this PCA scatter plot for dataset '{dataset_name}'. "
                f"Explain what PC1 and PC2 represent, and what separation or overlap "
                f"between classes means."
            )

            with st.spinner("Generating explanation..."):
                try:
                    explanation = ask_tutor(
                        question=question,
                        topic="pca",
                        chat_history=[],
                        model="mistral",
                    )
                    st.session_state["pca_scatter_explanation"] = explanation
                    st.session_state["pca_scatter_state"] = current_scatter_state
                except Exception as e:
                    st.session_state["pca_scatter_explanation"] = f"Error: {e}"
                    st.session_state["pca_scatter_state"] = current_scatter_state

    if st.session_state["pca_scatter_explanation"]:
        saved_scatter_state = st.session_state["pca_scatter_state"]
        scatter_is_stale = saved_scatter_state != current_scatter_state

        if scatter_is_stale:
            st.warning("⚠️ Chart changed. Refresh explanation.")

        with st.expander("Chart explanation", expanded=not scatter_is_stale):
            st.write(st.session_state["pca_scatter_explanation"])

# -------------------------
# Component loadings
# -------------------------
st.markdown("### Component loadings")
st.dataframe(pca_result.components_df, use_container_width=True)