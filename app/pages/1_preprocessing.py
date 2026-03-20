from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.components.charts import (
    make_missing_values_bar,
    make_distribution_comparison,
    make_boxplot_comparison,
    make_correlation_heatmap,
)
from app.components.dataset_preview import render_dataset_overview
from src.data.sample_datasets import load_sample_dataset
from src.features.preprocessing import scale_numeric_features, summarize_dataframe
from src.llm.tutor import ask_tutor
from src.llm.client import is_ollama_running


st.title("🛠️ Preprocessing Playground")
st.write("Understand how preprocessing transforms your data.")

# -------------------------
# Dataset selection
# -------------------------
dataset_name = st.selectbox(
    "Choose a sample dataset",
    options=["iris", "wine"],
    index=0,
)

df = load_sample_dataset(dataset_name)
render_dataset_overview(df, title="Raw Dataset")

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

selected_features = st.multiselect(
    "Select numeric features to transform",
    options=numeric_cols,
    default=numeric_cols[: min(3, len(numeric_cols))],
)

scaling_method = st.selectbox(
    "Choose scaling method",
    options=["none", "standardization", "normalization"],
    index=0,
)

processed_df = scale_numeric_features(
    df,
    method=scaling_method,
    exclude_cols=[col for col in numeric_cols if col not in selected_features],
)

# -------------------------
# Save page context for tutor
# -------------------------
st.session_state["current_page"] = "preprocessing"
st.session_state["preprocessing_context"] = {
    "dataset_name": dataset_name,
    "scaling_method": scaling_method,
    "selected_features": selected_features,
    "available_numeric_columns": numeric_cols,
}

# -------------------------
# Session state for inline explanations
# -------------------------
if "preprocessing_distribution_explanation" not in st.session_state:
    st.session_state["preprocessing_distribution_explanation"] = None
if "preprocessing_distribution_state" not in st.session_state:
    st.session_state["preprocessing_distribution_state"] = None

if "preprocessing_heatmap_explanation" not in st.session_state:
    st.session_state["preprocessing_heatmap_explanation"] = None
if "preprocessing_heatmap_state" not in st.session_state:
    st.session_state["preprocessing_heatmap_state"] = None

# -------------------------
# Before vs After comparison
# -------------------------
st.markdown("## 🔄 Before vs After Scaling")

feature_to_plot = None
if selected_features:
    feature_to_plot = st.selectbox(
        "Choose a feature to visualize",
        options=selected_features,
    )

    st.session_state["preprocessing_context"]["feature_to_plot"] = feature_to_plot

    st.plotly_chart(
        make_distribution_comparison(df, processed_df, feature_to_plot),
        use_container_width=True,
    )

    st.plotly_chart(
        make_boxplot_comparison(df, processed_df, feature_to_plot),
        use_container_width=True,
    )

    current_distribution_state = {
        "dataset_name": dataset_name,
        "feature_to_plot": feature_to_plot,
        "scaling_method": scaling_method,
        "selected_features": tuple(selected_features),
    }

    if st.button("🧠 Explain this chart", key="explain_preprocessing_distribution_inline"):
        if not is_ollama_running():
            st.warning("Ollama is not running.")
        else:
            question = (
                f"Explain this before-vs-after preprocessing chart for dataset '{dataset_name}'. "
                f"The selected feature is '{feature_to_plot}' and the scaling method is "
                f"'{scaling_method}'. Explain how the distribution and spread change and why that matters."
            )

            with st.spinner("Generating explanation..."):
                try:
                    explanation = ask_tutor(
                        question=question,
                        topic="preprocessing",
                        chat_history=[],
                        model="mistral",
                    )
                    st.session_state["preprocessing_distribution_explanation"] = explanation
                    st.session_state["preprocessing_distribution_state"] = current_distribution_state
                except Exception as e:
                    st.session_state["preprocessing_distribution_explanation"] = f"Error: {e}"
                    st.session_state["preprocessing_distribution_state"] = current_distribution_state

    if st.session_state["preprocessing_distribution_explanation"]:
        saved_distribution_state = st.session_state["preprocessing_distribution_state"]
        distribution_is_stale = saved_distribution_state != current_distribution_state

        if distribution_is_stale:
            st.warning("⚠️ Chart changed. Refresh explanation.")

        with st.expander("Chart explanation", expanded=not distribution_is_stale):
            st.write(st.session_state["preprocessing_distribution_explanation"])
else:
    st.info("Select at least one numeric feature to compare before and after scaling.")

# -------------------------
# Correlation heatmap
# -------------------------
st.markdown("## 🔗 Feature Relationships")
st.plotly_chart(
    make_correlation_heatmap(df),
    use_container_width=True,
)

current_heatmap_state = {
    "dataset_name": dataset_name,
    "selected_features": tuple(selected_features),
    "numeric_columns": tuple(numeric_cols),
}

if st.button("🧠 Explain this chart", key="explain_preprocessing_heatmap_inline"):
    if not is_ollama_running():
        st.warning("Ollama is not running.")
    else:
        question = (
            f"Explain this correlation heatmap for dataset '{dataset_name}'. "
            f"Explain what strong positive or negative correlations mean and why they matter "
            f"for preprocessing and PCA."
        )

        with st.spinner("Generating explanation..."):
            try:
                explanation = ask_tutor(
                    question=question,
                    topic="preprocessing",
                    chat_history=[],
                    model="mistral",
                )
                st.session_state["preprocessing_heatmap_explanation"] = explanation
                st.session_state["preprocessing_heatmap_state"] = current_heatmap_state
            except Exception as e:
                st.session_state["preprocessing_heatmap_explanation"] = f"Error: {e}"
                st.session_state["preprocessing_heatmap_state"] = current_heatmap_state

if st.session_state["preprocessing_heatmap_explanation"]:
    saved_heatmap_state = st.session_state["preprocessing_heatmap_state"]
    heatmap_is_stale = saved_heatmap_state != current_heatmap_state

    if heatmap_is_stale:
        st.warning("⚠️ Chart changed. Refresh explanation.")

    with st.expander("Chart explanation", expanded=not heatmap_is_stale):
        st.write(st.session_state["preprocessing_heatmap_explanation"])

# -------------------------
# Summary table
# -------------------------
st.markdown("## 📊 Dataset Summary")
summary_df = summarize_dataframe(df)
st.dataframe(summary_df, use_container_width=True)

# -------------------------
# Missing values
# -------------------------
st.markdown("## ⚠️ Missing Values")
st.plotly_chart(
    make_missing_values_bar(df),
    use_container_width=True,
)

# -------------------------
# Learning insights
# -------------------------
st.markdown("## 🧠 Learning Insights")

if scaling_method == "none":
    scaling_explanation = (
        "No scaling has been applied yet, so feature magnitudes remain unchanged."
    )
elif scaling_method == "standardization":
    scaling_explanation = (
        "Standardization shifts selected features to have mean 0 and standard deviation 1."
    )
else:
    scaling_explanation = (
        "Normalization rescales selected features so their values lie between 0 and 1."
    )

selected_features_text = ", ".join(selected_features) if selected_features else "None"

st.markdown(
    f"""
**Current configuration**
- Dataset: **{dataset_name}**
- Scaling method: **{scaling_method}**
- Selected features: **{selected_features_text}**

**What this means**
- {scaling_explanation}
- Only the selected numeric features are transformed.
- Features not selected remain in their original scale.

**What to observe**
- In the overlay histogram, check whether the feature values shift to a new scale.
- In the boxplot, compare the spread and outliers before vs after transformation.
- In the correlation heatmap, look for highly correlated features, since those often motivate dimensionality reduction methods like PCA.

**Why preprocessing matters**
- Some machine learning models are sensitive to feature magnitudes.
- Features with much larger numeric ranges can dominate distance-based or gradient-based learning.
- Scaling helps make features more comparable and prepares the data for downstream methods.
"""
)