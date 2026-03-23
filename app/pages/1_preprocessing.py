from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
from src.data.sample_datasets import (
    get_available_datasets,
    load_sample_dataset,
    load_uploaded_dataset,
)
from src.features.preprocessing import scale_numeric_features, summarize_dataframe
from src.llm.tutor import ask_tutor
from src.llm.client import is_ollama_running


st.title("🛠️ Preprocessing Playground")
st.write("Understand how preprocessing transforms your data.")

# -------------------------
# Dataset selection
# -------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    help="If you upload a dataset, it will be used instead of the sample datasets.",
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
    )
    df = load_sample_dataset(dataset_name)

render_dataset_overview(df, title="Raw Dataset")

# ------------------------- 
# Data Overview
# -------------------------
st.markdown("## 📊 Data Overview")
st.dataframe(df.describe(include="all").T, use_container_width=True)

st.markdown("### Data Types")
dtypes_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str)})
st.dataframe(dtypes_df, use_container_width=True)

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

st.info(f"**Numeric columns:** {len(numeric_cols)} | **Categorical columns:** {len(categorical_cols)}")

# ------------------------- 
# Cleaning Tools
# -------------------------
st.markdown("## 🧹 Data Cleaning")

# Missing values
st.markdown("### Missing Values")
st.plotly_chart(make_missing_values_bar(df), use_container_width=True, key="missing_before")

current_missing_option = st.session_state.get("missing_option", "None")
if st.button("🧠 Explain this chart", key="explain_preprocessing_missing_inline"):
    question = (
        f"Explain the missing-values bar chart for dataset '{dataset_name}' and why this is useful for preprocessing."
    )
    with st.spinner("Generating explanation..."):
        try:
            explanation = ask_tutor(
                question=question,
                topic="preprocessing",
                chat_history=[],
                model="mistral",
            )
            st.session_state["preprocessing_missing_explanation"] = explanation
            st.session_state["preprocessing_missing_state"] = {"dataset_name": dataset_name, "missing_option": current_missing_option}
        except Exception as e:
            st.session_state["preprocessing_missing_explanation"] = f"Error: {e}"
            st.session_state["preprocessing_missing_state"] = {"dataset_name": dataset_name, "missing_option": current_missing_option}

if st.session_state.get("preprocessing_missing_explanation"):
    saved_missing_state = st.session_state.get("preprocessing_missing_state")
    missing_is_stale = saved_missing_state != {"dataset_name": dataset_name, "missing_option": current_missing_option}
    if missing_is_stale:
        st.warning("⚠️ Missing-value settings changed. Refresh explanation.")
    with st.expander("Chart explanation", expanded=not missing_is_stale):
        st.write(st.session_state.get("preprocessing_missing_explanation"))

st.markdown("**Before handling:**")

missing_option = st.selectbox(
    "Handle missing values",
    options=["None", "Drop rows with missing", "Fill numeric with mean", "Fill numeric with median", "Fill categorical with mode"],
    index=0,
    help="Choose how to handle missing values in the dataset.",
    key="missing_option",
)

df_clean = df.copy()
if missing_option == "Drop rows with missing":
    df_clean = df_clean.dropna()
    st.success("Dropped rows with missing values.")
elif missing_option == "Fill numeric with mean":
    for col in numeric_cols:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    st.success("Filled numeric missing values with mean.")
elif missing_option == "Fill numeric with median":
    for col in numeric_cols:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    st.success("Filled numeric missing values with median.")
elif missing_option == "Fill categorical with mode":
    for col in categorical_cols:
        if df_clean[col].isna().any():
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
    st.success("Filled categorical missing values with mode.")

if missing_option != "None":
    st.markdown("**After handling:**")
    st.plotly_chart(make_missing_values_bar(df_clean), use_container_width=True, key="missing_after")

# Duplicates
st.markdown("### Duplicates")
if st.button("Remove duplicate rows"):
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    st.success(f"Removed {before - after} duplicate rows.")

# Column operations
st.markdown("### Column Operations")
drop_cols = st.multiselect("Select columns to drop", options=df_clean.columns.tolist(), key="drop_cols")
if drop_cols:
    df_clean = df_clean.drop(columns=drop_cols)
    st.success(f"Dropped columns: {', '.join(drop_cols)}")

# Outlier handling (for numeric)
st.markdown("### Outlier Handling")
outlier_col = st.selectbox("Select numeric column for outlier handling", options=numeric_cols, index=0 if numeric_cols else None, key="outlier_col")
df_before_outlier = df_clean.copy()
if outlier_col:
    st.plotly_chart(make_boxplot_comparison(df_before_outlier[[outlier_col]], df_before_outlier[[outlier_col]], outlier_col), use_container_width=True, key="outlier_before")
    st.markdown("**Before handling:**")

outlier_method = st.selectbox(
    "Outlier method",
    options=["None", "Clip to 1st-99th percentile", "Remove outliers (IQR)"],
    index=0,
    key="outlier_method",
)

if outlier_method == "Clip to 1st-99th percentile":
    if outlier_col in df_clean.columns:
        lower = df_clean[outlier_col].quantile(0.01)
        upper = df_clean[outlier_col].quantile(0.99)
        df_clean[outlier_col] = df_clean[outlier_col].clip(lower, upper)
        st.success(f"Clipped {outlier_col} to 1st-99th percentile.")
elif outlier_method == "Remove outliers (IQR)":
    if outlier_col in df_clean.columns:
        Q1 = df_clean[outlier_col].quantile(0.25)
        Q3 = df_clean[outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        before = len(df_clean)
        df_clean = df_clean[(df_clean[outlier_col] >= lower_bound) & (df_clean[outlier_col] <= upper_bound)]
        after = len(df_clean)
        st.success(f"Removed {before - after} outliers from {outlier_col} using IQR.")

if outlier_method != "None" and outlier_col:
    st.markdown("**After handling:**")
    st.plotly_chart(make_boxplot_comparison(df_before_outlier[[outlier_col]], df_clean[[outlier_col]], outlier_col), use_container_width=True, key="outlier_after")

# Encoding categoricals
st.markdown("### Encode Categorical Features")
encode_cols = st.multiselect("Select categorical columns to encode", options=categorical_cols, key="encode_cols")
encode_method = st.selectbox("Encoding method", options=["None", "One-hot", "Label encoding"], index=0, key="encode_method")

if encode_method == "One-hot" and encode_cols:
    df_clean = pd.get_dummies(df_clean, columns=encode_cols, drop_first=True)
    st.success(f"One-hot encoded: {', '.join(encode_cols)}")
elif encode_method == "Label encoding" and encode_cols:
    for col in encode_cols:
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
    st.success(f"Label encoded: {', '.join(encode_cols)}")

# Update numeric_cols after cleaning
numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()

st.markdown("## 📋 Cleaned Dataset Preview")
render_dataset_overview(df_clean, title="Cleaned Dataset")

# ------------------------- 
# Feature Relationships
# -------------------------
st.markdown("## 🔗 Feature Relationships")
st.plotly_chart(make_correlation_heatmap(df_clean), use_container_width=True, key="correlation_heatmap")

if st.button("🧠 Explain this chart", key="explain_preprocessing_heatmap_inline"):
    question = (
        f"Explain this correlation heatmap for dataset '{dataset_name}'. "
        f"Explain what strong positive or negative correlations mean and why they matter for preprocessing and PCA."
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
            st.session_state["preprocessing_heatmap_state"] = {
                "dataset_name": dataset_name,
                "selected_features": tuple(selected_features),
                "numeric_columns": tuple(numeric_cols),
            }
        except Exception as e:
            st.session_state["preprocessing_heatmap_explanation"] = f"Error: {e}"
            st.session_state["preprocessing_heatmap_state"] = {
                "dataset_name": dataset_name,
                "selected_features": tuple(selected_features),
                "numeric_columns": tuple(numeric_cols),
            }

if st.session_state.get("preprocessing_heatmap_explanation"):
    saved_heatmap_state = st.session_state.get("preprocessing_heatmap_state")
    heatmap_is_stale = saved_heatmap_state != {
        "dataset_name": dataset_name,
        "selected_features": tuple(selected_features),
        "numeric_columns": tuple(numeric_cols),
    }
    if heatmap_is_stale:
        st.warning("⚠️ Chart changed. Refresh explanation.")
    with st.expander("Chart explanation", expanded=not heatmap_is_stale):
        st.write(st.session_state.get("preprocessing_heatmap_explanation"))

# -------------------------
# Feature Selection & Scaling
# -------------------------
st.markdown("## 🔄 Feature Scaling")

# Initialize session state for explanations
if "preprocessing_distribution_explanation" not in st.session_state:
    st.session_state["preprocessing_distribution_explanation"] = None
if "preprocessing_distribution_state" not in st.session_state:
    st.session_state["preprocessing_distribution_state"] = None

selected_features = st.multiselect(
    "Select numeric features to transform",
    options=numeric_cols,
    default=numeric_cols[: min(3, len(numeric_cols))],
    key="selected_features",
)

scaling_method = st.selectbox(
    "Choose scaling method",
    options=["none", "standardization", "normalization"],
    index=0,
    key="scaling_method",
    help="Choose how to scale selected numeric features.",
)

processed_df = scale_numeric_features(
    df_clean,
    method=scaling_method,
    exclude_cols=[col for col in numeric_cols if col not in selected_features],
)

# Save page context for tutor
st.session_state["current_page"] = "preprocessing"
st.session_state["preprocessing_context"] = {
    "dataset_name": dataset_name,
    "scaling_method": scaling_method,
    "selected_features": selected_features,
    "available_numeric_columns": numeric_cols,
}

# -------------------------
# Before vs After Scaling
# -------------------------
st.markdown("## 📊 Scaling Impact Analysis")

feature_to_plot = None
if selected_features:
    feature_to_plot = st.selectbox(
        "Choose a feature to visualize",
        options=selected_features,
        key="feature_to_plot",
    )

    st.session_state["preprocessing_context"]["feature_to_plot"] = feature_to_plot

    st.plotly_chart(
        make_distribution_comparison(df, processed_df, feature_to_plot),
        use_container_width=True,
        key="distribution_comparison",
    )

    if st.button("🧠 Explain this chart", key="explain_preprocessing_distribution_inline"):
        if not is_ollama_running():
            st.warning("Ollama is not running.")
        else:
            question = (
                f"Explain this before-vs-after distribution chart for dataset '{dataset_name}'. "
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
                    st.session_state["preprocessing_distribution_state"] = {
                        "dataset_name": dataset_name,
                        "feature_to_plot": feature_to_plot,
                        "scaling_method": scaling_method,
                        "selected_features": tuple(selected_features),
                    }
                except Exception as e:
                    st.session_state["preprocessing_distribution_explanation"] = f"Error: {e}"
                    st.session_state["preprocessing_distribution_state"] = {
                        "dataset_name": dataset_name,
                        "feature_to_plot": feature_to_plot,
                        "scaling_method": scaling_method,
                        "selected_features": tuple(selected_features),
                    }

    if st.session_state.get("preprocessing_distribution_explanation"):
        saved_distribution_state = st.session_state.get("preprocessing_distribution_state")
        distribution_is_stale = saved_distribution_state != {
            "dataset_name": dataset_name,
            "feature_to_plot": feature_to_plot,
            "scaling_method": scaling_method,
            "selected_features": tuple(selected_features),
        }

        if distribution_is_stale:
            st.warning("⚠️ Chart changed. Refresh explanation.")

        with st.expander("Chart explanation", expanded=not distribution_is_stale):
            st.write(st.session_state.get("preprocessing_distribution_explanation"))

    st.plotly_chart(
        make_boxplot_comparison(df, processed_df, feature_to_plot),
        use_container_width=True,
        key="boxplot_comparison",
    )

    if st.button("🧠 Explain this chart", key="explain_preprocessing_boxplot_inline"):
        if not is_ollama_running():
            st.warning("Ollama is not running.")
        else:
            question = (
                f"Explain this before-vs-after boxplot for dataset '{dataset_name}'. "
                f"The selected feature is '{feature_to_plot}' and the scaling method is "
                f"'{scaling_method}'. Explain what the box, whiskers, and outliers represent, "
                f"and how they change after scaling."
            )

            with st.spinner("Generating explanation..."):
                try:
                    explanation = ask_tutor(
                        question=question,
                        topic="preprocessing",
                        chat_history=[],
                        model="mistral",
                    )
                    st.session_state["preprocessing_boxplot_explanation"] = explanation
                    st.session_state["preprocessing_boxplot_state"] = {
                        "dataset_name": dataset_name,
                        "feature_to_plot": feature_to_plot,
                        "scaling_method": scaling_method,
                        "selected_features": tuple(selected_features),
                    }
                except Exception as e:
                    st.session_state["preprocessing_boxplot_explanation"] = f"Error: {e}"
                    st.session_state["preprocessing_boxplot_state"] = {
                        "dataset_name": dataset_name,
                        "feature_to_plot": feature_to_plot,
                        "scaling_method": scaling_method,
                        "selected_features": tuple(selected_features),
                    }

    if st.session_state.get("preprocessing_boxplot_explanation"):
        saved_boxplot_state = st.session_state.get("preprocessing_boxplot_state")
        boxplot_is_stale = saved_boxplot_state != {
            "dataset_name": dataset_name,
            "feature_to_plot": feature_to_plot,
            "scaling_method": scaling_method,
            "selected_features": tuple(selected_features),
        }

        if boxplot_is_stale:
            st.warning("⚠️ Chart changed. Refresh explanation.")

        with st.expander("Chart explanation", expanded=not boxplot_is_stale):
            st.write(st.session_state.get("preprocessing_boxplot_explanation"))

    current_distribution_state = {
        "dataset_name": dataset_name,
        "feature_to_plot": feature_to_plot,
        "scaling_method": scaling_method,
        "selected_features": tuple(selected_features),
    }

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
# Summary table
# -------------------------
st.markdown("## 📊 Dataset Summary")
summary_df = summarize_dataframe(df)
st.dataframe(summary_df, use_container_width=True)

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

# ------------------------- 
# Export Cleaned Dataset
# -------------------------
st.markdown("## 💾 Export Processed Dataset")
st.write("Download the cleaned and transformed dataset for use in other tools or modules.")

csv_data = processed_df.to_csv(index=False)
st.download_button(
    label="Download Processed CSV",
    data=csv_data,
    file_name="processed_dataset.csv",
    mime="text/csv",
    key="download_processed_csv",
)

# Store in session for other pages
st.session_state["processed_df"] = processed_df
st.session_state["processed_df_source"] = dataset_name
st.success("Processed dataset stored in session for use in Kernel Trick and PCA Explorer.")