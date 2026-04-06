from pathlib import Path
import sys

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, PowerTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.components.charts import (
    make_correlation_heatmap,
    make_missing_values_bar,
    make_missing_values_comparison,
    make_multi_feature_boxplot_faceted,
    make_multi_feature_boxplot_comparison,
    make_all_numeric_distributions,
    make_multi_feature_distribution_side_by_side,
)
from app.components.dataset_preview import render_dataset_overview
from app.components.preprocessing_progress import (
    init_preprocessing_progress,
    is_step_unlocked,
    is_step_open,
    render_progress_tracker,
    render_step_header,
    render_complete_button,
)
from src.content.preprocessing_theory import (
    render_preprocessing_intro,
    render_data_understanding_theory,
    render_missing_values_theory,
    render_consistency_checks_theory,
    render_duplicate_removal_theory,
    render_outlier_theory,
    render_encoding_theory,
    render_scaling_theory,
    render_standardization_theory,
    render_power_transform_theory,
    render_review_theory,
    render_export_theory,
)
from src.data.sample_datasets import (
    get_available_datasets,
    load_sample_dataset,
    load_uploaded_dataset,
)
from src.features.preprocessing import scale_numeric_features, summarize_dataframe
from app.components.sidebar_tutor import render_sidebar_tutor

st.session_state["current_page"] = "preprocessing"
st.session_state["tutor_context"] = {
    "page": "preprocessing",
    "section": "loading",
    "visible_elements": [
        "preprocessing page header",
        "dataset uploader",
        "sample dataset selector",
    ],
    "hidden_elements": [
        "no kernel plot",
        "no PCA scree plot",
    ],
    "controls": {},
    "chart_summary": "The preprocessing pipeline page is open.",
    "summary": "User is on the preprocessing page.",
}

st.title("🛠️ Preprocessing Pipeline")
st.write("A guided workflow to understand, clean, transform, and export your dataset.")

init_preprocessing_progress()
render_preprocessing_intro()
render_progress_tracker()

# -------------------------
# Step 1: Data Overview
# -------------------------
with st.expander("Step 1 · Data Overview", expanded=is_step_open("overview")):
    render_step_header("overview")

    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"],
        help="If you upload a dataset, it will be used instead of the sample datasets.",
    )

    if uploaded_file is not None:
        try:
            df_raw = load_uploaded_dataset(uploaded_file)
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
        df_raw = load_sample_dataset(dataset_name)

    st.session_state["preprocessing_df_raw"] = df_raw
    st.session_state["preprocessing_dataset_name"] = dataset_name

    render_dataset_overview(df_raw, title="Raw Dataset")

    rows_col, cols_col, missing_col = st.columns(3)
    rows_col.metric("Total Rows", df_raw.shape[0])
    cols_col.metric("Columns", df_raw.shape[1])
    missing_col.metric("Missing Values", int(df_raw.isna().sum().sum()))

    dtypes_df = pd.DataFrame(
        {
            "column": df_raw.columns,
            "dtype": df_raw.dtypes.astype(str).values,
        }
    )
    st.markdown("### Data Types")
    st.dataframe(dtypes_df, use_container_width=True)

    render_complete_button("overview")

if "preprocessing_df_raw" not in st.session_state:
    render_sidebar_tutor()
    st.stop()

df_raw = st.session_state["preprocessing_df_raw"]
dataset_name = st.session_state["preprocessing_dataset_name"]

# -------------------------
# Step 2: Data Understanding
# -------------------------
with st.expander("Step 2 · Data Understanding", expanded=is_step_open("understanding")):
    if not is_step_unlocked("understanding"):
        st.info("Complete Step 1 to unlock this section.")
    else:
        render_step_header("understanding")
        render_data_understanding_theory()

        numeric_cols_raw = df_raw.select_dtypes(include=["number"]).columns.tolist()

        st.markdown("### Missing Values Overview")
        st.plotly_chart(
            make_missing_values_bar(df_raw, title="Missing Values (Before Cleaning)"),
            use_container_width=True,
        )

        if numeric_cols_raw:
            st.markdown("### Correlation Heatmap")
            st.plotly_chart(
                make_correlation_heatmap(df_raw, title="Correlation Heatmap (Before Cleaning)"),
                use_container_width=True,
            )

            st.markdown("### Numeric Feature Distributions")
            st.plotly_chart(
                make_all_numeric_distributions(
                    df_raw,
                    columns=numeric_cols_raw,
                    title="Numeric Feature Distributions (Before Cleaning)",
                ),
                use_container_width=True,
            )

        render_complete_button("understanding")

# -------------------------
# Step 3: Data Cleaning
# -------------------------
with st.expander("Step 3 · Data Cleaning",expanded=is_step_open("cleaning")):
    if not is_step_unlocked("cleaning"):
        st.info("Complete Step 2 to unlock this section.")
    else:
        render_step_header("cleaning")

        df_clean = df_raw.copy()

        st.markdown("### Handle Missing Values")
        render_missing_values_theory()

        missing_option = st.selectbox(
            "Choose a missing-value strategy",
            options=[
                "None",
                "Drop rows with missing values",
                "Fill numeric with mean",
                "Fill numeric with median",
                "Fill categorical with mode",
            ],
            index=0,
            key="missing_option",
        )

        missing_before = int(df_clean.isna().sum().sum())

        numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if missing_option == "Drop rows with missing values":
            df_clean = df_clean.dropna()
        elif missing_option == "Fill numeric with mean":
            for col in numeric_cols:
                if df_clean[col].isna().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif missing_option == "Fill numeric with median":
            for col in numeric_cols:
                if df_clean[col].isna().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif missing_option == "Fill categorical with mode":
            for col in categorical_cols:
                if df_clean[col].isna().any():
                    mode_val = df_clean[col].mode(dropna=True)
                    if not mode_val.empty:
                        df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])

        missing_after = int(df_clean.isna().sum().sum())

        if missing_option == "None":
            st.plotly_chart(
                make_missing_values_bar(df_clean, title="Missing Values"),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                make_missing_values_comparison(df_raw, df_clean),
                use_container_width=True,
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Missing before", missing_before)
        m2.metric("Missing after", missing_after)
        m3.metric("Change", missing_after - missing_before)

        st.markdown("### Consistency Checks")
        render_consistency_checks_theory()

        convert_numeric_cols = st.multiselect(
            "Columns to attempt numeric conversion",
            options=[
                c for c in df_clean.columns
                if str(df_clean[c].dtype) not in ("int64", "float64", "int32", "float32")
            ],
            key="convert_numeric_cols",
        )

        if convert_numeric_cols:
            for col in convert_numeric_cols:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            st.info("Selected columns were converted to numeric where possible.")

        st.markdown("### Duplicate Removal")
        render_duplicate_removal_theory()

        duplicate_before = int(df_clean.duplicated().sum())
        remove_duplicates = st.checkbox(
            "Remove duplicate rows",
            value=False,
            key="remove_duplicates",
        )

        if remove_duplicates:
            df_clean = df_clean.drop_duplicates()

        duplicate_after = int(df_clean.duplicated().sum())
        d1, d2, d3 = st.columns(3)
        d1.metric("Duplicates before", duplicate_before)
        d2.metric("Duplicates after", duplicate_after)
        d3.metric("Removed", duplicate_before - duplicate_after)

        st.session_state["preprocessing_df_clean"] = df_clean
        render_complete_button("cleaning")

if "preprocessing_df_clean" not in st.session_state:
    render_sidebar_tutor()
    st.stop()

df_clean = st.session_state["preprocessing_df_clean"]

# -------------------------
# Step 4: Feature Engineering & Selection
# -------------------------
with st.expander("Step 4 · Feature Engineering & Selection", expanded=is_step_open("engineering")):
    if not is_step_unlocked("engineering"):
        st.info("Complete Step 3 to unlock this section.")
    else:
        render_step_header("engineering")

        df_features = df_clean.copy()
        numeric_cols_clean = df_clean.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols_clean = df_clean.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        st.markdown("### Outlier Detection & Handling")
        render_outlier_theory()

        outlier_cols = st.multiselect(
            "Numeric columns to handle for outliers",
            options=numeric_cols_clean,
            key="outlier_cols",
        )

        outlier_method = st.selectbox(
            "Outlier method",
            options=["None", "Clip to 1st-99th percentile", "Remove outliers (IQR)"],
            index=0,
            key="outlier_method",
        )

        df_before_outlier = df_features.copy()
        rows_before_outlier = len(df_before_outlier)

        if outlier_cols:
            if outlier_method == "None":
                st.plotly_chart(
                    make_multi_feature_boxplot_faceted(
                        df_before_outlier,
                        outlier_cols,
                        title="Current Boxplots",
                    ),
                    use_container_width=True,
                )
            elif outlier_method == "Clip to 1st-99th percentile":
                for col in outlier_cols:
                    lower = df_features[col].quantile(0.01)
                    upper = df_features[col].quantile(0.99)
                    df_features[col] = df_features[col].clip(lower, upper)

                st.plotly_chart(
                    make_multi_feature_boxplot_comparison(
                        df_before_outlier,
                        df_features,
                        outlier_cols,
                        title="Before vs After Outlier Handling",
                    ),
                    use_container_width=True,
                )

            elif outlier_method == "Remove outliers (IQR)":
                keep_mask = pd.Series(True, index=df_features.index)
                for col in outlier_cols:
                    q1 = df_features[col].quantile(0.25)
                    q3 = df_features[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    keep_mask &= df_features[col].between(lower_bound, upper_bound)

                df_features = df_features.loc[keep_mask].copy()

                st.plotly_chart(
                    make_multi_feature_boxplot_comparison(
                        df_before_outlier,
                        df_features,
                        outlier_cols,
                        title="Before vs After Outlier Handling",
                    ),
                    use_container_width=True,
                )

        st.metric("Rows removed by outlier handling", rows_before_outlier - len(df_features))

        st.markdown("### Categorical Encoding")
        render_encoding_theory()

        encode_cols = st.multiselect(
            "Categorical columns to encode",
            options=categorical_cols_clean,
            key="encode_cols",
        )

        encode_method = st.selectbox(
            "Encoding method",
            options=["None", "One-hot encoding", "Label encoding"],
            index=0,
            key="encode_method",
        )

        if encode_method == "One-hot encoding" and encode_cols:
            df_features = pd.get_dummies(df_features, columns=encode_cols, drop_first=True)
        elif encode_method == "Label encoding" and encode_cols:
            for col in encode_cols:
                encoder = LabelEncoder()
                df_features[col] = encoder.fit_transform(df_features[col].astype(str))

              # -------------------------
        # Feature Selection via Correlation
        # -------------------------
        st.markdown("### Feature Selection via Correlation")
        st.info(
            """
### 🧠 How do we use a correlation matrix for feature selection?

A correlation matrix helps us identify **numeric features that are too similar to one another**.

If two features are highly correlated, they may be carrying nearly the same information.
Keeping both can make the model unnecessarily complex and reduce interpretability.

A common workflow is:
1. Compute the correlation matrix
2. Look for feature pairs with high absolute correlation
3. Remove one feature from each highly correlated pair

In practice:
- **|correlation| close to 1** → very strong relationship
- **|correlation| close to 0** → weak relationship

A typical threshold is **0.8 or 0.9**, but there is no single perfect rule.
You might keep the more interpretable feature, the cleaner feature, or the one that is more useful for your task.
"""
        )

        st.caption(
            "By default, the heatmap focuses on continuous numeric features. "
            "Low-cardinality numeric columns such as binary flags or encoded categories are excluded "
            "because they can create misleading correlations."
        )

        include_all_numeric = st.checkbox(
            "Include all numeric features (including low-cardinality / encoded features)",
            value=False,
            key="include_all_numeric_for_corr",
        )

        all_numeric_cols = df_features.select_dtypes(include=["number"]).columns.tolist()

        if include_all_numeric:
            numeric_cols_features = all_numeric_cols
        else:
            numeric_cols_features = [
                col for col in all_numeric_cols
                if df_features[col].nunique(dropna=True) > 10
            ]

        if len(numeric_cols_features) >= 2:
            st.plotly_chart(
                make_correlation_heatmap(
                    df_features[numeric_cols_features],
                    title="Correlation Heatmap",
                ),
                use_container_width=True,
            )

            threshold = st.slider(
                "Correlation threshold",
                min_value=0.50,
                max_value=1.00,
                value=0.80,
                step=0.05,
                key="correlation_threshold",
            )

            corr_matrix = df_features[numeric_cols_features].corr().abs()

            upper_triangle = corr_matrix.where(
                pd.DataFrame(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool),
                    index=corr_matrix.index,
                    columns=corr_matrix.columns,
                )
            )

            high_corr_pairs = []
            for col in upper_triangle.columns:
                for row in upper_triangle.index:
                    corr_value = upper_triangle.loc[row, col]
                    if pd.notna(corr_value) and corr_value > threshold:
                        high_corr_pairs.append((row, col, corr_value))

            if high_corr_pairs:
                st.markdown("#### Highly Correlated Feature Pairs")
                high_corr_df = pd.DataFrame(
                    high_corr_pairs,
                    columns=["Feature 1", "Feature 2", "Absolute Correlation"],
                )
                high_corr_df["Absolute Correlation"] = high_corr_df["Absolute Correlation"].round(3)
                st.dataframe(high_corr_df, use_container_width=True)

                candidate_drop_cols = sorted(
                    set(high_corr_df["Feature 1"]).union(set(high_corr_df["Feature 2"]))
                )

                cols_to_drop = st.multiselect(
                    "Select correlated features to drop",
                    options=candidate_drop_cols,
                    key="correlated_features_to_drop",
                    help="Choose which redundant features to remove from the dataset.",
                )

                if cols_to_drop:
                    df_features = df_features.drop(columns=cols_to_drop)
                    st.success(
                        f"Dropped {len(cols_to_drop)} correlated feature(s): {', '.join(cols_to_drop)}"
                    )
            else:
                st.success("No feature pairs exceed the selected correlation threshold.")
        else:
            st.info(
                "At least two suitable numeric features are needed to compute a useful correlation matrix. "
                "Try enabling the option to include all numeric features if needed."
            )

        st.session_state["preprocessing_df_features"] = df_features
        render_complete_button("engineering")

# -------------------------
# Step 5: Data Transformation
# -------------------------
with st.expander("Step 5 · Data Transformation", expanded=is_step_open("transformation")):
    if not is_step_unlocked("transformation"):
        st.info("Complete Step 4 to unlock this section.")
    else:
        render_step_header("transformation")
        render_scaling_theory()
        render_standardization_theory()
        render_power_transform_theory()

        df_before_transform = df_features.copy()
        df_transformed = df_features.copy()

        numeric_cols_features = df_features.select_dtypes(include=["number"]).columns.tolist()

        transform_cols = st.multiselect(
            "Numeric columns to transform",
            options=numeric_cols_features,
            default=numeric_cols_features[: min(3, len(numeric_cols_features))],
            key="transform_cols",
        )

        scaling_method = st.selectbox(
            "Scaling method",
            options=["none", "standardization", "normalization"],
            index=0,
            key="scaling_method",
        )

        power_method = st.selectbox(
            "Power transform",
            options=["None", "Yeo-Johnson"],
            index=0,
            key="power_method",
        )

        if transform_cols:
            df_transformed = scale_numeric_features(
                df_transformed,
                method=scaling_method,
                exclude_cols=[col for col in numeric_cols_features if col not in transform_cols],
            )

            if power_method == "Yeo-Johnson":
                transformer = PowerTransformer(method="yeo-johnson")
                df_transformed[transform_cols] = transformer.fit_transform(df_transformed[transform_cols])

            transformation_applied = not (scaling_method == "none" and power_method == "None")

            if not transformation_applied:
                st.plotly_chart(
                    make_all_numeric_distributions(
                        df_transformed,
                        columns=transform_cols,
                        title="Current Distributions",
                    ),
                    use_container_width=True,
                )
                st.plotly_chart(
                    make_multi_feature_boxplot_faceted(
                        df_transformed,
                        transform_cols,
                        title="Current Boxplots",
                    ),
                    use_container_width=True,
                )
            else:
                st.plotly_chart(
                    make_multi_feature_distribution_side_by_side(
                        df_before_transform,
                        df_transformed,
                        transform_cols,
                        title="Before vs After Distributions",
                    ),
                    use_container_width=True,
                )

                st.plotly_chart(
                    make_multi_feature_boxplot_comparison(
                        df_before_transform,
                        df_transformed,
                        transform_cols,
                        title="Before vs After Boxplots",
                    ),
                    use_container_width=True,
                )

        st.session_state["processed_df"] = df_transformed
        render_complete_button("transformation")

if "processed_df" not in st.session_state:
    render_sidebar_tutor()
    st.stop()

df_transformed = st.session_state["processed_df"]

# -------------------------
# Step 5: Processing Summary + Export
# -------------------------
st.markdown("## Processing Summary & Export")
render_review_theory()

original_rows, original_cols = df_raw.shape
final_rows, final_cols = df_transformed.shape

remaining_missing = int(df_transformed.isna().sum().sum())
remaining_categorical = df_transformed.select_dtypes(include=["object", "category", "bool"]).shape[1]
remaining_numeric = df_transformed.select_dtypes(include=["number"]).shape[1]

# -------------------------
# High-level metrics
# -------------------------
st.markdown("### 📊 High-Level Summary")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Rows", f"{original_rows} → {final_rows}", delta=final_rows - original_rows)
s2.metric("Columns", f"{original_cols} → {final_cols}", delta=final_cols - original_cols)
s3.metric("Missing Values", remaining_missing)
s4.metric("Numeric Features", remaining_numeric)

# -------------------------
# What was done
# -------------------------
st.markdown("### 🛠️ What was done?")

summary_rows = [
    {"Step": "Missing values", "Selection / Method": st.session_state.get("missing_option", "None")},
    {"Step": "Duplicate removal", "Selection / Method": "Applied" if st.session_state.get("remove_duplicates", False) else "Not applied"},
    {
        "Step": "Outlier handling",
        "Selection / Method": (
            f"{st.session_state.get('outlier_method', 'None')} | "
            f"Columns: {', '.join(st.session_state.get('outlier_cols', [])) if st.session_state.get('outlier_cols') else 'None'}"
        ),
    },
    {
        "Step": "Categorical encoding",
        "Selection / Method": (
            f"{st.session_state.get('encode_method', 'None')} | "
            f"Columns: {', '.join(st.session_state.get('encode_cols', [])) if st.session_state.get('encode_cols') else 'None'}"
        ),
    },
    {
        "Step": "Scaling",
        "Selection / Method": (
            f"{st.session_state.get('scaling_method', 'none')} | "
            f"Columns: {', '.join(st.session_state.get('transform_cols', [])) if st.session_state.get('transform_cols') else 'None'}"
        ),
    },
    {
        "Step": "Power transform",
        "Selection / Method": st.session_state.get("power_method", "None"),
    },
]

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)


# -------------------------
# Readiness checks
# -------------------------
st.markdown("### ✅ Dataset Readiness Check")

messages = []

if remaining_missing == 0:
    messages.append("✅ No missing values remain.")
else:
    messages.append(f"⚠️ {remaining_missing} missing values still exist.")

if remaining_categorical == 0:
    messages.append("✅ All categorical variables have been encoded.")
else:
    messages.append(f"ℹ️ {remaining_categorical} categorical column(s) still present.")

if final_rows > 0 and final_cols > 1:
    messages.append("✅ Dataset has sufficient size for machine learning.")
else:
    messages.append("⚠️ Dataset may be too small for modeling.")

if remaining_numeric >= 2:
    messages.append("✅ Suitable for PCA / feature space methods.")
else:
    messages.append("ℹ️ Not enough numeric features for PCA.")

for msg in messages:
    st.write(msg)

# -------------------------
# Final dataset preview
# -------------------------
st.markdown("### 📄 Final Dataset Preview")
render_dataset_overview(df_transformed, title="Processed Dataset")

st.markdown("### 📈 Final Summary Statistics")
st.dataframe(summarize_dataframe(df_transformed), use_container_width=True)

# -------------------------
# Export (merged here)
# -------------------------
st.markdown("### 📦 Export Processed Dataset")
render_export_theory()

csv_data = df_transformed.to_csv(index=False)

st.download_button(
    label="⬇️ Download Processed CSV",
    data=csv_data,
    file_name="processed_dataset.csv",
    mime="text/csv",
    key="download_processed_csv",
)

# -------------------------
# Store session state
# -------------------------
st.session_state["processed_df"] = df_transformed
st.session_state["processed_df_source"] = dataset_name
st.session_state["current_page"] = "preprocessing"

st.session_state["preprocessing_context"] = {
    "page": "preprocessing",
    "dataset_name": dataset_name,
    "section": "preprocessing pipeline",
    "missing_option": missing_option if "missing_option" in locals() else None,
    "missing_before": missing_before if "missing_before" in locals() else None,
    "missing_after": missing_after if "missing_after" in locals() else None,
    "convert_numeric_cols": convert_numeric_cols if "convert_numeric_cols" in locals() else [],
    "remove_duplicates": remove_duplicates if "remove_duplicates" in locals() else False,
    "duplicate_before": duplicate_before if "duplicate_before" in locals() else None,
    "duplicate_after": duplicate_after if "duplicate_after" in locals() else None,
    "outlier_cols": outlier_cols if "outlier_cols" in locals() else [],
    "outlier_method": outlier_method if "outlier_method" in locals() else None,
    "encode_cols": encode_cols if "encode_cols" in locals() else [],
    "encode_method": encode_method if "encode_method" in locals() else None,
    "transform_cols": transform_cols if "transform_cols" in locals() else [],
    "scaling_method": scaling_method if "scaling_method" in locals() else None,
    "power_method": power_method if "power_method" in locals() else None,
    "final_rows": int(df_transformed.shape[0]) if "df_transformed" in locals() else None,
    "final_cols": int(df_transformed.shape[1]) if "df_transformed" in locals() else None,
    "summary": (
        f"User is on preprocessing. Dataset={dataset_name}. "
        f"Scaling={scaling_method if 'scaling_method' in locals() else 'none'}, "
        f"encoding={encode_method if 'encode_method' in locals() else 'none'}, "
        f"outlier method={outlier_method if 'outlier_method' in locals() else 'none'}."
    ),
}
st.session_state["current_page"] = "preprocessing"
st.session_state["tutor_context"] = {
    "page": "preprocessing",
    "section": "preprocessing pipeline",
    "dataset_name": dataset_name,
    "visible_elements": [
        "dataset preview",
        "preprocessing controls",
        "transformed output",
    ],
    "hidden_elements": [
        "no kernel plot",
        "no PCA scree plot",
    ],
    "controls": {
        "scaling_method": scaling_method if "scaling_method" in locals() else None,
        "encoding_method": encode_method if "encode_method" in locals() else None,
        "outlier_method": outlier_method if "outlier_method" in locals() else None,
    },
    "chart_summary": (
        f"Dataset has {df_transformed.shape[0]} rows and {df_transformed.shape[1]} columns after preprocessing."
        if "df_transformed" in locals() else
        "Preprocessing controls are visible."
    ),
    "summary": "User is working on preprocessing and data transformation.",
}

st.success("✅ Processed dataset is ready and stored for use in other pages.")
render_sidebar_tutor()


