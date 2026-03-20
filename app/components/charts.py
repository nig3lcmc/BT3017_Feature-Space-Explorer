from __future__ import annotations

import pandas as pd
import plotly.express as px


# -------------------------
# Feature distribution
# -------------------------
def make_feature_distribution(df: pd.DataFrame, column: str):
    """
    Histogram for a single feature.
    Used for BEFORE vs AFTER comparison.
    """
    if column not in df.columns:
        raise ValueError(f"{column} not found in dataframe.")

    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=f"Distribution of {column}",
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


# -------------------------
# Correlation heatmap
# -------------------------
def make_correlation_heatmap(df: pd.DataFrame):
    """
    Correlation heatmap for numeric features.
    """
    numeric_df = df.select_dtypes(include=["number"])

    # Handle edge case: not enough numeric columns
    if numeric_df.shape[1] < 2:
        return px.imshow(
            [[0]],
            text_auto=True,
            title="Not enough numeric features for correlation",
        )

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


# -------------------------
# Missing values bar chart
# -------------------------
def make_missing_values_bar(df: pd.DataFrame):
    """
    Bar chart showing missing values per column.
    """
    missing_counts = df.isna().sum()

    missing_df = pd.DataFrame(
        {
            "column": missing_counts.index,
            "missing_values": missing_counts.values,
        }
    )

    fig = px.bar(
        missing_df,
        x="column",
        y="missing_values",
        title="Missing Values by Column",
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


# -------------------------
# NEW: Compare distributions (overlay)
# -------------------------
def make_distribution_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    column: str,
):
    """
    Overlay histogram to compare BEFORE vs AFTER scaling.
    """
    if column not in df_before.columns or column not in df_after.columns:
        raise ValueError(f"{column} must exist in both dataframes.")

    combined_df = pd.DataFrame(
        {
            "value": pd.concat([df_before[column], df_after[column]]),
            "dataset": ["Before"] * len(df_before) + ["After"] * len(df_after),
        }
    )

    fig = px.histogram(
        combined_df,
        x="value",
        color="dataset",
        barmode="overlay",
        nbins=30,
        opacity=0.6,
        title=f"Before vs After Distribution: {column}",
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


# -------------------------
# NEW: Boxplot comparison
# -------------------------
def make_boxplot_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    column: str,
):
    """
    Boxplot comparison for BEFORE vs AFTER scaling.
    Helps visualize spread and outliers.
    """
    combined_df = pd.DataFrame(
        {
            "value": pd.concat([df_before[column], df_after[column]]),
            "dataset": ["Before"] * len(df_before) + ["After"] * len(df_after),
        }
    )

    fig = px.box(
        combined_df,
        x="dataset",
        y="value",
        title=f"Boxplot Comparison: {column}",
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig