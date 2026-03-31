from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# -------------------------
# Generic helpers
# -------------------------
def _empty_figure(title: str):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _add_k_vline(fig: go.Figure, x, k: int) -> None:
    """Red dashed vertical line marking chosen k — visible in both light and dark mode."""
    fig.add_vline(
        x=x,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"k = {k}",
        annotation_position="top left",
        annotation_font_color="#ef4444",
        annotation_font_size=12,
    )


# -------------------------
# Feature distribution
# -------------------------
def make_feature_distribution(
    df: pd.DataFrame,
    column: str,
    title: str | None = None,
):
    if column not in df.columns:
        raise ValueError(f"{column} not found in dataframe.")

    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=title or f"Distribution of {column}",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


# -------------------------
# Before vs after distribution
# -------------------------
def make_distribution_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    column: str,
    title: str | None = None,
):
    if column not in df_before.columns or column not in df_after.columns:
        raise ValueError(f"{column} must exist in both dataframes.")

    combined_df = pd.DataFrame(
        {
            "value": pd.concat([df_before[column], df_after[column]], ignore_index=True),
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
        title=title or f"Before vs After Distribution: {column}",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


# -------------------------
# Correlation heatmap
# -------------------------
def make_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
):
    numeric_df = df.select_dtypes(include=["number"])

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
        title=title,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


# -------------------------
# Missing values bar chart
# -------------------------
def make_missing_values_bar(
    df: pd.DataFrame,
    title: str = "Missing Values by Column",
):
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
        title=title,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


# -------------------------
# Missing values comparison
# -------------------------
def make_missing_values_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    title: str = "Missing Values: Before vs After",
):
    before_counts = df_before.isna().sum()
    after_counts = df_after.isna().sum()

    columns = sorted(set(before_counts.index).union(after_counts.index))
    comparison_df = pd.DataFrame(
        {
            "column": columns * 2,
            "missing_values": (
                [before_counts.get(col, 0) for col in columns]
                + [after_counts.get(col, 0) for col in columns]
            ),
            "dataset": ["Before"] * len(columns) + ["After"] * len(columns),
        }
    )

    fig = px.bar(
        comparison_df,
        x="column",
        y="missing_values",
        color="dataset",
        barmode="group",
        title=title,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


# -------------------------
# Single boxplot
# -------------------------
def make_single_boxplot(
    df: pd.DataFrame,
    column: str,
    title: str | None = None,
):
    if column not in df.columns:
        raise ValueError(f"{column} not found in dataframe.")

    fig = px.box(
        df,
        y=column,
        title=title or f"Boxplot: {column}",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="",
        yaxis_title=column,
    )
    return fig


# -------------------------
# Boxplot comparison (single feature)
# -------------------------
def make_boxplot_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    column: str,
    title: str | None = None,
):
    if column not in df_before.columns or column not in df_after.columns:
        raise ValueError(f"{column} must exist in both dataframes.")

    combined_df = pd.DataFrame(
        {
            "value": pd.concat([df_before[column], df_after[column]], ignore_index=True),
            "dataset": ["Before"] * len(df_before) + ["After"] * len(df_after),
        }
    )

    fig = px.box(
        combined_df,
        x="dataset",
        y="value",
        color="dataset",
        title=title or f"Boxplot Comparison: {column}",
        category_orders={"dataset": ["Before", "After"]},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig


# -------------------------
# Multi-feature boxplot
# -------------------------
def make_multi_feature_boxplot(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "Boxplots by Feature",
):
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return _empty_figure(title)

    melted = df[valid_cols].melt(var_name="feature", value_name="value")

    fig = px.box(
        melted,
        x="feature",
        y="value",
        title=title,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Feature",
        yaxis_title="Value",
    )
    return fig


# -------------------------
# All numeric distributions (small multiples, separate bins per feature)
# -------------------------
def make_all_numeric_distributions(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Numeric Feature Distributions",
):
    numeric_df = df.select_dtypes(include=["number"]).copy()

    if columns is not None:
        numeric_df = numeric_df[[col for col in columns if col in numeric_df.columns]]

    if numeric_df.empty:
        return _empty_figure(title)

    valid_cols = numeric_df.columns.tolist()
    n_features = len(valid_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    subplot_titles = valid_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for i, col in enumerate(valid_cols):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1

        series = numeric_df[col].dropna()

        fig.add_trace(
            go.Histogram(
                x=series,
                nbinsx=20,
                marker=dict(color="#8ecbff"),
                showlegend=False,
            ),
            row=row,
            col=col_pos,
        )

        fig.update_xaxes(title_text="value", row=row, col=col_pos)
        fig.update_yaxes(title_text="count", row=row, col=col_pos)

    fig.update_layout(
        title=title,
        height=max(350, 280 * n_rows),
        margin=dict(l=20, r=20, t=50, b=20),
        bargap=0.05,
    )

    return fig


# -------------------------
# Multi-feature boxplot with separate facets
# -------------------------
def make_multi_feature_boxplot_faceted(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "Boxplots by Feature",
):
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return _empty_figure(title)

    melted = df[valid_cols].melt(var_name="feature", value_name="value")

    fig = px.box(
        melted,
        y="value",
        facet_col="feature",
        facet_col_wrap=2,
        title=title,
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        height=max(350, 320 * ((len(valid_cols) + 1) // 2)),
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(matches=None)

    return fig


# -------------------------
# Multi-feature boxplot comparison (Before vs After within each feature)
# -------------------------
def make_multi_feature_boxplot_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: list[str],
    title: str = "Before vs After Boxplots by Feature",
):
    valid_cols = [col for col in columns if col in df_before.columns and col in df_after.columns]
    if not valid_cols:
        return _empty_figure(title)

    before_melted = df_before[valid_cols].melt(var_name="feature", value_name="value")
    before_melted["dataset"] = "Before"

    after_melted = df_after[valid_cols].melt(var_name="feature", value_name="value")
    after_melted["dataset"] = "After"

    combined = pd.concat([before_melted, after_melted], ignore_index=True)

    fig = px.box(
        combined,
        x="dataset",
        y="value",
        color="dataset",
        facet_col="feature",
        facet_col_wrap=2,
        title=title,
        category_orders={"dataset": ["Before", "After"]},
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=max(350, 320 * ((len(valid_cols) + 1) // 2)),
        showlegend=False,
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(matches=None)

    return fig


# -------------------------
# Multi-feature distributions: side-by-side before vs after
# -------------------------
def make_multi_feature_distribution_side_by_side(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: list[str],
    title: str = "Before vs After Distributions",
):
    valid_cols = [col for col in columns if col in df_before.columns and col in df_after.columns]
    if not valid_cols:
        return _empty_figure(title)

    n_features = len(valid_cols)
    rows = n_features
    cols = 2

    subplot_titles = []
    for col in valid_cols:
        subplot_titles.extend([f"{col} — Before", f"{col} — After"])

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    for i, col in enumerate(valid_cols, start=1):
        fig.add_trace(
            go.Histogram(
                x=df_before[col],
                name="Before",
                marker=dict(color="#4C78A8"),
                opacity=0.75,
                nbinsx=20,
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=df_after[col],
                name="After",
                marker=dict(color="#F58518"),
                opacity=0.75,
                nbinsx=20,
                showlegend=(i == 1),
            ),
            row=i,
            col=2,
        )

        fig.update_xaxes(title_text=col, row=i, col=1)
        fig.update_xaxes(title_text=col, row=i, col=2)
        fig.update_yaxes(title_text="Count", row=i, col=1)
        fig.update_yaxes(title_text="Count", row=i, col=2)

    fig.update_layout(
        title=title,
        barmode="overlay",
        height=max(350, 300 * n_features),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# -------------------------
# Linear decision boundary
# -------------------------
def make_linear_boundary_line_plot(
    X: np.ndarray,
    y: np.ndarray,
    model,
    title: str,
) -> go.Figure:
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise ValueError("Model must expose coef_ and intercept_.")

    df_plot = pd.DataFrame(X, columns=["x1", "x2"])
    df_plot["target"] = y.astype(str)

    fig = px.scatter(
        df_plot,
        x="x1",
        y="x2",
        color="target",
        title=title,
    )

    w = model.coef_[0]
    b = model.intercept_[0]

    x_min, x_max = df_plot["x1"].min() - 0.15, df_plot["x1"].max() + 0.15
    y_min, y_max = df_plot["x2"].min() - 0.15, df_plot["x2"].max() + 0.15

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    # NEW: detect when the linear model has essentially no meaningful direction
    w_norm = np.linalg.norm(w)

    if w_norm < 1e-3:
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text="No meaningful straight-line boundary exists",
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(
            height=460,
            margin=dict(l=0, r=0, t=50, b=0),
            legend_title_text="",
        )
        return fig

    x_vals = np.linspace(x_min, x_max, 200)

    # safer threshold than 1e-8
    if abs(w[1]) > 1e-4:
        y_vals = -(w[0] * x_vals + b) / w[1]

        # keep only visible part of the boundary
        mask = (y_vals >= y_min - 1) & (y_vals <= y_max + 1)

        fig.add_trace(
            go.Scatter(
                x=x_vals[mask],
                y=y_vals[mask],
                mode="lines",
                name="Linear boundary",
                line=dict(width=3, dash="dash"),
            )
        )
    elif abs(w[0]) > 1e-4:
        x_boundary = -b / w[0]
        fig.add_trace(
            go.Scatter(
                x=[x_boundary, x_boundary],
                y=[y_min, y_max],
                mode="lines",
                name="Linear boundary",
                line=dict(width=3, dash="dash"),
            )
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text="No meaningful straight-line boundary exists",
            showarrow=False,
            font=dict(size=14),
        )

    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=50, b=0),
        legend_title_text="",
    )
    return fig


# -------------------------
# Kernel boundary
# -------------------------
def make_kernel_boundary_plot(model, raw_df, mapping_fn, title: str):
    import numpy as np

    x_min, x_max = raw_df["x1"].min() - 0.5, raw_df["x1"].max() + 0.5
    y_min, y_max = raw_df["x2"].min() - 0.5, raw_df["x2"].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    grid_df = pd.DataFrame(
        {
            "x1": xx.ravel(),
            "x2": yy.ravel(),
        }
    )

    mapped_grid = mapping_fn(grid_df)
    z = model.predict(mapped_grid.values).reshape(xx.shape)

    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 200),
            y=np.linspace(y_min, y_max, 200),
            z=z,
            showscale=False,
            opacity=0.3,
        )
    )

    for cls in raw_df["target"].unique():
        subset = raw_df[raw_df["target"] == cls]
        fig.add_trace(
            go.Scatter(
                x=subset["x1"],
                y=subset["x2"],
                mode="markers",
                name=f"Class {cls}",
            )
        )

    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# -------------------------
# PCA Scree + cumulative variance
# -------------------------
def make_pca_scree_plot(
    explained_variance: list[float],
    cumulative_variance: list[float],
    n_components: int,
    title: str = "Scree Plot",
):
    x_vals = list(range(1, len(explained_variance) + 1))
    tick_labels = [f"PC{i}" for i in x_vals]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=explained_variance,
            name="Individual variance",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=cumulative_variance,
            mode="lines+markers",
            name="Cumulative variance",
            yaxis="y2",
        )
    )

    if 1 <= n_components <= len(x_vals):
        fig.add_vline(
            x=n_components,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"k={n_components}",
            annotation_position="top left",
            annotation_font_color="#ef4444",
        )

    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            title="Principal Component",
            tickmode="array",
            tickvals=x_vals,
            ticktext=tick_labels,
        ),
        yaxis=dict(title="Variance ratio"),
        yaxis2=dict(
            title="Cumulative variance",
            overlaying="y",
            side="right",
            range=[0, 1.05],
        ),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# -------------------------
# PCA Loadings Bar  (horizontal diverging, rounded)
# -------------------------
def make_pca_loadings_bar(
    loadings: np.ndarray,
    feature_names: list[str],
    pc_index: int,
    explained_variance: float | None = None,
    title: str | None = None,
) -> go.Figure:
    """
    Horizontal diverging bar chart.
    Positive loadings → cyan (#22d3ee), negative → purple (#a855f7).
    Bars are sorted by absolute loading (largest at top).
    """
    values = loadings[pc_index]
    df = pd.DataFrame({"feature": feature_names, "loading": values})
    df = df.reindex(df["loading"].abs().sort_values(ascending=True).index)  # ascending=True → largest at top in horizontal

    colors = ["#22d3ee" if v >= 0 else "#a855f7" for v in df["loading"]]
    text = [f"{v:+.2f}" for v in df["loading"]]
    text_positions = ["outside" if v >= 0 else "outside" for v in df["loading"]]

    pct_str = f"  —  explains {explained_variance * 100:.1f}% variance" if explained_variance is not None else ""
    chart_title = title or f"Feature Contributions to PC{pc_index + 1}{pct_str}"

    fig = go.Figure(go.Bar(
        x=df["loading"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=text,
        textposition="outside",
        textfont=dict(size=11, color="#94a3b8"),
        cliponaxis=False,
    ))

    max_abs = max(abs(df["loading"].max()), abs(df["loading"].min())) * 1.35

    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=13, color="#e2e8f0")),
        margin=dict(l=10, r=60, t=50, b=40),
        height=max(320, len(df) * 32 + 100),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[-max_abs, max_abs],
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.15)",
            zerolinewidth=1.5,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#64748b"),
            showline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#cbd5e1", size=12),
            gridcolor="rgba(0,0,0,0)",
        ),
        bargap=0.35,
        annotations=[
            dict(x=-max_abs * 0.98, y=-0.08, xref="x", yref="paper",
                 text="← negative", showarrow=False,
                 font=dict(color="#a855f7", size=11)),
            dict(x=max_abs * 0.98, y=-0.08, xref="x", yref="paper",
                 text="positive →", showarrow=False,
                 font=dict(color="#22d3ee", size=11)),
        ],
    )

    # Rounded bar ends via marker.line trick — use shape 'spline' in bar is not available,
    # but we overlay invisible scatter with large marker to fake rounded caps
    fig.update_traces(marker_line_width=0)

    return fig


# -------------------------
# PCA Loadings Table  (replaces heatmap — styled, readable)
# -------------------------
def make_pca_loadings_heatmap(
    loadings: np.ndarray,
    feature_names: list[str],
    n_components: int,
    title: str = "Loadings Heatmap",
) -> go.Figure:
    """
    Styled table showing loading values per feature × PC.
    Cells are colour-coded: cyan intensity for positive, purple for negative.
    Much more readable than a Plotly imshow heatmap.
    """
    subset = loadings[:n_components]          # shape (n_components, n_features)
    pc_labels = [f"PC{i+1}" for i in range(n_components)]

    # Build cell colours and formatted text
    def _cell_colour(v: float) -> str:
        intensity = min(abs(v), 1.0)
        if v >= 0:
            r = int(34  + (0   - 34)  * intensity)
            g = int(211 + (150 - 211) * intensity)
            b = int(238 + (200 - 238) * intensity)
        else:
            r = int(168 + (120 - 168) * intensity)
            g = int(85  + (40  - 85)  * intensity)
            b = int(247 + (220 - 247) * intensity)
        return f"rgba({r},{g},{b},{0.15 + 0.65 * intensity:.2f})"

    # One column per PC
    cell_values: list[list[str]] = []
    cell_colors: list[list[str]] = []

    for pc_idx in range(n_components):
        col_vals = []
        col_cols = []
        for feat_idx in range(len(feature_names)):
            v = float(subset[pc_idx, feat_idx])
            col_vals.append(f"{v:+.2f}")
            col_cols.append(_cell_colour(v))
        cell_values.append(col_vals)
        cell_colors.append(col_cols)

    header_values = ["<b>Feature</b>"] + [f"<b>{p}</b>" for p in pc_labels]
    col_values = [feature_names] + cell_values
    col_colors = [["rgba(30,41,59,0.9)"] * len(feature_names)] + cell_colors

    fig = go.Figure(go.Table(
        columnwidth=[2.5] + [1.2] * n_components,
        header=dict(
            values=header_values,
            fill_color="rgba(15,23,42,0.95)",
            font=dict(color=["#94a3b8"] + ["#22d3ee"] * n_components, size=12, family="monospace"),
            align=["left"] + ["center"] * n_components,
            line_color="rgba(255,255,255,0.08)",
            height=36,
        ),
        cells=dict(
            values=col_values,
            fill_color=col_colors,
            font=dict(color="#e2e8f0", size=12, family="monospace"),
            align=["left"] + ["center"] * n_components,
            line_color="rgba(255,255,255,0.05)",
            height=34,
        ),
    ))

    n_rows = len(feature_names)
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:11px;letter-spacing:0.1em;color:#64748b'>"
                 f"LOADINGS TABLE — ALL {n_components} COMPONENT{'S' if n_components > 1 else ''}</span>",
            x=0, xanchor="left",
        ),
        margin=dict(l=0, r=0, t=44, b=8),
        height=n_rows * 34 + 90,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# -------------------------
# Correlation circle  (2D biplot of loadings)
# -------------------------
def make_correlation_circle(
    loadings: np.ndarray,
    feature_names: list[str],
    explained_variance: list[float],
    pc_x: int = 0,
    pc_y: int = 1,
) -> go.Figure:
    """
    Classic correlation circle (variable factor map).
    Each arrow = one feature. Arrow direction = loading direction in PC space.
    Arrow length = how well that feature is represented by these two PCs.
    Features pointing in the same direction are positively correlated.
    """
    lx = loadings[pc_x]
    ly = loadings[pc_y]
    pct_x = explained_variance[pc_x] * 100
    pct_y = explained_variance[pc_y] * 100

    fig = go.Figure()

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode="lines",
        line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    # Axes
    for xy, axis_label in [(([-1.15, 1.15], [0, 0]), f"PC{pc_x+1}"),
                            (([0, 0], [-1.15, 1.15]), f"PC{pc_y+1}")]:
        fig.add_trace(go.Scatter(
            x=xy[0], y=xy[1], mode="lines",
            line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))

    # Arrows + labels
    for i, name in enumerate(feature_names):
        x_end, y_end = float(lx[i]), float(ly[i])
        length = (x_end**2 + y_end**2) ** 0.5
        alpha = 0.55 + 0.45 * min(length, 1.0)

        fig.add_annotation(
            ax=0, ay=0, x=x_end, y=y_end,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.2, arrowwidth=1.8,
            arrowcolor=f"rgba(34,211,238,{alpha:.2f})",
            showarrow=True,
        )
        # Label offset slightly beyond tip
        offset = 0.10
        fig.add_trace(go.Scatter(
            x=[x_end * (1 + offset / max(length, 0.01))],
            y=[y_end * (1 + offset / max(length, 0.01))],
            mode="text",
            text=[name],
            textfont=dict(size=11, color=f"rgba(226,232,240,{alpha:.2f})"),
            showlegend=False,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"PC{pc_x+1} loading: {x_end:+.3f}<br>"
                f"PC{pc_y+1} loading: {y_end:+.3f}<br>"
                f"Representation: {length**2*100:.1f}%"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:11px;letter-spacing:.1em;color:#64748b'>"
                 f"CORRELATION CIRCLE (PC{pc_x+1} VS PC{pc_y+1})</span>",
            x=0, xanchor="left",
        ),
        xaxis=dict(
        title=dict(
            text=f"PC{pc_x+1} ({pct_x:.1f}%)",
            font=dict(color="#e2e8f0"),
        ),
        tickfont=dict(color="#94a3b8"),
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.15)",
        ),
        yaxis=dict(
            title=dict(
                text=f"PC{pc_y+1} ({pct_y:.1f}%)",
                font=dict(color="#e2e8f0"),
            ),
            tickfont=dict(color="#94a3b8"),
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.15)",
        ),
        margin=dict(l=50, r=20, t=44, b=50),
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
    )
    return fig


# -------------------------
# 3D Correlation space explorer
# -------------------------
def make_correlation_3d(
    loadings: np.ndarray,
    feature_names: list[str],
    explained_variance: list[float],
    scores: np.ndarray | None = None,
    label_values: list[str] | None = None,
) -> go.Figure:
    """
    3D biplot: PC1 / PC2 / PC3 axes.
    Loading vectors (arrows via cones) + optionally the score cloud.
    """
    if loadings.shape[0] < 3:
        fig = go.Figure()
        fig.update_layout(
            title="Need at least 3 components for 3D explorer",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    lx, ly, lz = loadings[0], loadings[1], loadings[2]
    pcts = [explained_variance[i] * 100 for i in range(3)]

    fig = go.Figure()

    # Optional: score cloud (data points projected into PC space)
    if scores is not None and scores.shape[1] >= 3:
        if label_values is not None:
            unique_labels = sorted(set(label_values))
            palette = [
                "#22d3ee", "#a78bfa", "#f97316", "#4ade80",
                "#f43f5e", "#facc15", "#60a5fa", "#fb7185"
            ]
            color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
            colors = [color_map[label] for label in label_values]
        else:
            colors = ["#22d3ee"] * len(scores)

        fig.add_trace(
            go.Scatter3d(
                x=scores[:, 0],
                y=scores[:, 1],
                z=scores[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=colors,
                    opacity=0.45,
                ),
                name="Data points",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Loading vectors as cones at the tip
    scale = 3.0
    for i, name in enumerate(feature_names):
        x_end = float(lx[i]) * scale
        y_end = float(ly[i]) * scale
        z_end = float(lz[i]) * scale
        length = (x_end**2 + y_end**2 + z_end**2) ** 0.5
        alpha = max(0.4, min(length / (scale * 0.8), 1.0))

        fig.add_trace(
            go.Scatter3d(
                x=[0, x_end],
                y=[0, y_end],
                z=[0, z_end],
                mode="lines+text",
                line=dict(color=f"rgba(34,211,238,{alpha:.2f})", width=3),
                text=["", name],
                textfont=dict(size=10, color="#e2e8f0"),
                textposition="top center",
                name=name,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"PC1: {lx[i]:+.3f}<br>"
                    f"PC2: {ly[i]:+.3f}<br>"
                    f"PC3: {lz[i]:+.3f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Cone(
                x=[x_end],
                y=[y_end],
                z=[z_end],
                u=[float(lx[i]) * 0.3],
                v=[float(ly[i]) * 0.3],
                w=[float(lz[i]) * 0.3],
                sizemode="absolute",
                sizeref=0.25,
                colorscale=[
                    [0, f"rgba(34,211,238,{alpha:.2f})"],
                    [1, f"rgba(34,211,238,{alpha:.2f})"],
                ],
                showscale=False,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text="<span style='font-size:11px;letter-spacing:.1em;color:#64748b'>"
                 "3D CORRELATION SPACE EXPLORER</span>",
            x=0,
            xanchor="left",
        ),
        scene=dict(
            xaxis=dict(
                title=f"PC1 ({pcts[0]:.1f}%)",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.2)",
            ),
            yaxis=dict(
                title=f"PC2 ({pcts[1]:.1f}%)",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.2)",
            ),
            zaxis=dict(
                title=f"PC3 ({pcts[2]:.1f}%)",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.2)",
            ),
            bgcolor="rgba(15,23,42,0.8)",
        ),
        margin=dict(l=0, r=0, t=44, b=0),
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# -------------------------
# PCA scores plot
# -------------------------
def make_pca_scores_plot(
    scores: np.ndarray,
    label_values: list[str] | None = None,
    label_name: str | None = None,
    pc_x: int = 0,
    pc_y: int = 1,
    title: str = "Scores Plot",
) -> go.Figure:
    df = pd.DataFrame(
        {
            f"PC{pc_x+1}": scores[:, pc_x],
            f"PC{pc_y+1}": scores[:, pc_y],
        }
    )

    color_col = None
    if label_values is not None and len(label_values) == len(df):
        df["label"] = label_values
        color_col = "label"

    fig = px.scatter(
        df,
        x=f"PC{pc_x+1}",
        y=f"PC{pc_y+1}",
        color=color_col,
        title=title,
        labels={"label": label_name or "Label"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


# -------------------------
# Reconstruction Plot
# -------------------------
def make_reconstruction_plot(
    reconstruction_errors: list[float],
    cumulative_variance: list[float],
    n_components: int,
    title: str = "Reconstruction Error vs Components",
):
    ks = list(range(1, len(reconstruction_errors) + 1))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ks,
            y=reconstruction_errors,
            mode="lines+markers",
            name="Reconstruction error",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ks,
            y=cumulative_variance,
            mode="lines+markers",
            name="Cumulative variance",
            yaxis="y2",
        )
    )

    fig.add_vline(
        x=n_components,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"k={n_components}",
        annotation_position="top left",
        annotation_font_color="#ef4444",
    )

    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(title="Number of components"),
        yaxis=dict(title="Reconstruction error"),
        yaxis2=dict(
            title="Cumulative variance",
            overlaying="y",
            side="right",
            range=[0, 1.05],
        ),
        legend=dict(orientation="h", y=1.12),
    )
    return fig