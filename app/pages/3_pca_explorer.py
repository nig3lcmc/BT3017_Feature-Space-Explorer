from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.components.charts import (
    make_pca_scree_plot,
    make_pca_loadings_bar,
    make_pca_loadings_heatmap,
    make_pca_scores_plot,
    make_reconstruction_plot,
    make_correlation_circle,
    make_correlation_3d,
)
from app.components.sidebar_tutor import render_sidebar_tutor
from src.content.pca_theory import render_pca_theory_panel
from src.data.sample_datasets import (
    get_available_datasets,
    load_sample_dataset,
    load_uploaded_dataset,
)
from src.llm.context_builder import set_page_context
from src.pca.pipeline import run_pca


# -------------------------------------------------------------------
# Page bootstrapping for tutor
# -------------------------------------------------------------------
st.session_state["current_page"] = "pca"

# Minimal safe context first so the tutor always knows the page,
# even before a dataset is loaded.
set_page_context(
    page="pca",
    section="loading",
    visible_elements=["dataset uploader", "theory panel"],
    hidden_elements=["no PCA chart visible until a dataset is loaded"],
    controls={},
    chart_summary="The PCA page is open, but no dataset has been loaded yet.",
    notes=["Do not describe a scree plot unless a dataset has been loaded."],
)

render_sidebar_tutor()


TABS = [
    {"id": "variance", "label": "1 · Variance", "desc": "How much does each component explain?"},
    {"id": "features", "label": "2 · Features", "desc": "Which features drive each component?"},
    {"id": "scores", "label": "3 · Data Points", "desc": "Where does each data point land in PC space?"},
    {"id": "reconstruct", "label": "4 · Reconstruction", "desc": "How much information is lost?"},
]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _compute_k_rules(
    explained_variance: list[float],
    cumulative_variance: list[float],
    eigenvalues: list[float],
) -> dict:
    n = len(explained_variance)

    k_90 = next((i + 1 for i, cv in enumerate(cumulative_variance) if cv >= 0.90), n)
    k_80 = next((i + 1 for i, cv in enumerate(cumulative_variance) if cv >= 0.80), n)

    kaiser_k = sum(1 for ev in eigenvalues if ev >= 1.0)
    kaiser_k = max(1, kaiser_k)

    if len(explained_variance) >= 3:
        drops = [
            explained_variance[i] - explained_variance[i + 1]
            for i in range(len(explained_variance) - 1)
        ]
        elbow_k = int(drops.index(max(drops))) + 1
        elbow_k = max(1, min(elbow_k, n))
    else:
        elbow_k = 1

    return {"k_90": k_90, "k_80": k_80, "kaiser_k": kaiser_k, "elbow_k": elbow_k}


def _k_guidance(
    explained_variance: list[float],
    cumulative_variance: list[float],
    eigenvalues: list[float],
    k: int,
    context: str = "variance",
) -> None:
    rules = _compute_k_rules(explained_variance, cumulative_variance, eigenvalues)
    k_90, k_80 = rules["k_90"], rules["k_80"]
    kaiser_k, elbow_k = rules["kaiser_k"], rules["elbow_k"]

    current_cv = cumulative_variance[k - 1] * 100
    current_ev = explained_variance[k - 1] * 100

    if current_cv >= 90:
        icon, color, verdict = "✅", "#4ade80", "Strong choice — captures ≥ 90% of variance."
    elif current_cv >= 80:
        icon, color, verdict = "⚡", "#facc15", f"Good — raise to k = {k_90} for ≥ 90% coverage."
    else:
        icon, color, verdict = "⚠️", "#f87171", f"Low coverage — consider raising k to {k_90}."

    if context == "variance":
        tip = (
            "**Reading the scree plot:** each bar = one component's individual variance. "
            "The cumulative line (right axis) shows the running total. "
            "Look for the **elbow** — where the bars stop dropping steeply. "
            "The red line shows your current k."
        )
    else:
        tip = (
            "**Reading the reconstruction plot:** the error line falls as k grows. "
            "The variance line rises on the right axis. "
            "Look for where the **error curve flattens** — adding more components there "
            "buys little improvement. The red line marks your current k."
        )

    st.markdown(tip)

    cv_at_90 = cumulative_variance[k_90 - 1] * 100
    cv_at_80 = cumulative_variance[k_80 - 1] * 100
    cv_kaiser = cumulative_variance[kaiser_k - 1] * 100
    cv_elbow = cumulative_variance[elbow_k - 1] * 100

    rows = [
        ("90% variance threshold", k_90, f"{cv_at_90:.1f}%", "Most common practical rule"),
        ("80% variance threshold", k_80, f"{cv_at_80:.1f}%", "Lighter compression, acceptable loss"),
        ("Kaiser criterion (λ ≥ 1)", kaiser_k, f"{cv_kaiser:.1f}%", "Keep components above average variance"),
        ("Scree elbow", elbow_k, f"{cv_elbow:.1f}%", "Diminishing returns beyond this point"),
    ]

    header = "| Rule | Suggested k | Variance captured | Why |\n|---|:---:|:---:|---|\n"
    body = "".join(
        f"| {'**' + rule + '**' if suggested == k else rule} "
        f"| {'**k = ' + str(suggested) + '**' if suggested == k else 'k = ' + str(suggested)} "
        f"| {cv} | {why} |\n"
        for rule, suggested, cv, why in rows
    )
    st.markdown(header + body)

    st.markdown(
        f"""
        <div style="
            border-radius:12px;padding:12px 18px;margin:4px 0 16px 0;
            border:1px solid {color}55;background:{color}10;
            display:flex;align-items:center;gap:10px;
        ">
            <span style="font-size:1.3rem;line-height:1;">{icon}</span>
            <div>
                <strong style="color:{color};">k = {k}</strong>
                <span style="color:#cbd5e1;"> captures
                    <strong>{current_cv:.1f}%</strong> of total variance
                    (PC{k} alone contributes <strong>{current_ev:.1f}%</strong>).
                </span>
                <span style="color:{color};"> {verdict}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str):
    st.markdown(
        f"""
        <div style="
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 10px;
            margin-top: 6px;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tab_buttons():
    if "pca_active_tab" not in st.session_state:
        st.session_state["pca_active_tab"] = "variance"

    cols = st.columns(4)
    for col, tab in zip(cols, TABS):
        active = st.session_state["pca_active_tab"] == tab["id"]
        if col.button(
            f"{tab['label']}\n{tab['desc']}",
            key=f"tab_{tab['id']}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            if st.session_state["pca_active_tab"] != tab["id"]:
                st.session_state["pca_active_tab"] = tab["id"]
                st.rerun()


# -------------------------------------------------------------------
# Page header
# -------------------------------------------------------------------
st.title("PCA Explorer")
st.write(
    "Upload your data or try a demo. PCA finds the directions of maximum variance — "
    "helping you understand structure, correlations, and redundancy in your features."
)

render_pca_theory_panel()


# -------------------------------------------------------------------
# Data loader
# -------------------------------------------------------------------
section_label("Load your dataset")

uploaded_file = st.file_uploader(
    "Drop a CSV file here",
    type=["csv"],
    help="Numeric features are used for PCA. Non-numeric columns may be used as labels for colouring.",
)

demo_dataset = None
if uploaded_file is None:
    section_label("Or try a demo dataset")
    demo_dataset = st.selectbox(
        "Choose a demo dataset",
        options=[""] + get_available_datasets(),
        index=0,
        format_func=lambda x: "Select a demo dataset" if x == "" else x.title(),
    )

df = None
dataset_name = None

if uploaded_file is not None:
    try:
        df = load_uploaded_dataset(uploaded_file)
        dataset_name = f"uploaded:{uploaded_file.name}"
        st.success(f"Loaded uploaded dataset: {uploaded_file.name}")
    except Exception as exc:
        st.error(str(exc))
        st.stop()
elif demo_dataset:
    df = load_sample_dataset(demo_dataset)
    dataset_name = demo_dataset

if df is None:
    set_page_context(
        page="pca",
        section="loading",
        visible_elements=["dataset uploader", "demo dataset selector"],
        hidden_elements=["no PCA charts visible"],
        controls={},
        chart_summary="No dataset has been loaded yet, so no PCA output is visible.",
        notes=["Do not talk about a scree plot until a dataset exists."],
    )
    st.info("Load a dataset to begin. Try Iris or Wine above.")
    st.stop()

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if not numeric_cols:
    set_page_context(
        page="pca",
        section="loading error",
        visible_elements=["uploaded dataset"],
        hidden_elements=["no PCA charts visible"],
        controls={"dataset_name": dataset_name},
        chart_summary="The uploaded dataset has no numeric columns, so PCA cannot be run.",
        notes=["Say that PCA needs numeric columns."],
    )
    st.error("No numeric columns found for PCA.")
    st.stop()

non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

label_candidates = []
if "target_name" in df.columns:
    label_candidates.append("target_name")
if "target" in df.columns:
    label_candidates.append("target")
for c in non_numeric_cols:
    if c not in label_candidates:
        label_candidates.append(c)


# -------------------------------------------------------------------
# Run PCA
# -------------------------------------------------------------------
pca_result = run_pca(df, exclude_cols=[])

max_k = min(pca_result.d, pca_result.n - 1, 20)
if "pca_k" not in st.session_state:
    st.session_state["pca_k"] = min(2, max_k)
if st.session_state["pca_k"] > max_k:
    st.session_state["pca_k"] = max_k

k = st.slider(
    "Number of components (k)",
    min_value=1,
    max_value=max_k,
    value=st.session_state["pca_k"],
)
st.session_state["pca_k"] = k

if "pca_selected_pc" not in st.session_state:
    st.session_state["pca_selected_pc"] = 0
if st.session_state["pca_selected_pc"] >= k:
    st.session_state["pca_selected_pc"] = 0

if "pca_label_col" not in st.session_state:
    st.session_state["pca_label_col"] = label_candidates[0] if label_candidates else None

label_col = st.session_state["pca_label_col"]
label_values = (
    df[label_col].astype(str).tolist()
    if label_col is not None and label_col in df.columns
    else None
)

# Summary metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Samples", pca_result.n)
m2.metric("Original Dims", pca_result.d)
m3.metric("Components (k)", k)
m4.metric("Variance Kept", f"{pca_result.cumulative_variance[k - 1] * 100:.1f}%")

# Tabs
render_tab_buttons()
active_tab = st.session_state["pca_active_tab"]

st.markdown(
    """
    <div style="
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 22px;
        margin-top: 18px;
        background: rgba(255,255,255,0.02);
    ">
    """,
    unsafe_allow_html=True,
)

if active_tab == "variance":
    st.subheader("How much does each component explain?")

    _k_guidance(
        pca_result.explained_variance,
        pca_result.cumulative_variance,
        pca_result.eigenvalues,
        k,
        context="variance",
    )

    st.plotly_chart(
        make_pca_scree_plot(
            pca_result.explained_variance,
            pca_result.cumulative_variance,
            n_components=k,
            title="Scree Plot",
        ),
        use_container_width=True,
    )

    st.markdown("#### Eigenvalue table")
    st.caption(
        "Rows at or below the red line (your current k) are the components you are keeping. "
        "Kaiser criterion keeps rows where λ ≥ 1."
    )

    variance_df = pd.DataFrame(
        {
            "PC": [f"PC{i+1}" for i in range(len(pca_result.eigenvalues))],
            "Eigenvalue (λ)": [round(v, 4) for v in pca_result.eigenvalues],
            "λ ≥ 1 (Kaiser)": ["✅" if v >= 1.0 else "—" for v in pca_result.eigenvalues],
            "Variance %": [round(v * 100, 2) for v in pca_result.explained_variance],
            "Cumulative %": [round(v * 100, 2) for v in pca_result.cumulative_variance],
            "Kept (k)": ["✅" if i < k else "—" for i in range(len(pca_result.eigenvalues))],
        }
    )
    st.dataframe(variance_df, use_container_width=True, hide_index=True)

elif active_tab == "features":
    st.subheader("Which features drive each component?")
    st.markdown(
        "<span style='color:#22d3ee;font-weight:600;'>Loadings</span> show how strongly each original feature "
        "contributes to a PC. A large positive loading means the feature pushes data in that direction; "
        "negative means the opposite.",
        unsafe_allow_html=True,
    )

    pc_options = [f"PC{i+1}" for i in range(k)]
    sel_label = st.session_state.get("pca_selected_pc_label", pc_options[0])
    if sel_label not in pc_options:
        sel_label = pc_options[0]

    pc_pct = pca_result.explained_variance

    header_cols = st.columns([2, 3])
    with header_cols[0]:
        st.markdown("**Inspect PC:**")
    with header_cols[1]:
        st.markdown(
            f"<span style='color:#94a3b8;font-family:monospace;font-size:.9rem;'>"
            f"explains <b style='color:#e2e8f0;'>{pc_pct[pc_options.index(sel_label)] * 100:.1f}%</b> variance"
            f"</span>",
            unsafe_allow_html=True,
        )

    pill_cols = st.columns(min(k, 10) + 1)
    for i, pc_label in enumerate(pc_options):
        is_sel = pc_label == sel_label
        if pill_cols[i].button(
            pc_label,
            key=f"pc_pill_{pc_label}",
            type="primary" if is_sel else "secondary",
        ):
            if sel_label != pc_label:
                st.session_state["pca_selected_pc_label"] = pc_label
                st.rerun()

    selected_pc = pc_options.index(sel_label)
    st.session_state["pca_selected_pc"] = selected_pc

    st.plotly_chart(
        make_pca_loadings_bar(
            pca_result.loadings,
            pca_result.feature_names,
            pc_index=selected_pc,
            explained_variance=pca_result.explained_variance[selected_pc],
        ),
        use_container_width=True,
    )

    st.divider()

    st.markdown(
        f"<span style='font-size:.75rem;letter-spacing:.1em;font-weight:700;"
        f"text-transform:uppercase;color:#64748b'>Component Loadings</span><br>"
        f"<span style='font-size:.85rem;color:#94a3b8;'>How much each original feature contributes to each PC. "
        f"<span style='color:#22d3ee'>Cyan = positive</span> · "
        f"<span style='color:#a855f7'>Purple = negative</span></span>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_pca_loadings_heatmap(
            pca_result.loadings,
            pca_result.feature_names,
            n_components=k,
        ),
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Feature correlation space")
    st.write(
        "Arrows pointing in the **same direction** = positively correlated features. "
        "**Arrow length** = how much of that feature's variance is captured by these two PCs. "
        "Features near the edge of the circle are well-represented."
    )

    if k >= 3:
        view_mode = st.radio(
            "View mode",
            ["2D Correlation Circle", "3D Space Explorer"],
            horizontal=True,
            label_visibility="collapsed",
        )
    else:
        view_mode = "2D Correlation Circle"

    if view_mode == "2D Correlation Circle":
        if k >= 2:
            axis_cols = st.columns(2)
            with axis_cols[0]:
                pc_x_label = st.selectbox("Horizontal axis", pc_options, index=0, key="circ_pcx")
            with axis_cols[1]:
                remaining = [p for p in pc_options if p != pc_x_label]
                pc_y_label = st.selectbox("Vertical axis", remaining, index=0, key="circ_pcy")

            pc_x_idx = pc_options.index(pc_x_label)
            pc_y_idx = pc_options.index(pc_y_label)

            st.plotly_chart(
                make_correlation_circle(
                    pca_result.loadings,
                    pca_result.feature_names,
                    pca_result.explained_variance,
                    pc_x=pc_x_idx,
                    pc_y=pc_y_idx,
                ),
                use_container_width=True,
            )
        else:
            st.info("Select at least 2 components (k ≥ 2) to view the correlation circle.")
    else:
        show_scores = st.checkbox("Show data points (scores) in 3D space", value=True)
        st.plotly_chart(
            make_correlation_3d(
                pca_result.loadings,
                pca_result.feature_names,
                pca_result.explained_variance,
                scores=pca_result.scores if show_scores else None,
                label_values=label_values if show_scores else None,
            ),
            use_container_width=True,
        )
        st.caption(
            "Drag to rotate · Scroll to zoom · Click a feature name in the legend to hide it. "
            "Arrow direction = loading vector in PC1/PC2/PC3 space."
        )

elif active_tab == "scores":
    st.subheader("Where does each data point land in PC space?")
    st.write(
        "Each dot is one row from your dataset, projected into principal component space. "
        "Clusters or separation patterns suggest that the original features contain meaningful structure."
    )

    if label_candidates:
        label_option = st.selectbox(
            "Colour points by",
            options=["None"] + label_candidates,
            index=(label_candidates.index(label_col) + 1) if label_col in label_candidates else 0,
        )
        label_col = None if label_option == "None" else label_option
        st.session_state["pca_label_col"] = label_col
        label_values = (
            df[label_col].astype(str).tolist()
            if label_col is not None and label_col in df.columns
            else None
        )

    if k >= 2:
        st.plotly_chart(
            make_pca_scores_plot(
                pca_result.scores[:, :k],
                label_values=label_values,
                label_name=label_col,
                pc_x=0,
                pc_y=1,
                title="Scores Plot (PC1 vs PC2)",
            ),
            use_container_width=True,
        )
    else:
        st.info("Use at least 2 components to view a scores plot.")

elif active_tab == "reconstruct":
    st.subheader("How much information is lost?")

    _k_guidance(
        pca_result.explained_variance,
        pca_result.cumulative_variance,
        pca_result.eigenvalues,
        k,
        context="reconstruct",
    )

    st.plotly_chart(
        make_reconstruction_plot(
            pca_result.reconstruction_errors,
            pca_result.cumulative_variance,
            n_components=k,
            title="Reconstruction Error",
        ),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("Variance retained", f"{pca_result.cumulative_variance[k - 1] * 100:.1f}%")
    c2.metric("Information lost", f"{(1 - pca_result.cumulative_variance[k - 1]) * 100:.1f}%")

st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Grounded tutor context (tab-specific)
# -------------------------------------------------------------------
selected_pc_label = f"PC{st.session_state.get('pca_selected_pc', 0) + 1}"
current_label_col = st.session_state.get("pca_label_col")
current_view_mode = locals().get("view_mode", None)

top_loadings = []
if active_tab == "features":
    try:
        selected_pc_idx = st.session_state.get("pca_selected_pc", 0)
        loadings = pca_result.loadings[selected_pc_idx]
        pairs = list(zip(pca_result.feature_names, loadings))
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:5]
        top_loadings = [f"{name} ({value:+.3f})" for name, value in pairs_sorted]
    except Exception:
        top_loadings = []

pc1_pct = pca_result.explained_variance[0] * 100 if len(pca_result.explained_variance) > 0 else 0
pc2_pct = pca_result.explained_variance[1] * 100 if len(pca_result.explained_variance) > 1 else 0
cum_k_pct = pca_result.cumulative_variance[k - 1] * 100

if active_tab == "variance":
    visible_elements = ["scree plot", "eigenvalue table", "variance guidance"]
    hidden_elements = ["no kernel-step tabs", "no preprocessing workflow"]
    chart_summary = (
        f"The scree plot is visible. PC1 explains about {pc1_pct:.1f}% of the variance, "
        f"PC2 explains about {pc2_pct:.1f}%, and the first {k} components explain about "
        f"{cum_k_pct:.1f}% cumulatively."
    )
elif active_tab == "features":
    visible_elements = ["loadings chart", "loadings heatmap", "feature correlation explorer"]
    hidden_elements = ["no kernel-step tabs", "no preprocessing workflow"]
    chart_summary = (
        f"The user is viewing PCA feature contributions for {selected_pc_label}. "
        f"Top loadings: {top_loadings if top_loadings else 'not available'}."
    )
elif active_tab == "scores":
    visible_elements = ["scores plot"]
    hidden_elements = ["no kernel-step tabs", "no preprocessing workflow"]
    chart_summary = (
        f"The user is viewing the PCA scores plot"
        f"{f' coloured by {current_label_col}' if current_label_col else ''}."
    )
else:
    visible_elements = ["reconstruction error plot", "variance retention metrics"]
    hidden_elements = ["no kernel-step tabs", "no preprocessing workflow"]
    chart_summary = (
        f"The user is viewing reconstruction error. "
        f"With k={k}, about {cum_k_pct:.1f}% of variance is retained."
    )

set_page_context(
    page="pca",
    section=active_tab,
    visible_elements=visible_elements,
    hidden_elements=hidden_elements,
    controls={
        "dataset_name": dataset_name,
        "n_components": k,
        "selected_pc": selected_pc_label,
        "label_col": current_label_col,
        "view_mode": current_view_mode,
    },
    chart_summary=chart_summary,
    notes=[
        "If the user asks about optimal k, answer from the scree plot only when the variance tab is active.",
        "Do not describe kernel controls on the PCA page.",
    ],
)

st.session_state["pca_context"] = {
    "page": "pca",
    "section": active_tab,
    "dataset_name": dataset_name,
    "n_components": k,
    "selected_pc": selected_pc_label,
    "label_col": current_label_col,
    "view_mode": current_view_mode,
    "explained_variance_ratio": [float(v) for v in pca_result.explained_variance[:k]],
    "cumulative_explained_variance": [float(v) for v in pca_result.cumulative_variance[:k]],
    "eigenvalues": [float(v) for v in pca_result.eigenvalues[:k]],
    "feature_names": list(pca_result.feature_names),
    "top_loadings": top_loadings,
    "summary": (
        f"User is on PCA tab '{active_tab}' with k={k}, "
        f"selected component {selected_pc_label}, dataset={dataset_name}."
    ),
}

st.markdown(
    """
    <div style="
        border: 1px solid rgba(74,222,128,0.25);
        border-radius: 20px;
        padding: 18px 20px;
        margin-top: 18px;
        background: rgba(74,222,128,0.05);
    ">
        <div style="font-size: 1.05rem; font-weight: 800; color: #4ade80; margin-bottom: 8px;">
            Have nonlinear structure in your data?
        </div>
        <div style="color: #94a3b8; line-height: 1.7;">
            When standard PCA cannot separate your groups, it may be because the structure is nonlinear.
            That is where the Kernel Trick becomes useful.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)