from __future__ import annotations

import streamlit as st


def build_tutor_messages(topic: str, question: str) -> list[dict[str, str]]:
    topic = topic.strip().lower()

    if topic == "preprocessing":
        system_prompt = (
            "You are a helpful machine learning tutor for undergraduate students. "
            "Explain preprocessing concepts such as scaling, normalization, "
            "standardization, missing values, feature distributions, and correlations. "
            "Be accurate, clear, and beginner-friendly."
        )
    elif topic == "kernel":
        system_prompt = (
            "You are a helpful machine learning tutor for undergraduate students. "
            "Explain nonlinear separability, feature mapping, polynomial features, "
            "and the intuition behind the kernel trick. "
            "Be accurate, clear, and beginner-friendly."
        )
    elif topic == "pca":
        system_prompt = (
            "You are a helpful machine learning tutor for undergraduate students. "
            "Explain dimensionality reduction, explained variance, principal components, "
            "covariance, eigenvectors, and when PCA is useful. "
            "Be accurate, clear, and beginner-friendly."
        )
    else:
        system_prompt = (
            "You are a helpful machine learning tutor for undergraduate students. "
            "Be accurate, clear, and beginner-friendly."
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def build_context_summary(topic: str) -> str:
    topic = topic.strip().lower()

    if topic == "preprocessing":
        ctx = st.session_state.get("preprocessing_context", {})
        chart_ctx = st.session_state.get("preprocessing_chart_context", {})
        if not ctx:
            return "No preprocessing context available."

        return (
            f"Page: Preprocessing\n"
            f"Dataset: {ctx.get('dataset_name', 'unknown')}\n"
            f"Scaling method: {ctx.get('scaling_method', 'unknown')}\n"
            f"Selected features: {ctx.get('selected_features', [])}\n"
            f"Feature visualized: {ctx.get('feature_to_plot', 'none')}\n"
            f"Chart context: {chart_ctx}"
        )

    if topic == "kernel":
        ctx = st.session_state.get("kernel_context", {})
        chart_ctx = st.session_state.get("kernel_chart_context", {})
        if not ctx:
            return "No kernel context available."

        return (
            f"Page: Kernel Trick\n"
            f"Dataset: {ctx.get('dataset_name', 'unknown')}\n"
            f"Samples: {ctx.get('n_samples', 'unknown')}\n"
            f"Noise: {ctx.get('noise', 'unknown')}\n"
            f"Mapping: {ctx.get('mapping', 'unknown')}\n"
            f"Chart context: {chart_ctx}"
        )

    if topic == "pca":
        ctx = st.session_state.get("pca_context", {})
        chart_ctx = st.session_state.get("pca_chart_context", {})
        if not ctx:
            return "No PCA context available."

        return (
            f"Page: PCA\n"
            f"Dataset: {ctx.get('dataset_name', 'unknown')}\n"
            f"Components: {ctx.get('n_components', 'unknown')}\n"
            f"Explained variance ratio: {ctx.get('explained_variance_ratio', [])}\n"
            f"Cumulative explained variance: {ctx.get('cumulative_explained_variance', [])}\n"
            f"Chart context: {chart_ctx}"
        )

    return "No context available."