from __future__ import annotations

import streamlit as st

from src.llm.client import is_ollama_running, list_models
from src.llm.tutor import ask_tutor


def _init_state() -> None:
    """Initialize session state for the tutor chat"""
    if "tutor_chat" not in st.session_state:
        st.session_state["tutor_chat"] = []
    if "tutor_model" not in st.session_state:
        st.session_state["tutor_model"] = "mistral"


def render_tutor_overlay() -> None:
    """Render the AI tutor in the sidebar"""
    _init_state()

    # Add custom CSS for sidebar styling
    st.markdown(
        """
        <style>
        /* Make sidebar chat more compact */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            font-size: 0.9rem;
        }
        .chat-message {
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💬 AI Tutor")
        
        current_page = st.session_state.get("current_page", "home")
        page_label = {
            "preprocessing": "🛠️ Preprocessing",
            "kernel trick": "🌀 Kernel Trick",
            "kernel": "🌀 Kernel Trick",
            "pca": "📉 PCA Explorer",
            "home": "🏠 Home",
        }.get(current_page, "🏠 Home")
        
        st.caption(f"Context: {page_label}")

        # Check Ollama
        if not is_ollama_running():
            st.warning("⚠️ Ollama not running. Start with `ollama serve`")
            return

        # Model selector
        available_models = list_models()
        if not available_models:
            st.warning("No models found. Pull with `ollama pull mistral`")
            return

        current_model = st.session_state.get("tutor_model", "mistral")
        if current_model not in available_models:
            current_model = available_models[0]

        st.session_state["tutor_model"] = st.selectbox(
            "Model",
            options=available_models,
            index=available_models.index(current_model),
            key="tutor_model_select",
            label_visibility="collapsed",
        )

        # Suggestions
        suggestions = _get_suggestions(current_page)
        if suggestions:
            st.markdown("**Quick questions:**")
            for i, suggestion in enumerate(suggestions[:2]):  # Show 2 suggestions max
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    _send_message(suggestion, current_page)
                    st.rerun()

        st.divider()

        # Chat history container with fixed height
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["tutor_chat"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input at bottom of sidebar
        question = st.chat_input("Ask about this page...", key="tutor_chat_input")
        if question:
            _send_message(question, current_page)
            st.rerun()

        # Clear chat button
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state["tutor_chat"] = []
            st.rerun()


def _send_message(question: str, topic: str) -> None:
    """Send a message to the tutor and get a response"""
    chat_history = st.session_state["tutor_chat"]
    model = st.session_state.get("tutor_model", "mistral")

    chat_history.append({"role": "user", "content": question})

    try:
        answer = ask_tutor(
            question=question,
            topic=topic,
            chat_history=chat_history[:-1],
            model=model,
        )
    except Exception as exc:
        answer = f"⚠️ Request failed: {exc}"

    chat_history.append({"role": "assistant", "content": answer})


def _get_suggestions(page: str) -> list[str]:
    """Get context-specific suggestions"""
    ctx = st.session_state

    if page == "preprocessing":
        pctx = ctx.get("preprocessing_context", {})
        scaling = pctx.get("scaling_method", "none")
        dataset = pctx.get("dataset_name", "the dataset")
        return [
            f"Why use {scaling} scaling?",
            f"What changed in {dataset} after preprocessing?",
            "When should I remove outliers?",
        ]

    if page in {"kernel trick", "kernel"}:
        kctx = ctx.get("kernel_context", {})
        kernel = kctx.get("kernel_name", "the selected kernel")
        return [
            f"Why does {kernel} work here?",
            "What is the kernel trick in simple words?",
            "How do I choose between RBF and Polynomial?",
        ]

    if page == "pca":
        pca = ctx.get("pca_context", {})
        k = pca.get("n_components", 2)
        return [
            f"Why choose k = {k}?",
            "What does cumulative variance mean?",
            "How do I interpret this PCA plot?",
        ]

    return [
        "What does this page do?",
        "How should I use this tool?",
        "What should I look at first?",
    ]