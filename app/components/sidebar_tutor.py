from __future__ import annotations

import uuid

import streamlit as st

from src.llm.client import is_ollama_running, list_models
from src.llm.tutor import ask_tutor

DEFAULT_TUTOR_MODEL = "mistral"
NEW_CHAT_TITLE = "New chat"
DEFAULT_SUGGESTIONS = [
    "What does this page do?",
    "How should I use this tool?",
    "What should I look at first?",
]
PAGE_LABELS = {
    "home": "Home",
    "preprocessing": "Preprocessing",
    "kernel": "Kernel Trick",
    "kernel trick": "Kernel Trick",
    "pca": "PCA Explorer",
}
PAGE_SUGGESTIONS = {
    "preprocessing": [
        "Why do we scale features here?",
        "What changed after preprocessing?",
        "When should I remove outliers?",
    ],
    "kernel": [
        "Why does this kernel work here?",
        "What is the kernel trick in simple words?",
        "How do I choose between RBF and Polynomial?",
    ],
    "pca": [
        "What is the optimal k in this scree plot?",
        "What does cumulative variance mean here?",
        "How do I interpret this PCA plot?",
    ],
}


def render_sidebar_tutor() -> None:
    _init_state()

    current_page = st.session_state.get("current_page", "home")
    page_label = PAGE_LABELS.get(current_page, PAGE_LABELS["home"])

    with st.sidebar:
        _inject_sidebar_styles()

        st.markdown("## AI Tutor")
        st.caption("Ask questions about what you're learning on this page.")

        st.markdown(
            f"""
            <div class="tutor-context-card">
                <div class="tutor-context-label">Currently helping with</div>
                <div class="tutor-context-value">{page_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not is_ollama_running():
            st.warning("Ollama is not running.")
            st.code("ollama serve")
            return

        with st.expander("Tutor settings", expanded=False):
            available_models = list_models()
            current_model = st.session_state.get("tutor_model", DEFAULT_TUTOR_MODEL)

            if not available_models:
                st.info("No Ollama models found. Pull one first, e.g. `ollama pull mistral`.")
                return

            if current_model not in available_models:
                current_model = available_models[0]

            st.session_state["tutor_model"] = st.selectbox(
                "Model",
                options=available_models,
                index=available_models.index(current_model),
                key="sidebar_tutor_model_select",
            )

        suggestions = _get_suggestions(current_page)
        if suggestions:
            st.markdown("**Try asking:**")
            for index, suggestion in enumerate(suggestions[:3]):
                if st.button(
                    suggestion,
                    key=f"sidebar_tutor_suggestion_{current_page}_{index}",
                    use_container_width=True,
                ):
                    _queue_user_message(suggestion, current_page)
                    st.rerun()

        st.markdown("<div class='tutor-section-label'>Chats</div>", unsafe_allow_html=True)

        if st.button("+ New chat", key="sidebar_new_chat", use_container_width=True):
            _create_new_thread(current_page)
            st.rerun()

        active_thread = _get_active_thread()

        with st.expander(
            "Rename current chat",
            expanded=st.session_state.get("open_rename_for_active", False),
        ):
            rename_value = st.text_input(
                "Rename chat",
                value=active_thread["title"],
                key=f"rename_chat_input_{active_thread['id']}",
                label_visibility="collapsed",
                placeholder="Enter a chat title",
            )
            save_col, reset_col = st.columns(2)
            with save_col:
                if st.button(
                    "Save name",
                    key=f"rename_chat_save_{active_thread['id']}",
                    use_container_width=True,
                ):
                    active_thread["title"] = _normalize_title(rename_value)
                    st.session_state["open_rename_for_active"] = False
                    st.rerun()
            with reset_col:
                if st.button(
                    "Reset",
                    key=f"rename_chat_reset_{active_thread['id']}",
                    use_container_width=True,
                ):
                    active_thread["title"] = NEW_CHAT_TITLE
                    st.session_state["open_rename_for_active"] = False
                    st.rerun()

        for thread in st.session_state["tutor_threads"]:
            is_active = thread["id"] == st.session_state["active_tutor_thread_id"]
            title_col, delete_col = st.columns([8, 2], gap="small")
            display_title = _truncate_title(thread["title"], 22)

            with title_col:
                if st.button(
                    display_title,
                    key=f"thread_select_{thread['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    help=thread["title"],
                ):
                    st.session_state["active_tutor_thread_id"] = thread["id"]
                    st.session_state["open_rename_for_active"] = False
                    st.rerun()

            with delete_col:
                if st.button(
                    "Del",
                    key=f"thread_delete_{thread['id']}",
                    use_container_width=True,
                    help="Delete chat",
                ):
                    _delete_thread(thread["id"])
                    st.rerun()

        st.markdown("<div class='tutor-section-label'>Conversation</div>", unsafe_allow_html=True)
        active_thread = _get_active_thread()
        _render_messages(active_thread["messages"])

        pending = st.session_state.get("tutor_pending")
        if pending and pending["thread_id"] == active_thread["id"]:
            with st.spinner("Tutor is thinking..."):
                _resolve_pending_message()
            st.rerun()

        if st.button("Clear current chat", key="sidebar_tutor_clear", use_container_width=True):
            active_thread["messages"] = []
            active_thread["title"] = NEW_CHAT_TITLE
            st.rerun()

        question = st.chat_input("Ask about this page...", key="sidebar_tutor_chat_input")
        if question and question.strip():
            _queue_user_message(question.strip(), current_page)
            st.rerun()


def _render_messages(messages: list[dict[str, str]]) -> None:
    if not messages:
        st.markdown(
            """
            <div class="tutor-chat-wrap">
                <div class="tutor-empty-state">
                    No messages yet.<br><br>
                    Ask the tutor to:
                    <ul>
                        <li>explain a chart</li>
                        <li>clarify a concept</li>
                        <li>compare methods</li>
                        <li>suggest what to look at next</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown("<div class='tutor-chat-wrap'>", unsafe_allow_html=True)
    for message in messages:
        role_class = "user" if message["role"] == "user" else "assistant"
        role_label = "You" if message["role"] == "user" else "Tutor"
        st.markdown(
            f"""
            <div class="tutor-msg tutor-msg-{role_class}">
                <div class="tutor-msg-role">{role_label}</div>
                <div class="tutor-msg-content">{message["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _init_state() -> None:
    if "tutor_model" not in st.session_state:
        st.session_state["tutor_model"] = DEFAULT_TUTOR_MODEL

    if "tutor_threads" not in st.session_state:
        first_thread = _new_thread(st.session_state.get("current_page", "home"))
        st.session_state["tutor_threads"] = [first_thread]

    if "active_tutor_thread_id" not in st.session_state:
        st.session_state["active_tutor_thread_id"] = st.session_state["tutor_threads"][0]["id"]

    if "open_rename_for_active" not in st.session_state:
        st.session_state["open_rename_for_active"] = False

    if "tutor_pending" not in st.session_state:
        st.session_state["tutor_pending"] = None


def _new_thread(page: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": NEW_CHAT_TITLE,
        "page": page,
        "messages": [],
    }


def _create_new_thread(page: str) -> None:
    new_thread = _new_thread(page)
    st.session_state["tutor_threads"].insert(0, new_thread)
    st.session_state["active_tutor_thread_id"] = new_thread["id"]
    st.session_state["open_rename_for_active"] = False


def _delete_thread(thread_id: str) -> None:
    threads = [thread for thread in st.session_state["tutor_threads"] if thread["id"] != thread_id]
    active_id = st.session_state["active_tutor_thread_id"]

    if not threads:
        new_thread = _new_thread(st.session_state.get("current_page", "home"))
        threads = [new_thread]
        st.session_state["active_tutor_thread_id"] = new_thread["id"]
    elif active_id == thread_id:
        st.session_state["active_tutor_thread_id"] = threads[0]["id"]

    st.session_state["tutor_threads"] = threads
    st.session_state["open_rename_for_active"] = False


def _find_thread(thread_id: str) -> dict | None:
    for thread in st.session_state["tutor_threads"]:
        if thread["id"] == thread_id:
            return thread
    return None


def _get_active_thread() -> dict:
    active_id = st.session_state["active_tutor_thread_id"]
    thread = _find_thread(active_id)
    if thread is not None:
        return thread
    return st.session_state["tutor_threads"][0]


def _queue_user_message(question: str, topic: str) -> None:
    active_thread = _get_active_thread()
    active_thread["messages"].append({"role": "user", "content": question})

    if active_thread["title"] == NEW_CHAT_TITLE:
        active_thread["title"] = _make_chat_title(question)

    st.session_state["tutor_pending"] = {
        "thread_id": active_thread["id"],
        "question": question,
        "topic": topic,
    }


def _resolve_pending_message() -> None:
    pending = st.session_state.get("tutor_pending")
    if not pending:
        return

    thread = _find_thread(pending["thread_id"])
    if thread is None:
        st.session_state["tutor_pending"] = None
        return

    chat_history = thread["messages"]
    model = st.session_state.get("tutor_model", DEFAULT_TUTOR_MODEL)

    try:
        answer = ask_tutor(
            question=pending["question"],
            topic=pending["topic"],
            chat_history=chat_history[:-1],
            model=model,
        )
    except Exception as exc:
        answer = f"Request failed: {exc}"

    thread["messages"].append({"role": "assistant", "content": answer})
    st.session_state["tutor_pending"] = None


def _normalize_title(title: str | None) -> str:
    cleaned = (title or "").strip()
    return cleaned or NEW_CHAT_TITLE


def _make_chat_title(question: str) -> str:
    title = question.strip()
    if len(title) > 36:
        title = title[:36].rstrip() + "..."
    return title or NEW_CHAT_TITLE


def _truncate_title(title: str, max_len: int) -> str:
    title = title.strip() or NEW_CHAT_TITLE
    if len(title) <= max_len:
        return title
    return title[: max_len - 3].rstrip() + "..."


def _get_suggestions(page: str) -> list[str]:
    normalized_page = "kernel" if page == "kernel trick" else page
    return PAGE_SUGGESTIONS.get(normalized_page, DEFAULT_SUGGESTIONS)


def _inject_sidebar_styles() -> None:
    st.markdown(
        """
        <style>
        .tutor-context-card {
            background: rgba(99,102,241,0.10);
            border: 1px solid rgba(99,102,241,0.20);
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 14px;
        }

        .tutor-context-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin-bottom: 4px;
            font-weight: 700;
        }

        .tutor-context-value {
            font-size: 0.98rem;
            font-weight: 700;
            color: #e2e8f0;
        }

        .tutor-section-label {
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-top: 14px;
            margin-bottom: 8px;
        }

        .tutor-chat-wrap {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 10px;
            background: rgba(255,255,255,0.02);
            max-height: 420px;
            overflow-y: auto;
            margin-bottom: 10px;
        }

        .tutor-empty-state {
            color: #94a3b8;
            font-size: 0.92rem;
            line-height: 1.55;
            padding: 4px 4px 0 4px;
        }

        .tutor-empty-state ul {
            padding-left: 18px;
            margin-top: 8px;
            margin-bottom: 0;
        }

        .tutor-msg {
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 10px;
        }

        .tutor-msg-user {
            background: rgba(99,102,241,0.12);
            border: 1px solid rgba(99,102,241,0.18);
        }

        .tutor-msg-assistant {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .tutor-msg-role {
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin-bottom: 4px;
        }

        .tutor-msg-content {
            font-size: 0.95rem;
            line-height: 1.6;
            color: #e2e8f0;
            white-space: pre-wrap;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            overflow: hidden;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] button p {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
