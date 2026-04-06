from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.llm.client import is_ollama_running
from src.llm.context_builder import build_context_summary
from src.llm.tutor import ask_tutor


st.title("🤖 AI Tutor")
st.write("Ask questions about preprocessing, the kernel trick, and PCA.")

topic = st.session_state.get("current_page", "preprocessing")
st.info(f"Current topic: {topic}")

chat_key = f"chat_history_{topic}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = []

chat_history = st.session_state[chat_key]

model_name = st.selectbox(
    "Choose local model",
    options=["mistral", "gemma3"],
    index=0,
    key="selected_model",
)

st.markdown("### Current App Context")
st.code(build_context_summary(topic))

if not is_ollama_running():
    st.warning("Ollama is not running.")
    st.code("ollama serve", language="bash")
    st.stop()

if st.button("Clear Chat"):
    st.session_state[chat_key] = []
    st.rerun()

st.markdown("### Conversation")

for msg in chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

pending_question = st.session_state.pop("pending_tutor_question", None)

if pending_question:
    chat_history.append({"role": "user", "content": pending_question})

    with st.chat_message("user"):
        st.write(pending_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask_tutor(
                    question=pending_question,
                    topic=topic,
                    chat_history=chat_history[:-1],
                    model=model_name,
                )
                st.write(answer)
            except Exception as exc:
                answer = f"Request failed: {exc}"
                st.error(answer)

    chat_history.append({"role": "assistant", "content": answer})

question = st.chat_input("Ask a follow-up question...")

if question:
    chat_history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask_tutor(
                    question=question,
                    topic=topic,
                    chat_history=chat_history[:-1],
                    model=model_name,
                )
                st.write(answer)
            except Exception as exc:
                answer = f"Request failed: {exc}"
                st.error(answer)

    chat_history.append({"role": "assistant", "content": answer})
