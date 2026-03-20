from __future__ import annotations

import requests

from src.llm.context_builder import build_context_summary

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"


def ask_tutor(
    question: str,
    topic: str,
    chat_history: list[dict[str, str]] | None = None,
    model: str = "mistral",
) -> str:
    context_summary = build_context_summary(topic)
    chat_history = chat_history or []

    system_prompt = (
        "You are a helpful machine learning tutor for an undergraduate learning app. "
        "Use the provided app context when it is relevant. "
        "Do not invent results that are not present in the context. "
        "Explain simply, clearly, and step by step. "
        "Keep the answer under 150 words unless the student asks for more detail."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": f"App context:\n{context_summary}",
        },
    ]

    for msg in chat_history:
        if msg.get("role") in {"user", "assistant"} and msg.get("content"):
            messages.append(
                {"role": msg["role"], "content": msg["content"]}
            )

    messages.append({"role": "user", "content": question})

    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()

    payload = response.json()
    return payload["message"]["content"].strip()