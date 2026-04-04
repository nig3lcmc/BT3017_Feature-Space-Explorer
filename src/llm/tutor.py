from __future__ import annotations

from src.llm.client import generate_chat_response
from src.llm.context_builder import get_tutor_context_json, build_grounding_block


SYSTEM_PROMPT = """
You are an AI tutor inside Feature Space Explorer.

Your job is to help students understand the CURRENT page they are looking at.

Safety and grounding rules:
1. Only refer to UI elements, tabs, buttons, controls, charts, and parameters that are explicitly present in the provided UI context.
2. Never invent buttons, tabs, plots, controls, or features.
3. If something is not shown in the context, say: "I don't see that in your current view."
4. If the student asks "this tab", "this chart", "here", or similar, interpret that using ONLY the current UI context.
5. Prefer concrete explanation of the current chart or controls over generic textbook explanations.
6. If uncertain, say what is missing rather than guessing.
7. Keep answers beginner-friendly and concise unless the user asks for more depth.
"""


def ask_tutor(
    question: str,
    topic: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    model: str = "mistral",
) -> str:
    chat_history = chat_history or []

    # Keep only recent turns
    recent_history = [
        msg for msg in chat_history
        if msg.get("role") in {"user", "assistant"} and msg.get("content")
    ][-6:]

    ui_context_json = get_tutor_context_json()
    grounding_block = build_grounding_block()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                "This is the grounded UI context. Use it strictly.\n\n"
                f"{grounding_block}\n"
                f"Structured context JSON:\n{ui_context_json}"
            ),
        },
    ]

    messages.extend(recent_history)
    messages.append({"role": "user", "content": question})

    answer = generate_chat_response(messages=messages, model=model)

    # Lightweight post-check against common hallucination patterns
    forbidden_phrases = [
        "plot svm",
        "visualize tab",
    ]
    hidden_text = ui_context_json.lower()

    for phrase in forbidden_phrases:
        if phrase in answer.lower() and phrase not in hidden_text:
            return (
                "I should correct myself: I should only refer to elements that are actually "
                "present in your current view. Based on the current UI context, I don't see "
                f"'{phrase}' here.\n\n"
                + answer
            )

    return answer