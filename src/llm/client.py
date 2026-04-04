from __future__ import annotations

import os
import requests


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def is_ollama_running() -> bool:
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def list_models() -> list[str]:
    if LLM_PROVIDER == "ollama":
        try:
            response = requests.get(OLLAMA_TAGS_URL, timeout=5)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", [])
            names = [m.get("name", "") for m in models if m.get("name")]
            return names or ["mistral"]
        except requests.RequestException:
            return ["mistral"]

    # Keep this simple for cloud mode
    if LLM_PROVIDER == "gemini":
        return ["gemini"]

    return ["default"]


def generate_chat_response(
    messages: list[dict[str, str]],
    model: str = "mistral",
    timeout: int = 180,
) -> str:
    if LLM_PROVIDER == "ollama":
        return _chat_with_ollama(messages, model=model, timeout=timeout)

    if LLM_PROVIDER == "gemini":
        return _chat_with_gemini(messages, model=model, timeout=timeout)

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _chat_with_ollama(
    messages: list[dict[str, str]],
    model: str = "mistral",
    timeout: int = 180,
) -> str:
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["message"]["content"].strip()


def _chat_with_gemini(
    messages: list[dict[str, str]],
    model: str = "gemini",
    timeout: int = 180,
) -> str:
    """
    Keep this adapter isolated so you can swap transport without touching tutor.py.
    Replace this stub with your preferred Gemini SDK or REST implementation.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    # Convert messages to plain prompt text for a simple first version.
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        prompt_parts.append(f"{role}:\n{content}")
    prompt = "\n\n".join(prompt_parts)

    # Stub on purpose: implement with your Gemini SDK or REST call here.
    # The point is that ONLY this function changes when you switch providers.
    raise NotImplementedError(
        "Gemini transport not implemented yet. "
        "Add your Gemini SDK or REST call inside _chat_with_gemini()."
    )