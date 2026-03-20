from __future__ import annotations

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


def is_ollama_running() -> bool:
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False