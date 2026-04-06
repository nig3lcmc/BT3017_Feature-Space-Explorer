from __future__ import annotations

import json
from typing import Any

import streamlit as st


def set_page_context(
    *,
    page: str,
    section: str | None = None,
    visible_elements: list[str] | None = None,
    hidden_elements: list[str] | None = None,
    controls: dict[str, Any] | None = None,
    chart_summary: str | None = None,
    notes: list[str] | None = None,
) -> None:
    """
    Store a grounded UI snapshot for the tutor.
    Pages should call this whenever the visible state changes.
    """
    current = st.session_state.get("tutor_context", {})

    current["page"] = page
    if section is not None:
        current["section"] = section
    if visible_elements is not None:
        current["visible_elements"] = visible_elements
    if hidden_elements is not None:
        current["hidden_elements"] = hidden_elements
    if controls is not None:
        current["controls"] = controls
    if chart_summary is not None:
        current["chart_summary"] = chart_summary
    if notes is not None:
        current["notes"] = notes

    st.session_state["tutor_context"] = _json_safe(current)


def get_current_page() -> str:
    return st.session_state.get("current_page", "home")


def get_tutor_context() -> dict[str, Any]:
    """
    Returns the current grounded tutor context.
    """
    ctx = st.session_state.get("tutor_context", {})
    if not ctx:
        return {"page": get_current_page()}
    return _json_safe(ctx)


def get_tutor_context_json() -> str:
    return json.dumps(get_tutor_context(), indent=2, ensure_ascii=False)


def build_grounding_block() -> str:
    ctx = get_tutor_context()
    page = ctx.get("page", "unknown")
    section = ctx.get("section", "unknown")
    visible = ctx.get("visible_elements", [])
    hidden = ctx.get("hidden_elements", [])
    controls = ctx.get("controls", {})
    chart_summary = ctx.get("chart_summary", "No chart summary available.")
    notes = ctx.get("notes", [])

    return (
        f"Current page: {page}\n"
        f"Current section: {section}\n"
        f"Visible elements: {visible}\n"
        f"Hidden or absent elements: {hidden}\n"
        f"Current controls: {controls}\n"
        f"Chart summary: {chart_summary}\n"
        f"Notes: {notes}\n"
    )


def build_context_summary(_: str | None = None) -> str:
    """
    Backwards-compatible alias used by the legacy tutor page.
    """
    return build_grounding_block()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
