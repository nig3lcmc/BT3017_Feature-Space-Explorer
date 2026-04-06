from __future__ import annotations

from pathlib import Path
import sys


def add_project_root_to_path(current_file: str, levels_up: int = 2) -> None:
    """
    Add the project root to sys.path so Streamlit pages can import from src/.
    """
    project_root = Path(current_file).resolve().parents[levels_up]
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.append(project_root_str)
