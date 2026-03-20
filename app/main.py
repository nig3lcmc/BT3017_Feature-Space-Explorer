from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import CONFIG


# Page config MUST be here (entrypoint)
st.set_page_config(
    page_title=CONFIG.app_title,
    page_icon=CONFIG.app_icon,
    layout=CONFIG.layout,
)

# Define pages
home_page = st.Page(
    "pages/0_home.py",
    title="Home",
    icon="🏠",
    default=True,
)

preprocessing_page = st.Page(
    "pages/1_preprocessing.py",
    title="Preprocessing",
    icon="🛠️",
)

kernel_page = st.Page(
    "pages/2_kernel_trick.py",
    title="Kernel Trick",
    icon="🌀",
)

pca_page = st.Page(
    "pages/3_pca_explorer.py",
    title="PCA Explorer",
    icon="📉",
)

ai_tutor_page = st.Page(
    "pages/4_ai_tutor.py",
    title="AI Tutor",
    icon="🤖",
)

# Top navigation
pg = st.navigation(
    [
        home_page,
        preprocessing_page,
        kernel_page,
        pca_page,
        ai_tutor_page,
    ],
    position="top",
)

# Run selected page
pg.run()