from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


st.title("📊 Feature Space Explorer")
st.subheader("An interactive learning tool for feature engineering, kernel mapping, and PCA")

st.markdown(
    """
    Welcome to **Feature Space Explorer**.

    This project is designed to help students understand how data representation affects
    machine learning through:
    - **Preprocessing and feature engineering**
    - **Kernel-based transformations**
    - **Principal Component Analysis (PCA)**
    - **AI-assisted concept clarification**
    """
)

st.markdown("## Modules")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🛠️ Preprocessing")
    st.markdown("Clean, scale, and prepare your data")

    st.markdown("### 🌀 Kernel Trick")
    st.markdown("Understand nonlinear transformations")

with col2:
    st.markdown("### 📉 PCA Explorer")
    st.markdown("Reduce dimensionality and visualize variance")

    st.markdown("### 🤖 AI Tutor")
    st.markdown("Ask questions and clarify concepts")

st.info("Use the navigation bar at the top to explore each module.")