import pandas as pd
import streamlit as st


def render_dataset_overview(df: pd.DataFrame, title: str = "Dataset Overview") -> None:
    st.markdown(f"### {title}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isna().sum().sum()))

    st.dataframe(df.head(10), use_container_width=True)