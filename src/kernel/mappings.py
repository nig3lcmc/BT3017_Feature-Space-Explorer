from __future__ import annotations

import pandas as pd


def apply_polynomial_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a simple polynomial feature mapping to 2D input data.

    Input columns expected:
    - x1
    - x2

    Output columns:
    - x1
    - x2
    - x1_squared
    - x2_squared
    - x1_x2
    """
    required_cols = {"x1", "x2"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Input dataframe must contain 'x1' and 'x2' columns.")

    mapped_df = pd.DataFrame(index=df.index)
    mapped_df["x1"] = df["x1"]
    mapped_df["x2"] = df["x2"]
    mapped_df["x1_squared"] = df["x1"] ** 2
    mapped_df["x2_squared"] = df["x2"] ** 2
    mapped_df["x1_x2"] = df["x1"] * df["x2"]

    return mapped_df