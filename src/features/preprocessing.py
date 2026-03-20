from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_numeric_columns(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> list[str]:
    """
    Return numeric columns, optionally excluding specific columns.
    """
    exclude_cols = exclude_cols or []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude_cols]


def scale_numeric_features(
    df: pd.DataFrame,
    method: str,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Scale numeric features in a dataframe while preserving excluded columns.
    Supported methods:
    - none
    - standardization
    - normalization
    """
    exclude_cols = exclude_cols or []
    scaled_df = df.copy()

    numeric_cols = get_numeric_columns(scaled_df, exclude_cols=exclude_cols)
    if not numeric_cols or method == "none":
        return scaled_df

    if method == "standardization":
        scaler = StandardScaler()
    elif method == "normalization":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")

    scaled_df[numeric_cols] = scaler.fit_transform(scaled_df[numeric_cols])
    return scaled_df


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a compact dataframe summary.
    """
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_values": df.isna().sum().values,
            "n_unique": df.nunique(dropna=False).values,
        }
    )
    return summary