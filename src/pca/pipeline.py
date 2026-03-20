from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    transformed_df: pd.DataFrame
    explained_variance_ratio: list[float]
    cumulative_explained_variance: list[float]
    components_df: pd.DataFrame


def run_pca(df: pd.DataFrame, n_components: int, exclude_cols: list[str] | None = None) -> PCAResult:
    """
    Run PCA on numeric columns of a dataframe after standardization.
    Excluded columns are ignored.
    """
    exclude_cols = exclude_cols or []
    numeric_df = df.select_dtypes(include=["number"]).drop(columns=exclude_cols, errors="ignore")

    if numeric_df.empty:
        raise ValueError("No numeric columns available for PCA.")

    if n_components < 1 or n_components > numeric_df.shape[1]:
        raise ValueError(
            f"n_components must be between 1 and {numeric_df.shape[1]}, got {n_components}."
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(scaled)

    transformed_columns = [f"PC{i + 1}" for i in range(n_components)]
    transformed_df = pd.DataFrame(transformed, columns=transformed_columns, index=df.index)

    explained_variance_ratio = pca.explained_variance_ratio_.tolist()
    cumulative_explained_variance = pd.Series(explained_variance_ratio).cumsum().tolist()

    components_df = pd.DataFrame(
        pca.components_,
        columns=numeric_df.columns,
        index=transformed_columns,
    )

    return PCAResult(
        transformed_df=transformed_df,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_explained_variance=cumulative_explained_variance,
        components_df=components_df,
    )