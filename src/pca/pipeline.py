from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    eigenvalues: list[float]
    explained_variance: list[float]
    cumulative_variance: list[float]
    loadings: np.ndarray
    scores: np.ndarray
    reconstruction_errors: list[float]
    means: np.ndarray
    stds: np.ndarray
    cov_matrix: np.ndarray
    n: int
    d: int
    feature_names: list[str]
    scaled_df: pd.DataFrame


def run_pca(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> PCAResult:
    exclude_cols = exclude_cols or []

    numeric_df = df.select_dtypes(include=["number"]).drop(columns=exclude_cols, errors="ignore")

    if numeric_df.empty:
        raise ValueError("No numeric columns available for PCA.")

    X = numeric_df.to_numpy(dtype=float)
    n, d = X.shape

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=numeric_df.columns, index=df.index)

    cov_matrix = np.cov(X_scaled, rowvar=False)

    pca = PCA(n_components=min(n, d))
    scores = pca.fit_transform(X_scaled)

    eigenvalues = pca.explained_variance_.tolist()
    explained_variance = pca.explained_variance_ratio_.tolist()
    cumulative_variance = np.cumsum(explained_variance).tolist()
    loadings = pca.components_

    reconstruction_errors = []
    max_k = min(n, d)
    for k in range(1, max_k + 1):
        pca_k = PCA(n_components=k)
        scores_k = pca_k.fit_transform(X_scaled)
        X_reconstructed = pca_k.inverse_transform(scores_k)
        err = np.mean((X_scaled - X_reconstructed) ** 2)
        reconstruction_errors.append(float(err))

    return PCAResult(
        eigenvalues=eigenvalues,
        explained_variance=explained_variance,
        cumulative_variance=cumulative_variance,
        loadings=loadings,
        scores=scores,
        reconstruction_errors=reconstruction_errors,
        means=scaler.mean_,
        stds=scaler.scale_,
        cov_matrix=cov_matrix,
        n=n,
        d=d,
        feature_names=numeric_df.columns.tolist(),
        scaled_df=scaled_df,
    )