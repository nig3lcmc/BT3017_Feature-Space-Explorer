from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import make_circles, make_moons


@dataclass
class KernelDataset:
    features_df: pd.DataFrame
    target_series: pd.Series


def generate_kernel_dataset(
    dataset_name: str,
    n_samples: int = 300,
    noise: float = 0.1,
    random_state: int = 42,
) -> KernelDataset:
    """
    Generate a synthetic dataset for kernel visualization.
    Supported datasets:
    - moons
    - circles
    """
    normalized_name = dataset_name.strip().lower()

    if normalized_name == "moons":
        X, y = make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state,
        )
    elif normalized_name == "circles":
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=0.5,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported kernel dataset: {dataset_name}")

    features_df = pd.DataFrame(X, columns=["x1", "x2"])
    target_series = pd.Series(y, name="target")

    return KernelDataset(features_df=features_df, target_series=target_series)