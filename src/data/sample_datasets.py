from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris, load_wine


def load_sample_dataset(name: str) -> pd.DataFrame:
    """
    Load a sample dataset by name and return it as a pandas DataFrame.
    Includes a target column for convenience.
    """
    normalized_name = name.strip().lower()

    if normalized_name == "iris":
        dataset = load_iris(as_frame=True)
        df = dataset.frame.copy()
        df["target_name"] = df["target"].map(
            {idx: label for idx, label in enumerate(dataset.target_names)}
        )
        return df

    if normalized_name == "wine":
        dataset = load_wine(as_frame=True)
        df = dataset.frame.copy()
        return df

    raise ValueError(f"Unsupported dataset: {name}")


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """Load a dataset uploaded by the user from a CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Unable to parse uploaded CSV file: {e}")

    if df.empty:
        raise ValueError("Uploaded dataset is empty")

    return df


def get_available_datasets() -> list[str]:
    """
    Return the list of supported sample datasets.
    """
    return ["iris", "wine"]