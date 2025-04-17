import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from pathlib import Path
from ..config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_banknote_data(filename: str) -> pd.DataFrame:
    """Load banknote authentication CSV into DataFrame."""
    path = RAW_DATA_DIR / filename
    df = pd.read_csv(path, header=None)
    df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
    return df


def load_wine_data() -> tuple[np.ndarray, np.ndarray]:
    """Load sklearn wine dataset into numpy arrays."""
    data = load_wine()
    X = np.array(data.data, dtype=np.float64)
    y = np.array(data.target, dtype=np.int64)
    return X, y


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save processed DataFrame to CSV."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / filename, index=False)
