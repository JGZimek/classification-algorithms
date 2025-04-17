import numpy as np
from typing import Tuple


def shuffle_and_split(
    X: np.ndarray,
    y: np.ndarray,
    split_ratio: float,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle X, y (opcjonalnie) i podziel na dwie części.
    Zwraca:
      X1, y1 – pierwsza część (split_ratio * n_samples)
      X2, y2 – druga część
    """
    if shuffle:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]
    n1 = int(split_ratio * len(X))
    return X[:n1], y[:n1], X[n1:], y[n1:]
