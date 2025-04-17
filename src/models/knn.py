import numpy as np
from typing import List
from ..utils.metrics import accuracy


class KNNClassifier:
    def __init__(self, k: int = 3, metric: str = "euclidean", p: int = 2) -> None:
        self.k = k
        self.metric = metric
        self.p = p
        self._X = None
        self._y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self._X = X.copy()
        self._y = y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            if self.metric == "euclidean":
                dists = np.linalg.norm(self._X - x, axis=1)
            else:
                dists = np.sum(np.abs(self._X - x) ** self.p, axis=1) ** (1 / self.p)
            neighbors = self._y[np.argsort(dists)[: self.k]]
            labels, counts = np.unique(neighbors, return_counts=True)
            preds.append(labels[np.argmax(counts)])
        return np.array(preds)


def optimize_k(X_train, y_train, X_val, y_val, k_values: List[int]) -> List[float]:
    return [
        1 - accuracy(y_val, KNNClassifier(k=k).fit(X_train, y_train).predict(X_val))
        for k in k_values
    ]


def optimize_p(
    X_train, y_train, X_val, y_val, p_values: List[float], k: int = 3
) -> List[float]:
    return [
        1
        - accuracy(
            y_val,
            KNNClassifier(k=k, metric="minkowski", p=p)
            .fit(X_train, y_train)
            .predict(X_val),
        )
        for p in p_values
    ]
