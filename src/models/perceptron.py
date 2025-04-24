import numpy as np
from typing import List, Tuple
from ..utils.metrics import accuracy, precision, recall, f1_score


class Perceptron:
    def __init__(
        self, learning_rate: float = 0.01, n_iter: int = 1000, tolerance: float = 1e-4
    ) -> None:
        self.lr = learning_rate
        self.n_iter = n_iter
        self.tol = tolerance
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0.0
        for i in range(self.n_iter):
            update_sum = 0.0
            for xi, t in zip(X, y):
                out = xi.dot(self.weights) + self.bias
                pred = 1 if out >= 0 else 0
                error = t - pred
                if error:
                    u = self.lr * error
                    self.weights += u * xi
                    self.bias += u
                    update_sum += abs(u)
            if update_sum < self.tol:
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = X.dot(self.weights) + self.bias
        return np.where(out >= 0, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.weights) + self.bias


def optimize_learning_rate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    lrs: List[float],
    average: str = "macro",
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Sweep over lrs, zwraca: (lrs, acc, prec, rec, f1) dla zadanego average.
    """
    acc, prec, rec, f1 = [], [], [], []
    for lr in lrs:
        m = Perceptron(learning_rate=lr).fit(X_train, y_train)
        y_pred = m.predict(X_test)
        acc.append(accuracy(y_test, y_pred))
        prec.append(precision(y_test, y_pred, average))
        rec.append(recall(y_test, y_pred, average))
        f1.append(f1_score(y_test, y_pred, average))
    return lrs, acc, prec, rec, f1
