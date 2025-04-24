import numpy as np
from .perceptron import Perceptron


class OneVsRestClassifier:
    """
    One-vs-Rest multiclass wrapper using a Perceptron for each class.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iter: int = 1000,
        tolerance: float = 1e-4,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.classes_: np.ndarray = np.array([])
        self.classifiers_: dict[int, Perceptron] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneVsRestClassifier":
        """
        Train one Perceptron per unique class in y.
        """
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_bin = (y == cls).astype(int)
            clf = Perceptron(
                learning_rate=self.learning_rate,
                n_iter=self.n_iter,
                tolerance=self.tolerance,
            ).fit(X, y_bin)
            self.classifiers_[cls] = clf
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class for each sample by selecting the classifier
        with the highest raw score (predict_proba).
        """
        # stack raw scores from each binary classifier: shape (n_classes, n_samples)
        scores = np.vstack(
            [self.classifiers_[cls].predict_proba(X) for cls in self.classes_]
        )
        # pick index of highest score per sample
        idx = np.argmax(scores, axis=0)
        return self.classes_[idx]
