import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from src.utils.utils import accuracy, precision, recall, f1_score


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_wine_data() -> (np.ndarray, np.ndarray):
    """Load the Wine dataset and return features and labels."""
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names).values
    y = data.target
    return X, y


class KNNClassifier:
    """k-Nearest Neighbors classifier."""

    def __init__(self, k: int = 3, metric: str = "euclidean", p: int = 2) -> None:
        self.k = k
        self.metric = metric
        self.p = p
        self._X_train = None
        self._y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store training data."""
        self._X_train = X.copy()
        self._y_train = y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for each sample in X."""
        X = np.asarray(X)
        predictions = []
        for x in X:
            if self.metric == "euclidean":
                dists = np.linalg.norm(self._X_train - x, axis=1)
            elif self.metric == "minkowski":
                dists = np.sum(np.abs(self._X_train - x) ** self.p, axis=1) ** (
                    1 / self.p
                )
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            neighbors = self._y_train[np.argsort(dists)[: self.k]]
            labels, counts = np.unique(neighbors, return_counts=True)
            predictions.append(labels[np.argmax(counts)])

        return np.array(predictions)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute confusion matrix as a pandas DataFrame."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, cls in enumerate(classes):
        for j, pred in enumerate(classes):
            matrix[i, j] = np.sum((y_true == cls) & (y_pred == pred))

    idx = [f"Actual: {c}" for c in classes]
    cols = [f"Predicted: {c}" for c in classes]
    return pd.DataFrame(matrix, index=idx, columns=cols)


def optimize_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k_values: list,
) -> list:
    """Return misclassification errors for a range of k values."""
    errors = []
    for k in k_values:
        model = KNNClassifier(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        errors.append(1 - accuracy(y_val, y_pred))
    return errors


def optimize_p(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    p_values: list,
    k: int = 3,
) -> list:
    """Return misclassification errors for a range of p values (Minkowski metric)."""
    errors = []
    for p in p_values:
        model = KNNClassifier(k=k, metric="minkowski", p=p)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        errors.append(1 - accuracy(y_val, y_pred))
    return errors


def plot_line(x, y, xlabel, ylabel, title, results_dir: Path, filename: str) -> None:
    """Generic line plot utility."""
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()


def tsne_visualization(
    X: np.ndarray, y: np.ndarray, results_dir: Path, title: str, filename: str
) -> None:
    """Transform data with t-SNE and plot colored by labels."""
    embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedded[:, 0], embedded[:, 1], c=y, cmap="viridis", s=50, edgecolor="k"
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()


def main():
    results_dir = Path("docs/task_knn_results")
    ensure_directory(results_dir)

    X, y = load_wine_data()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    k_values = list(range(1, 16))
    errors_k = optimize_k(X_train, y_train, X_val, y_val, k_values)
    plot_line(
        k_values,
        errors_k,
        "k",
        "Error rate",
        "Optimize k",
        results_dir,
        "optimize_k.png",
    )
    best_k = k_values[np.argmin(errors_k)]
    print(f"Best k: {best_k}")

    p_values = [1, 1.5, 2, 3, 4]
    errors_p = optimize_p(X_train, y_train, X_val, y_val, p_values, k=best_k)
    plot_line(
        p_values,
        errors_p,
        "p",
        "Error rate",
        "Optimize p",
        results_dir,
        "optimize_p.png",
    )
    best_p = p_values[np.argmin(errors_p)]
    print(f"Best p: {best_p}")

    model = KNNClassifier(k=best_k, metric="minkowski", p=best_p)
    model.fit(X_train_full, y_train_full)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Final confusion matrix:")
    print(cm)

    print("Metrics:")
    print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
    print(f"Precision: {precision(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall: {recall(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred, average='macro'):.4f}")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png")
    plt.show()

    tsne_visualization(
        X_test, y_pred, results_dir, "t-SNE k-NN Results", "tsne_knn.png"
    )


if __name__ == "__main__":
    main()
