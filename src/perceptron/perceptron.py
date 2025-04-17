import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from src.utils.utils import accuracy, precision, recall, f1_score


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_csv_data(filepath: str, column_names: list) -> pd.DataFrame:
    """Load CSV file and assign column names."""
    df = pd.read_csv(filepath, header=None)
    df.columns = column_names
    return df


def summarize_data(df: pd.DataFrame) -> None:
    """Print head, info and class distribution."""
    print("First 5 records:\n", df.head(), sep="")
    print("\nInfo:")
    df.info()
    print("\nClass distribution:\n", df["class"].value_counts(), sep="")


def plot_pairplot(df: pd.DataFrame, results_dir: Path) -> None:
    sns.pairplot(df, hue="class")
    plt.suptitle("Pairwise feature plot", y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / "pairplot.png")
    plt.show()


def tsne_plot(
    X: np.ndarray, y: np.ndarray, results_dir: Path, title: str, filename: str
) -> np.ndarray:
    """Compute t-SNE embedding and plot colored by labels."""
    embedded = TSNE(
        n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42
    ).fit_transform(X)
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        mask = y == label
        plt.scatter(
            embedded[mask, 0], embedded[mask, 1], label=f"Class {label}", alpha=0.7
        )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()
    return embedded


class Perceptron:
    """Simple Perceptron classifier."""

    def __init__(
        self, learning_rate: float = 0.01, n_iter: int = 1000, tolerance: float = 1e-4
    ) -> None:
        self.lr = learning_rate
        self.n_iter = n_iter
        self.tol = tolerance
        self.weights = None
        self.bias = 0.0

    @staticmethod
    def _activate(z: np.ndarray) -> np.ndarray:
        return np.where(z >= 0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0.0

        for i in range(self.n_iter):
            update_sum = 0.0
            for xi, target in zip(X, y):
                out = xi.dot(self.weights) + self.bias
                pred = self._activate(out)
                error = target - pred
                if error:
                    update = self.lr * error
                    self.weights += update * xi
                    self.bias += update
                    update_sum += abs(update)
            if update_sum < self.tol:
                print(f"Converged after {i+1} iterations")
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = X.dot(self.weights) + self.bias
        return self._activate(out)


def evaluate(model: Perceptron, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    y_pred = model.predict(X_test)
    return (
        accuracy(y_test, y_pred),
        precision(y_test, y_pred, average="macro"),
        recall(y_test, y_pred, average="macro"),
        f1_score(y_test, y_pred, average="macro"),
    )


def optimize_learning_rate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    lrs: list,
) -> dict:
    results = {"lr": [], "acc": [], "prec": [], "rec": [], "f1": []}
    for lr in lrs:
        model = Perceptron(learning_rate=lr)
        model.fit(X_train, y_train)
        acc, prec, rec, f1 = evaluate(model, X_test, y_test)
        results["lr"].append(lr)
        results["acc"].append(acc)
        results["prec"].append(prec)
        results["rec"].append(rec)
        results["f1"].append(f1)
        print(
            f"lr={lr:.3f} | acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}"
        )
    return results


def plot_metrics(res: dict, results_dir: Path, filename: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(res["lr"], res["acc"], marker="o", label="Accuracy")
    plt.plot(res["lr"], res["prec"], marker="s", linestyle="--", label="Precision")
    plt.plot(res["lr"], res["rec"], marker="^", linestyle="--", label="Recall")
    plt.plot(res["lr"], res["f1"], marker="d", linestyle="--", label="F1")
    plt.xlabel("Learning rate")
    plt.ylabel("Score")
    plt.title("Learning rate optimization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()


def plot_decision_boundary(
    X_emb: np.ndarray,
    y: np.ndarray,
    model: Perceptron,
    results_dir: Path,
    filename: str,
) -> None:
    x_min, x_max = X_emb[:, 0].min() - 5, X_emb[:, 0].max() + 5
    y_min, y_max = X_emb[:, 1].min() - 5, X_emb[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.5)
    for cls in np.unique(y):
        mask = y == cls
        plt.scatter(X_emb[mask, 0], X_emb[mask, 1], label=f"Class {cls}", edgecolor="k")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Decision boundaries")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()


def main():
    results_dir = Path("docs/task_perceptron_results")
    ensure_directory(results_dir)

    cols = ["variance", "skewness", "curtosis", "entropy", "class"]
    df = load_csv_data("data/data_banknote_authentication.csv", cols)
    summarize_data(df)
    plot_pairplot(df, results_dir)

    X = df.iloc[:, :-1].values
    y = df["class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lrs = list(np.linspace(0.001, 0.1, 10))
    res = optimize_learning_rate(X_train, X_test, y_train, y_test, lrs)
    plot_metrics(res, results_dir, "lr_opt.png")

    X_emb = tsne_plot(X_train, y_train, results_dir, "TSNE Train", "tsne_train.png")
    model_tsne = Perceptron(learning_rate=0.01)
    model_tsne.fit(X_emb, y_train)
    plot_decision_boundary(
        X_emb, y_train, model_tsne, results_dir, "decision_boundary_tsne.png"
    )


if __name__ == "__main__":
    main()
