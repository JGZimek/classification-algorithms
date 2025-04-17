import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_pairplot(
    df, hue: str, results_dir: Path, filename: str = "pairplot.png"
) -> None:
    sns.pairplot(df, hue=hue)
    plt.suptitle("Pairwise feature plot", y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()


def plot_tsne(
    X: np.ndarray, y: np.ndarray, results_dir: Path, title: str, filename: str
) -> np.ndarray:
    emb = TSNE(
        **__import__("..config", fromlist=["default_params"]).default_params["tsne"]
    ).fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        emb[:, 0], emb[:, 1], c=y, cmap="viridis", s=50, edgecolor="k"
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()
    return emb


def plot_line(
    x, y, xlabel: str, ylabel: str, title: str, results_dir: Path, filename: str
) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()


def plot_confusion_matrix(
    cm, results_dir: Path, filename: str = "confusion_matrix.png"
) -> None:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()
