import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from ..config import default_params


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
    """Transform data with t-SNE and plot."""
    tsne_params = default_params.get("tsne", {})
    emb = TSNE(**tsne_params).fit_transform(X)

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


def plot_metric_sweep(
    x,
    metrics: dict,
    results_dir: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """
    Plot multiple metric curves against a shared x-axis in a single figure.

    Args:
        x: list or array of x-axis values (e.g., hyperparameter values).
        metrics: dict mapping metric names to lists of metric values.
        results_dir: Path to save the plot.
        xlabel: label for the x axis.
        ylabel: label for the y axis.
        title: plot title.
        filename: filename for saving the plot.
    """
    plt.figure(figsize=(8, 6))
    for name, y_vals in metrics.items():
        plt.plot(x, y_vals, marker="o", label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()


def plot_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    results_dir: Path,
    average: str = "macro",
    filename: str = "evaluation_metrics.png",
) -> None:
    """
    Compute accuracy, precision, recall, F1 and plot as horizontal bar chart with labels.
    """
    from ..utils.metrics import accuracy, precision, recall, f1_score

    metrics = {
        "Accuracy": accuracy(y_true, y_pred),
        f"Precision ({average})": precision(y_true, y_pred, average=average),
        f"Recall ({average})": recall(y_true, y_pred, average=average),
        f"F1 Score ({average})": f1_score(y_true, y_pred, average=average),
    }

    ensure_directory(results_dir)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(list(metrics.keys()), list(metrics.values()))
    ax.set_xlabel("Score")
    ax.set_title("Model Evaluation Metrics")
    ax.bar_label(bars, fmt="%.4f", padding=3)

    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()
