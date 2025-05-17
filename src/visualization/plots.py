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


def print_metric_sweep(
    x: list[float],
    metrics: dict[str, list[float]],
) -> None:
    """
    Print a table of metric values for each x (e.g. learning rate).

    Args:
        x: list of hyperparameter values.
        metrics: dict mapping metric names to lists of values.
    """
    # header
    headers = ["lr"] + list(metrics.keys())
    header_line = "".join(f"{h:<20}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    # rows
    for i, xv in enumerate(x):
        row = f"{xv:<20.3f}"
        for name in metrics:
            row += f"{metrics[name][i]:<20.3f}"
        print(row)


def plot_scaling_normalization_demo(
    df_orig, df_std, df_norm, demo_feats: list[str], results_dir: Path,
    filename: str = "scaling_normalization_demo.png"
) -> None:
    """
    Plot histograms of two features before and after standardization and normalization.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    titles = [
        "Original Distributions",
        "Standardized Distributions",
        "Normalized Distributions",
    ]
    for ax, dataset, title in zip(axes, [df_orig, df_std, df_norm], titles):
        ax.hist(dataset[demo_feats[0]], bins=30, alpha=0.6, label=demo_feats[0])
        ax.hist(dataset[demo_feats[1]], bins=30, alpha=0.6, label=demo_feats[1])
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()
    plt.close()