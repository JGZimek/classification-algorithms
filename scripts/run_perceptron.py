from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import PERCEPTRON_RESULTS_DIR
from src.data.loader import load_banknote_data
from src.visualization.plots import (
    ensure_directory,
    plot_pairplot,
    plot_tsne,
    plot_line,
    plot_confusion_matrix,
)
from src.models.perceptron import Perceptron, optimize_learning_rate
from src.models.knn import confusion_matrix as knn_confusion_matrix
from src.utils.metrics import accuracy, precision, recall, f1_score


def main():
    results_dir = Path(PERCEPTRON_RESULTS_DIR)
    ensure_directory(results_dir)

    # Load and summarize data
    df = load_banknote_data("data_banknote_authentication.csv")
    plot_pairplot(df, hue="class", results_dir=results_dir)

    # Prepare features and labels
    X = df.iloc[:, :-1].values
    y = df["class"].values

    # Visualize full dataset using t-SNE
    plot_tsne(
        X,
        y,
        results_dir,
        title="t-SNE Banknote Authentication (Full Dataset)",
        filename="tsne_full.png",
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Optimize learning rate
    lrs = list(np.linspace(0.001, 0.1, 10))
    lrs, acc, prec, rec, f1 = optimize_learning_rate(
        X_train, X_test, y_train, y_test, lrs
    )

    # Plot metrics vs. learning rate
    plot_line(
        lrs,
        acc,
        xlabel="Learning Rate",
        ylabel="Accuracy",
        title="Learning Rate vs Accuracy",
        results_dir=results_dir,
        filename="lr_accuracy.png",
    )
    plot_line(
        lrs,
        prec,
        xlabel="Learning Rate",
        ylabel="Precision",
        title="Learning Rate vs Precision",
        results_dir=results_dir,
        filename="lr_precision.png",
    )
    plot_line(
        lrs,
        rec,
        xlabel="Learning Rate",
        ylabel="Recall",
        title="Learning Rate vs Recall",
        results_dir=results_dir,
        filename="lr_recall.png",
    )
    plot_line(
        lrs,
        f1,
        xlabel="Learning Rate",
        ylabel="F1 Score",
        title="Learning Rate vs F1 Score",
        results_dir=results_dir,
        filename="lr_f1.png",
    )

    # Train final model with best F1
    best_idx = int(np.argmax(f1))
    best_lr = lrs[best_idx]
    print(f"Best learning_rate: {best_lr:.3f}")

    model = Perceptron(learning_rate=best_lr).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Print final metrics
    print(f"Final Accuracy: {accuracy(y_test, y_pred):.4f}")
    print(f"Final Precision: {precision(y_test, y_pred, average='macro'):.4f}")
    print(f"Final Recall: {recall(y_test, y_pred, average='macro'):.4f}")
    print(f"Final F1-score: {f1_score(y_test, y_pred, average='macro'):.4f}")

    # Confusion matrix
    cm = knn_confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, results_dir)

    # Decision boundaries on t-SNE reduced training data
    X_train_emb = plot_tsne(
        X_train,
        y_train,
        results_dir,
        title="t-SNE Banknote Authentication (Train)",
        filename="tsne_train.png",
    )
    model_tsne = Perceptron(learning_rate=best_lr).fit(X_train_emb, y_train)

    # Create grid for decision boundary
    x_min, x_max = X_train_emb[:, 0].min() - 5, X_train_emb[:, 0].max() + 5
    y_min, y_max = X_train_emb[:, 1].min() - 5, X_train_emb[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model_tsne.predict(grid).reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.5)
    for cls in np.unique(y_train):
        mask = y_train == cls
        plt.scatter(
            X_train_emb[mask, 0],
            X_train_emb[mask, 1],
            label=f"Class {cls}",
            edgecolor="k",
        )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Decision Boundaries on t-SNE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "decision_boundary_tsne.png")
    plt.show()


if __name__ == "__main__":
    main()
