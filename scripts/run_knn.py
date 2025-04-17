from pathlib import Path
import numpy as np
from src.config import KNN_RESULTS_DIR, default_params
from src.data.loader import load_wine_data
from src.utils.metrics import (
    confusion_matrix,
    plot_evaluation_metrics,
)
from src.visualization.plots import (
    ensure_directory,
    plot_line,
    plot_tsne,
    plot_confusion_matrix,
)
from src.models.knn import KNNClassifier, optimize_k, optimize_p
from sklearn.model_selection import train_test_split


def main():
    results_dir = Path(KNN_RESULTS_DIR)
    ensure_directory(results_dir)

    X, y = load_wine_data()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    k_vals = list(range(1, 16))
    errs_k = optimize_k(X_train, y_train, X_val, y_val, k_vals)
    plot_line(
        k_vals, errs_k, "k", "Error rate", "Optimize k", results_dir, "optimize_k.png"
    )
    best_k = k_vals[errs_k.index(min(errs_k))]
    print(f"The best k value: {best_k}")

    p_vals = [1, 1.5, 2, 3, 4]
    errs_p = optimize_p(X_train, y_train, X_val, y_val, p_vals, k=best_k)
    plot_line(
        p_vals, errs_p, "p", "Error rate", "Optimize p", results_dir, "optimize_p.png"
    )
    best_p = p_vals[errs_p.index(min(errs_p))]
    print(f"The best p value: {best_p}")

    model = KNNClassifier(k=best_k, metric="minkowski", p=best_p).fit(
        X_train_full, y_train_full
    )
    y_pred = model.predict(X_test)

    plot_evaluation_metrics(y_test, y_pred, results_dir, average="macro")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, results_dir)

    plot_tsne(X_test, y_pred, results_dir, "t-SNE k-NN Results", "tsne_knn.png")


if __name__ == "__main__":
    main()
