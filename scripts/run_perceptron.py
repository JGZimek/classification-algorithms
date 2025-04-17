from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import PERCEPTRON_RESULTS_DIR
from src.data.loader import load_banknote_data
from src.visualization.plots import (
    ensure_directory,
    plot_pairplot,
    plot_confusion_matrix,
    plot_metric_sweep,
)
from src.utils.metrics import confusion_matrix, plot_evaluation_metrics
from src.models.perceptron import Perceptron, optimize_learning_rate


def main():
    results_dir = Path(PERCEPTRON_RESULTS_DIR)
    ensure_directory(results_dir)

    X, y, df = load_banknote_data("data_banknote_authentication.csv")
    plot_pairplot(df, hue="class", results_dir=results_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lrs = list(np.linspace(0.001, 1, 10))
    lrs, acc, prec, rec, f1 = optimize_learning_rate(
        X_train, X_test, y_train, y_test, lrs
    )

    metrics = {
        "Accuracy": acc,
        "Precision (macro)": prec,
        "Recall (macro)": rec,
        "F1 Score (macro)": f1,
    }
    plot_metric_sweep(
        lrs,
        metrics,
        results_dir,
        xlabel="Learning Rate",
        ylabel="Score",
        title="Learning Rate Optimization",
        filename="lr_metrics.png",
    )

    best_idx = int(np.argmax(f1))
    best_lr = lrs[best_idx]
    print(f"Best learning_rate: {best_lr:.3f}")

    model = Perceptron(learning_rate=best_lr).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plot_evaluation_metrics(y_test, y_pred, results_dir, average="macro")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, results_dir)


if __name__ == "__main__":
    main()
