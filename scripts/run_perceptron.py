from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import PERCEPTRON_RESULTS_DIR
from src.data.loader import load_banknote_data
from src.visualization.plots import (
    ensure_directory,
    plot_pairplot,
    plot_confusion_matrix,
    print_metric_sweep,
)
from src.utils.metrics import confusion_matrix
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

    # 1) Macro
    _, acc_mac, prec_mac, rec_mac, f1_mac = optimize_learning_rate(
        X_train, X_test, y_train, y_test, lrs, average="macro"
    )
    # 2) Micro
    _, acc_mic, prec_mic, rec_mic, f1_mic = optimize_learning_rate(
        X_train, X_test, y_train, y_test, lrs, average="micro"
    )

    metrics = {
        "Accuracy (macro)": acc_mac,
        "Precision (macro)": prec_mac,
        "Recall (macro)": rec_mac,
        "F1 Score (macro)": f1_mac,
        "Accuracy (micro)": acc_mic,
        "Precision (micro)": prec_mic,
        "Recall (micro)": rec_mic,
        "F1 Score (micro)": f1_mic,
    }
    print_metric_sweep(lrs, metrics)

    best_idx_mac = int(np.argmax(f1_mac))
    best_lr_mac = lrs[best_idx_mac]
    best_idx_mic = int(np.argmax(f1_mic))
    best_lr_mic = lrs[best_idx_mic]

    print(f"\nBest learning_rate (macro F1): {best_lr_mac:.3f}")
    print(f"Best learning_rate (micro F1): {best_lr_mic:.3f}")
    best_lr = best_lr_mac

    model = Perceptron(learning_rate=best_lr).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, results_dir)


if __name__ == "__main__":
    main()
