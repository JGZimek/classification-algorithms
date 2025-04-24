from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.loader import load_wine_data
from src.models.one_vs_rest import OneVsRestClassifier
from src.utils.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
)
from src.visualization.plots import ensure_directory, plot_confusion_matrix


def main():
    # create results directory
    results_dir = Path("docs/task_ovr_wine_results")
    ensure_directory(results_dir)

    # load and split wine dataset
    X, y = load_wine_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # train One-vs-Rest perceptron
    ovr = OneVsRestClassifier(learning_rate=0.01, n_iter=1000, tolerance=1e-4)
    ovr.fit(X_train, y_train)
    y_pred = ovr.predict(X_test)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, results_dir)

    # compute macro‐averaged metrics
    acc_mac = accuracy(y_test, y_pred)
    prec_mac = precision(y_test, y_pred, average="macro")
    rec_mac = recall(y_test, y_pred, average="macro")
    f1_mac = f1_score(y_test, y_pred, average="macro")

    # compute micro‐averaged metrics
    acc_mic = accuracy(y_test, y_pred)
    prec_mic = precision(y_test, y_pred, average="micro")
    rec_mic = recall(y_test, y_pred, average="micro")
    f1_mic = f1_score(y_test, y_pred, average="micro")

    # print results
    print("\nMacro-averaged metrics:")
    print(f"  Accuracy : {acc_mac:.3f}")
    print(f"  Precision: {prec_mac:.3f}")
    print(f"  Recall   : {rec_mac:.3f}")
    print(f"  F1 Score : {f1_mac:.3f}")

    print("\nMicro-averaged metrics:")
    print(f"  Accuracy : {acc_mic:.3f}")
    print(f"  Precision: {prec_mic:.3f}")
    print(f"  Recall   : {rec_mic:.3f}")
    print(f"  F1 Score : {f1_mic:.3f}")


if __name__ == "__main__":
    main()
