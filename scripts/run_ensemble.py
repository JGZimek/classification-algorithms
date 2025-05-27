# scripts/run_ensemble.py

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

from src.models.knn import KNNClassifier, optimize_k, optimize_p
from src.models.one_vs_rest import OneVsRestClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from src.utils.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
)
from src.visualization.plots import (
    ensure_directory,
    plot_confusion_matrix,
)

RESULTS_DIR = "docs/task_ensemble_results"


def describe_data(X: pd.DataFrame, y: pd.Series):
    print("=== Data description ===")
    print(f"Number of samples: {len(y)}")
    print(f"Number of features: {X.shape[1]}")
    print("Class counts:")
    print(y.value_counts())
    print("Balanced ratios:", y.value_counts(normalize=True).to_dict())


def evaluate_model(name, model, X_train, X_test, y_train, y_test, results_dir):
    # Jeśli to własny KNN, przekonwertuj y na NumPy
    if isinstance(model, KNNClassifier):
        y_train_fit = np.array(y_train)
        y_test_pred = y_test  # zostawimy do metryk
    else:
        y_train_fit = y_train
    # Trening i predykcja
    model.fit(X_train, y_train_fit)
    y_pred = model.predict(X_test)

    # Metryki macro
    acc_mac = accuracy(y_test, y_pred)
    prec_mac = precision(y_test, y_pred, average="macro")
    rec_mac = recall(y_test, y_pred, average="macro")
    f1_mac = f1_score(y_test, y_pred, average="macro")
    # Metryki micro
    acc_mic = accuracy(y_test, y_pred)
    prec_mic = precision(y_test, y_pred, average="micro")
    rec_mic = recall(y_test, y_pred, average="micro")
    f1_mic = f1_score(y_test, y_pred, average="micro")

    print(f"\n{name}:")
    print(
        f"  Macro →  Acc:{acc_mac:.3f}  Prec:{prec_mac:.3f}  Rec:{rec_mac:.3f}  F1:{f1_mac:.3f}"
    )
    print(
        f"  Micro →  Acc:{acc_mic:.3f}  Prec:{prec_mic:.3f}  Rec:{rec_mic:.3f}  F1:{f1_mic:.3f}"
    )

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, results_dir, filename=f"cm_{name}.png")


def main():
    results_dir = Path(RESULTS_DIR)
    ensure_directory(results_dir)

    # Wczytanie i opis danych
    data = load_wine()
    X_df = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    describe_data(X_df, y)

    # Podział
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)
    normalizer = Normalizer(norm="l2")
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)

    # 1) Baseline: własny KNN
    evaluate_model(
        "KNN_wlasny",
        KNNClassifier(k=3, metric="euclidean"),
        X_train,
        X_test,
        y_train,
        y_test,
        results_dir,
    )

    # 2) Baseline: własny Perceptron-OVR
    evaluate_model(
        "Perceptron_OVR",
        OneVsRestClassifier(learning_rate=0.01, n_iter=1000, tolerance=1e-4),
        X_train,
        X_test,
        y_train,
        y_test,
        results_dir,
    )

    # 3) RandomForest
    evaluate_model(
        "RandomForest",
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train,
        X_test,
        y_train,
        y_test,
        results_dir,
    )

    # 4) Bagging z DecisionTree
    evaluate_model(
        "Bagging_DT",
        BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=50,
            random_state=42,
        ),
        X_train,
        X_test,
        y_train,
        y_test,
        results_dir,
    )

    # 5) GradientBoosting
    evaluate_model(
        "GradientBoosting",
        GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        X_train,
        X_test,
        y_train,
        y_test,
        results_dir,
    )

    # 6) HistGradientBoosting
    evaluate_model(
        "HistGradientBoosting",
        HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, random_state=42
        ),
        X_train,
        X_test,
        y_train,
        y_test,
        results_dir,
    )


if __name__ == "__main__":
    main()
