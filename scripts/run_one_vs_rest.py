from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from src.data.loader import load_wine_data
from src.models.one_vs_rest import OneVsRestClassifier
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
    plot_scaling_normalization_demo,
)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine


def main():
    # create results directory
    results_dir = Path("docs/task_ovr_wine_results")
    ensure_directory(results_dir)

    # load raw data for demonstration
    data = load_wine()
    df_raw = pd.DataFrame(data.data, columns=data.feature_names)

    # select two features with very different scales
    demo_feats = ["malic_acid", "proline"]

    # prepare DataFrame for demo
    df = df_raw[demo_feats]

    # original distributions
    df_orig = df.copy()

    # standardized (mean=0, std=1)
    scaler = StandardScaler()
    df_std = pd.DataFrame(scaler.fit_transform(df), columns=demo_feats)

    # normalized (each sample to unit length)
    normalizer = Normalizer(norm="l2")
    df_norm = pd.DataFrame(normalizer.fit_transform(df_std), columns=demo_feats)

    # plot distributions before and after
    plot_scaling_normalization_demo(df_orig, df_std, df_norm, demo_feats, results_dir)

    # now load and split for modeling
    X, y = load_wine_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # standardization (mean=0, std=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # vector normalization (L2)
    normalizer = Normalizer(norm="l2")
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)

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
