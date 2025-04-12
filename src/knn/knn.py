import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from utils.utils import (
    accuracy,
    precision,
    recall,
    f1_score,
)

results_dir = "docs/task_knn_results"
os.makedirs(results_dir, exist_ok=True)


class KNNClassifier:
    def __init__(self, k=3, metric="euclidean", p=2):
        """
        Inicjalizacja klasyfikatora k-NN.

        Parametry:
        - k: liczba sąsiadów do głosowania,
        - metric: 'euclidean' lub 'minkowski'. Jeśli 'euclidean', p=2 jest używane.
        - p: parametr dla metryki Minkowskiego (ignorowany, jeśli metric=='euclidean').
        """
        self.k = k
        self.metric = metric
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []
        for x in X_test:
            if self.metric == "euclidean":
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            elif self.metric == "minkowski":
                distances = np.sum(np.abs(self.X_train - x) ** self.p, axis=1) ** (
                    1 / self.p
                )
            else:
                raise ValueError("Nieobsługiwana metryka: {}".format(self.metric))

            k_indices = np.argsort(distances)[: self.k]
            neighbor_labels = self.y_train[k_indices]

            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            pred_label = unique_labels[np.argmax(counts)]
            predictions.append(pred_label)
        return np.array(predictions)


def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            matrix[i, j] = np.sum((y_true == cls) & (y_pred == pred_cls))
    return pd.DataFrame(
        matrix,
        index=[f"Rzeczywiste: {c}" for c in classes],
        columns=[f"Predykowane: {c}" for c in classes],
    )


def optimize_k(X_train, y_train, X_val, y_val, k_range):
    errors = []
    for k in k_range:
        knn = KNNClassifier(k=k, metric="euclidean")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        err = 1 - accuracy(y_val, y_pred)
        errors.append(err)
    return errors


def optimize_p(X_train, y_train, X_val, y_val, p_values, k=3):
    errors = []
    for p in p_values:
        knn = KNNClassifier(k=k, metric="minkowski", p=p)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        err = 1 - accuracy(y_val, y_pred)
        errors.append(err)
    return errors


data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
print(X.head())
y = pd.Series(data.target)


np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
train_size = int(0.8 * len(X))
X_train_full = X.iloc[indices[:train_size]]
y_train_full = y.iloc[indices[:train_size]]
X_test = X.iloc[indices[train_size:]]
y_test = y.iloc[indices[train_size:]]


X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

k_range = range(1, 16)
errors_k = optimize_k(X_train, y_train, X_val, y_val, k_range)

plt.figure(figsize=(8, 6))
plt.plot(k_range, errors_k, marker="o", color="b")
plt.xlabel("Liczba sąsiadów (k)")
plt.ylabel("Błąd klasyfikacji (1 - accuracy)")
plt.title("Optymalizacja parametru k")
plt.grid(True)
plt.savefig(os.path.join(results_dir, "optimize_k.png"))
plt.show()

best_k = k_range[np.argmin(errors_k)]
print("Najlepsza wartość k:", best_k)

p_values = [1, 1.5, 2, 3, 4]
errors_p = optimize_p(X_train, y_train, X_val, y_val, p_values, k=best_k)

plt.figure(figsize=(8, 6))
plt.plot(p_values, errors_p, marker="o", color="g")
plt.xlabel("Wartość p w metryce Minkowskiego")
plt.ylabel("Błąd klasyfikacji (1 - accuracy)")
plt.title("Optymalizacja parametru p")
plt.grid(True)
plt.savefig(os.path.join(results_dir, "optimize_p.png"))
plt.show()

best_p = p_values[np.argmin(errors_p)]
print("Najlepsza wartość p:", best_p)

knn = KNNClassifier(k=best_k, metric="minkowski", p=best_p)
knn.fit(X_train_full, y_train_full)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
final_acc = accuracy(y_test, y_pred)
final_prec = precision(y_test, y_pred, average="macro")
final_rec = recall(y_test, y_pred, average="macro")
final_f1 = f1_score(y_test, y_pred, average="macro")

print("Ostateczna macierz pomyłek:")
print(cm, "\n")
print("Ostateczna Accuracy:", final_acc)
print("Ostateczna Precision (macro):", final_prec)
print("Ostateczna Recall (macro):", final_rec)
print("Ostateczna F1-score (macro):", final_f1)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Ostateczna macierz pomyłek")
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.show()

tsne = TSNE(n_components=2, random_state=42)
X_test_embedded = tsne.fit_transform(X_test)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_test_embedded[:, 0],
    X_test_embedded[:, 1],
    c=y_pred,
    cmap="viridis",
    edgecolor="k",
    s=50,
)
plt.title("t-SNE: Wizualizacja wyników k-NN (przewidywane klasy)")
plt.xlabel("Komponent 1")
plt.ylabel("Komponent 2")
plt.legend(*scatter.legend_elements(), title="Klasy")
plt.savefig(os.path.join(results_dir, "tsne_visualization.png"))
plt.show()
