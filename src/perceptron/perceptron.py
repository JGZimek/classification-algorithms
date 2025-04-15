import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from src.utils.utils import (
    accuracy,
    precision,
    recall,
    f1_score,
)

results_dir = "docs/task_perceptron_results"
os.makedirs(results_dir, exist_ok=True)

data = pd.read_csv("data/data_banknote_authentication.csv", header=None)
data.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

print("Pierwsze 5 rekordów danych:")
print(data.head())

print("\nInformacje o zbiorze danych:")
print(data.info())

print("\nRozkład klas:")
print(data["class"].value_counts())

# Wizualizacja par cech (ekspoloracja danych)
sns.pairplot(data, hue="class")
plt.suptitle("Wizualizacja par cech", y=1.02)
plt.savefig(os.path.join(results_dir, "pairplot.png"))
plt.show()

# Wizualizacja danych w przestrzeni 2D przy użyciu t-SNE (redukcja wszystkich cech)
X_all = data.iloc[:, :-1].values  # wszystkie cechy
y_all = data["class"].values  # etykiety
tsne = TSNE(
    n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42
)
X_tsne = tsne.fit_transform(X_all)
plt.figure(figsize=(8, 6))
for label in np.unique(y_all):
    plt.scatter(
        X_tsne[y_all == label, 0],
        X_tsne[y_all == label, 1],
        label=f"Klasa {label}",
        alpha=0.7,
    )
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("Wizualizacja t-SNE danych Banknote Authentication")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "tsne_visualization.png"))
plt.show()


# IMPLEMENTACJA KLASYFIKATORA PERCEPTRON
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000, tolerance=1e-4):
        """
        Inicjalizacja klasyfikatora Perceptron.

        Parametry:
        - learning_rate: współczynnik uczenia (0,1),
        - n_iter: maksymalna liczba iteracji,
        - tolerance: próg zbieżności zmian wag i biasu.
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.weights = None
        self.bias = None

    def _activation_function(self, x):
        """
        Funkcja skokowa Heaviside'a.
        """
        return np.where(x >= 0, 1, 0)

    def fit(self, X_train, y_train):
        """
        Trenuje perceptron na danych treningowych.

        Parametry:
        - X_train: macierz cech,
        - y_train: wektor etykiet.
        """
        n_samples, n_features = X_train.shape
        # Inicjalizacja wag jako małe losowe wartości
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0

        for iteration in range(self.n_iter):
            total_update = 0
            for idx, x_j in enumerate(X_train):
                linear_output = np.dot(x_j, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)
                if y_predicted != y_train[idx]:
                    update = self.learning_rate * (y_train[idx] - y_predicted)
                    self.weights += update * x_j
                    self.bias += update
                    total_update += abs(update)
            if total_update < self.tolerance:
                print(f"Zbieżność osiągnięta po {iteration+1} iteracjach")
                break
        return self

    def predict(self, X_test):
        """
        Przewiduje etykiety dla danych testowych.
        """
        linear_output = np.dot(X_test, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted


# Przygotowanie danych do treningu (korzystamy z oryginalnych cech)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# OPTIMALIZACJA HIPERPARAMETRU learning_rate
learning_rates = np.linspace(0.001, 0.1, 10)

acc_list = []
prec_list = []
rec_list = []
f1_list = []

print("\nOptymalizacja hiperparametru learning_rate:")
for lr in learning_rates:
    perceptron = Perceptron(learning_rate=lr, n_iter=1000, tolerance=1e-4)
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)

    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred, average="macro")
    rec = recall(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    acc_list.append(acc)
    prec_list.append(prec)
    rec_list.append(rec)
    f1_list.append(f1)

    print(
        f"learning_rate: {lr:.3f} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}"
    )

# Wizualizacja wpływu learning_rate na wyniki klasyfikacji
plt.figure(figsize=(8, 6))
plt.plot(
    learning_rates, acc_list, marker="o", linestyle="-", color="blue", label="Accuracy"
)
plt.plot(
    learning_rates,
    prec_list,
    marker="s",
    linestyle="--",
    color="green",
    label="Precision (macro)",
)
plt.plot(
    learning_rates,
    rec_list,
    marker="^",
    linestyle="--",
    color="red",
    label="Recall (macro)",
)
plt.plot(
    learning_rates,
    f1_list,
    marker="d",
    linestyle="--",
    color="purple",
    label="F1-score (macro)",
)
plt.xlabel("Learning Rate")
plt.ylabel("Wartość metryki")
plt.title("Wpływ learning_rate na wyniki klasyfikacji")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "learning_rate_optimization.png"))
plt.show()

# WIZUALIZACJA GRANIC DECYZYJNYCH przy użyciu t-SNE
# Aby wizualizować granice decyzyjne na przestrzeni 2D otrzymanej przez t-SNE,
# najpierw redukujemy dane treningowe do 2D, a następnie trenujemy perceptron na tych danych.
tsne_model = TSNE(
    n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42
)
X_train_tsne = tsne_model.fit_transform(X_train)

# Trenowanie perceptronu na t-SNE zredukowanych danych
perceptron_tsne = Perceptron(learning_rate=0.01, n_iter=1000, tolerance=1e-4)
perceptron_tsne.fit(X_train_tsne, y_train)

# Przygotowanie siatki dla wizualizacji granic decyzyjnych w przestrzeni t-SNE
x_min, x_max = X_train_tsne[:, 0].min() - 5, X_train_tsne[:, 0].max() + 5
y_min, y_max = X_train_tsne[:, 1].min() - 5, X_train_tsne[:, 1].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_tsne = np.c_[xx.ravel(), yy.ravel()]

Z_tsne = perceptron_tsne.predict(grid_tsne)
Z_tsne = Z_tsne.reshape(xx.shape)

from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z_tsne, alpha=0.5, cmap=cmap_light)
plt.scatter(
    X_train_tsne[y_train == 0, 0],
    X_train_tsne[y_train == 0, 1],
    color="red",
    marker="o",
    edgecolor="k",
    label="Klasa 0",
)
plt.scatter(
    X_train_tsne[y_train == 1, 0],
    X_train_tsne[y_train == 1, 1],
    color="blue",
    marker="s",
    edgecolor="k",
    label="Klasa 1",
)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("Granice decyzyjne perceptronu na danych zredukowanych przez t-SNE")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "decision_boundaries_tsne.png"))
plt.show()
