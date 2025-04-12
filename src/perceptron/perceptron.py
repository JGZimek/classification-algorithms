import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

from src.utils.utils import (
    accuracy,
    precision,
    recall,
    f1_score,
)

results_dir = "docs/task_perceptron_results"
os.makedirs(results_dir, exist_ok=True)

data = pd.read_csv("data/data_banknote_authentication.csv", header=None)


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
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0

        for _ in range(self.n_iter):
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
                break
        return self

    def predict(self, X_test):
        """
        Przewiduje etykiety dla danych testowych.
        """
        linear_output = np.dot(X_test, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

learning_rates = np.linspace(0.001, 0.1, 10)

acc_list = []
prec_list = []
rec_list = []
f1_list = []

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
