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
        - learning_rate: współczynnik uczenia z przedziału (0, 1),
        - n_iter: maksymalna liczba iteracji,
        - tolerance: próg zbieżności dla zmian wag i biasu.
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
        - X_train: macierz cech (numpy array),
        - y_train: wektor etykiet (numpy array).
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

        Parametry:
        - X_test: macierz cech (numpy array).

        Zwraca:
        - Wektor przewidywanych etykiet.
        """
        linear_output = np.dot(X_test, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

perceptron = Perceptron(learning_rate=0.01, n_iter=1000, tolerance=1e-4)
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)

print("Accuracy:", accuracy(y_test, y_pred))
print("Precision:", precision(y_test, y_pred))
print("Recall:", recall(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
