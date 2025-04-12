import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


from src.utils.utils import (
    accuracy,
    precision,
    recall,
    f1_score,
)

results_dir = "docs/task_perceptron_results"
os.makedirs(results_dir, exist_ok=True)

data = pd.read_csv("data/data_banknote_authentication.csv", header=None)

print(data.head())


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        """
        Inicjalizacja klasyfikatora Perceptron.

        Parametry:
        - learning_rate: współczynnik uczenia z przedziału (0, 1),
        - n_iter: liczba iteracji.
        """

        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self

    def predict(self, X_test):
        pass
