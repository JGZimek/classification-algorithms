import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# ============================
# Implementacja klasy k-NN
# ============================
class KNNClassifier:
    def __init__(self, k=3, metric='euclidean'):
        """
        Inicjalizacja klasyfikatora k-NN.
        
        Parametry:
        - k: liczba sąsiadów, których głosy będą brane pod uwagę,
        - metric: metryka odległości (tylko 'euclidean' jest zaimplementowana).
        """
        self.k = k
        self.metric = metric

    def fit(self, X_train, y_train):
        """
        Zapamiętuje dane treningowe.
        
        Parametry:
        - X_train: macierz cech (np. DataFrame lub array),
        - y_train: etykiety klas (np. Series lub array).
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self

    def predict(self, X_test):
        """
        Dokonuje predykcji etykiet dla danych X_test.
        
        Parametry:
        - X_test: macierz cech, dla których chcemy przewidzieć etykiety.
        
        Zwraca:
        - Numpy array z przewidywanymi etykietami.
        """
        X_test = np.array(X_test)
        predictions = []
        
        for x in X_test:
            # Obliczanie odległości euklidesowych między x a wszystkimi punktami treningowymi
            if self.metric == 'euclidean':
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            else:
                raise ValueError("Nieobsługiwana metryka: {}".format(self.metric))
            
            # Wybór indeksów k najbliższych sąsiadów
            k_indices = np.argsort(distances)[:self.k]
            neighbor_labels = self.y_train[k_indices]
            
            # Głosowanie większościowe: wybór etykiety, która wystąpiła najczęściej
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            pred_label = unique_labels[np.argmax(counts)]
            predictions.append(pred_label)
        
        return np.array(predictions)

# ============================
# Implementacja macierzy pomyłek
# ============================
def confusion_matrix(y_true, y_pred):
    """
    Tworzy macierz pomyłek w postaci DataFrame.
    
    Wiersze odpowiadają etykietom rzeczywistym, a kolumny etykietom przewidywanym.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            matrix[i, j] = np.sum((y_true == cls) & (y_pred == pred_cls))
    return pd.DataFrame(matrix,
                        index=[f"Rzeczywiste: {c}" for c in classes],
                        columns=[f"Predykowane: {c}" for c in classes])

# ============================
# Implementacja metryk jakości
# ============================
def accuracy(y_true, y_pred):
    """
    Oblicza dokładność klasyfikacji.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, average='macro'):
    """
    Oblicza precyzję.
    
    Parametr average: 'macro' (domyślnie) lub 'micro'.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    if average == 'macro':
        precisions = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(p)
        return np.mean(precisions)
    
    elif average == 'micro':
        tp_total = 0
        fp_total = 0
        for cls in classes:
            tp_total += np.sum((y_true == cls) & (y_pred == cls))
            fp_total += np.sum((y_true != cls) & (y_pred == cls))
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    
    else:
        raise ValueError("average musi być 'macro' lub 'micro'.")

def recall(y_true, y_pred, average='macro'):
    """
    Oblicza recall.
    
    Parametr average: 'macro' (domyślnie) lub 'micro'.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    if average == 'macro':
        recalls = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(r)
        return np.mean(recalls)
    
    elif average == 'micro':
        tp_total = 0
        fn_total = 0
        for cls in classes:
            tp_total += np.sum((y_true == cls) & (y_pred == cls))
            fn_total += np.sum((y_true == cls) & (y_pred != cls))
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    
    else:
        raise ValueError("average musi być 'macro' lub 'micro'.")

def f1_score(y_true, y_pred, average='macro'):
    """
    Oblicza F1-score.
    """
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

# ============================
# Przykładowe użycie na Wine Dataset
# ============================
# Załadowanie danych
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Podział danych na treningowy (80%) i testowy (20%)
np.random.seed(42)  # dla powtarzalności
indices = np.arange(len(X))
np.random.shuffle(indices)
train_size = int(0.8 * len(X))

X_train = X.iloc[indices[:train_size]]
y_train = y.iloc[indices[:train_size]]
X_test = X.iloc[indices[train_size:]]
y_test = y.iloc[indices[train_size:]]

# Trenowanie klasyfikatora k-NN
knn = KNNClassifier(k=3, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Obliczenie macierzy pomyłek i metryk jakości
cm = confusion_matrix(y_test, y_pred)
acc = accuracy(y_test, y_pred)
prec_macro = precision(y_test, y_pred, average='macro')
rec_macro = recall(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

print("Macierz pomyłek:")
print(cm, "\n")
print("Accuracy:", acc)
print("Precision (macro):", prec_macro)
print("Recall (macro):", rec_macro)
print("F1-score (macro):", f1_macro)

# Opcjonalna wizualizacja macierzy pomyłek
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Macierz pomyłek")
plt.show()
