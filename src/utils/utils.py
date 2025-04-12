import numpy as np

def accuracy(y_true, y_pred):
    """Oblicza dokładność klasyfikacji."""
    return np.mean(np.array(y_true) == np.array(y_pred))


# Ważna, gdy koszt fałszywych alarmów (FP) jest wysoki (np. w spam filtrach, diagnozach medycznych).
def precision(y_true, y_pred, average="macro"):
    """Oblicza precyzję. Obsługuje micro/macro averaging."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    if average == "macro":
        precisions = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(prec)
        return np.mean(precisions)
    elif average == "micro":
        tp_total = 0
        fp_total = 0
        for cls in classes:
            tp_total += np.sum((y_true == cls) & (y_pred == cls))
            fp_total += np.sum((y_true != cls) & (y_pred == cls))
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    else:
        raise ValueError("average musi być 'macro' lub 'micro'.")


# Ważna, gdy koszt przeoczenia (FN) jest wysoki (np. w wykrywaniu chorób, oszustw).
def recall(y_true, y_pred, average="macro"):
    """Oblicza recall. Obsługuje micro/macro averaging."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    if average == "macro":
        recalls = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(rec)
        return np.mean(recalls)
    elif average == "micro":
        tp_total = 0
        fn_total = 0
        for cls in classes:
            tp_total += np.sum((y_true == cls) & (y_pred == cls))
            fn_total += np.sum((y_true == cls) & (y_pred != cls))
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    else:
        raise ValueError("average musi być 'macro' lub 'micro'.")


def f1_score(y_true, y_pred, average="macro"):
    """Oblicza F1-score. Obsługuje micro/macro averaging."""
    prec = precision(y_true, y_pred, average)
    rec = recall(y_true, y_pred, average)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
