import numpy as np
from typing import Sequence, Union


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Compute the classification accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def precision(y_true: Sequence, y_pred: Sequence, average: str = "macro") -> float:
    """Compute precision with micro or macro averaging."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))

    if average == "macro":
        scores = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            scores.append(tp / (tp + fp) if tp + fp else 0.0)
        return float(np.mean(scores))
    elif average == "micro":
        tp = np.sum((y_true == y_pred))
        fp = np.sum((y_true != y_pred))
        return float(tp / (tp + fp)) if tp + fp else 0.0
    else:
        raise ValueError("average must be 'macro' or 'micro'")


def recall(y_true: Sequence, y_pred: Sequence, average: str = "macro") -> float:
    """Compute recall with micro or macro averaging."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))

    if average == "macro":
        scores = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            scores.append(tp / (tp + fn) if tp + fn else 0.0)
        return float(np.mean(scores))
    elif average == "micro":
        tp = np.sum((y_true == y_pred))
        fn = np.sum((y_true != y_pred))
        return float(tp / (tp + fn)) if tp + fn else 0.0
    else:
        raise ValueError("average must be 'macro' or 'micro'")


def f1_score(y_true: Sequence, y_pred: Sequence, average: str = "macro") -> float:
    """Compute F1-score with micro or macro averaging."""
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    return float((2 * p * r) / (p + r)) if p + r else 0.0
