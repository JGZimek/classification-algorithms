from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = BASE_DIR / "docs"
KNN_RESULTS_DIR = RESULTS_DIR / "task_knn_results"
PERCEPTRON_RESULTS_DIR = RESULTS_DIR / "task_perceptron_results"

# Default hyperparameters
default_params = {
    "knn": {"k": 3, "metric": "euclidean", "p": 2},
    "perceptron": {"learning_rate": 0.01, "n_iter": 1000, "tolerance": 1e-4},
    "tsne": {"n_components": 2, "random_state": 42},
}
