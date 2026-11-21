"""Score the chosen model on the test flights and write the report files and the confusion matrix figure.

Called by src/train.py at the end of the pipeline.

Output:
    results/metrics.csv                validation scores for every model, test scores for the chosen one
    results/classification_report.txt  per-class precision, recall and F1 on the test flights
    figures/confusion_matrix.png       test set confusion matrix for the chosen model
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix

from src.config import ROOT

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MODEL_LABELS = {"baseline": "Baseline", "logistic_regression": "Logistic regression",
                "random_forest": "Random forest", "svm": "SVM (RBF kernel)"}
CLASS_LABELS = {"car_van": "Car / van", "person": "Person", "truck_bus": "Truck / bus", "two_wheeler": "Two-wheeler"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, SURFACE = "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", [SURFACE, "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])  # same blue steps as Project 2

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def confusion_figure(y_test, pred, model_name, n_flights):
    """Test set confusion matrix with the count in each cell."""
    classes = list(CLASS_LABELS)
    matrix = confusion_matrix(y_test, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(matrix, cmap=BLUES)
    for i in range(len(classes)):
        for j in range(len(classes)):
            dark_cell = matrix[i, j] > matrix.max() / 2
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=12, color=SURFACE if dark_cell else INK)
    ax.set_xticks(range(len(classes)), [CLASS_LABELS[c] for c in classes])
    ax.set_yticks(range(len(classes)), [CLASS_LABELS[c] for c in classes])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test set confusion matrix", pad=12)
    fig.text(0.01, -0.02, f"{model_name} on {len(y_test)} test crops from {n_flights} flights it never trained on.\n"
             "Rows are the true class, so the diagonal is the crops it got right.", fontsize=9, color=MUTED)
    path = FIGURES / "confusion_matrix.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def evaluate_model(model, X_test, y_test, best, n_flights, val_scores, test_scores):
    """Write the metrics table, the per-class report and the figure. Return their paths for MLflow."""
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    pred = model.predict(X_test)

    # 1. Validation scores for every model, test scores only for the chosen one (the test set is scored once)
    rows = [{"model": name,
             "val_accuracy": round(s["val_accuracy"], 3),
             "val_macro_f1": round(s["val_macro_f1"], 3),
             "test_accuracy": round(test_scores["test_accuracy"], 3) if name == best else None,
             "test_macro_f1": round(test_scores["test_macro_f1"], 3) if name == best else None}
            for name, s in val_scores.items()]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)

    # 2. Per-class report for the chosen model
    report = classification_report(y_test, pred, digits=3)
    (RESULTS / "classification_report.txt").write_text(f"Model: {best} (refit on train + validation flights)\n\n{report}")

    # 3. Confusion matrix
    figure = confusion_figure(y_test, pred, MODEL_LABELS[best], n_flights)

    print(f"\n{metrics.to_string(index=False)}\n\n{best} on the test flights\n{report}")
    return [RESULTS / "metrics.csv", RESULTS / "classification_report.txt", figure]