"""Train and track the aerial object classifier: the whole pipeline from the cloud data to a saved model.

Steps:
    1. Ingest: download the cleaned crops from Google Cloud Storage (src/ingest.py)
    2. Features: HOG and shape features for every crop (src/features.py)
    3. Split by flight sequence into train, validation and test, so frames from one flight never land on two sides
    4. Train the four models on the training flights and compare them on the validation flights
    5. Refit the best one on train + validation and score it once on the test flights
    6. Log everything to MLflow and save the model for the API

Crops from the same flight look alike (the same parked cars show up frame after frame), so a random split would put
near-copies of test objects into training and make the score look better than it is.

Models:
    baseline             always guesses the most common class (the floor to beat)
    logistic_regression  linear model on standardized features
    random_forest        decision trees
    svm                  support vector machine with an RBF kernel, the classic pairing for HOG features

Output:
    models/model.joblib, models/metadata.json   the chosen model and what the API needs to use it
    results/, figures/                          scores, per-class report and confusion matrix (src/evaluate.py)
    frontend/examples/                          a few test crops for the front-end
    mlruns/                                     MLflow runs and the registered model
"""

import json
import shutil
import subprocess

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import ROOT, load_config
from src.evaluate import CLASS_LABELS, evaluate_model
from src.features import features
from src.ingest import ingest

MODELS = ROOT / "models"
EXAMPLES = ROOT / "frontend" / "examples"
EXAMPLES_PER_CLASS = 4
EXAMPLE_MIN_SIDE_PX = 40  # bigger crops are easier to see in the front-end


def build_models(cfg, seed):
    """The four candidate models, with settings from config.yaml."""
    m = cfg["models"]
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(**m["logistic_regression"])),
        "random_forest": RandomForestClassifier(**m["random_forest"], n_jobs=-1, random_state=seed),
        "svm": make_pipeline(StandardScaler(), SVC(**m["svm"], probability=True, random_state=seed)),
    }


def load_features(data_dir, feature_cfg):
    """Compute the features of every crop listed in labels.csv."""
    labels = pd.read_csv(data_dir / "labels.csv", dtype={"crop_id": str, "sequence": str, "image_id": str})
    X = []
    for row in labels.itertuples():
        with Image.open(data_dir / "crops" / f"{row.crop_id}.png") as crop:
            X.append(features(crop, row.frame_w * row.frame_h, feature_cfg))
    return np.array(X), labels["label"].to_numpy(), labels["sequence"].to_numpy(), labels


def split_by_flight(X, y, groups, split_cfg, seed):
    """Hold out test_size of the flights for test, then val_size of the rest for validation."""
    outer = GroupShuffleSplit(n_splits=1, test_size=split_cfg["test_size"], random_state=seed)
    train_val, test = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=split_cfg["val_size"], random_state=seed)
    tr, va = next(inner.split(X[train_val], y[train_val], groups[train_val]))
    train, val = train_val[tr], train_val[va]
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not set(groups[a]) & set(groups[b])  # no flight is on two sides
    return train, val, test


def scores(model, X, y, prefix):
    pred = model.predict(X)
    return {f"{prefix}_accuracy": accuracy_score(y, pred), f"{prefix}_macro_f1": f1_score(y, pred, average="macro")}


def git_commit():
    """Short hash of the code that made this run, or "unknown" outside a git checkout (e.g. in Docker)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_examples(data_dir, labels, test, seed):
    """Copy a few test-flight crops to frontend/examples/. The model never trained on these flights."""
    shutil.rmtree(EXAMPLES, ignore_errors=True)
    EXAMPLES.mkdir(parents=True)
    pool = labels.iloc[test]
    pool = pool[pool[["w", "h"]].min(axis=1) >= EXAMPLE_MIN_SIDE_PX].sample(frac=1, random_state=seed)
    for label, group in pool.groupby("label"):
        for k, crop_id in enumerate(group["crop_id"].head(EXAMPLES_PER_CLASS), start=1):
            shutil.copy(data_dir / "crops" / f"{crop_id}.png", EXAMPLES / f"{label}_{k}.png")


def main():
    cfg = load_config()
    seed = cfg["seed"]
    commit = git_commit()

    # 1-2. Ingest from the cloud and compute features
    data_dir = ingest(cfg)
    X, y, groups, labels = load_features(data_dir, cfg["features"])
    print(f"Features: {X.shape[0]} crops x {X.shape[1]}")

    # 3. Split by flight
    train, val, test = split_by_flight(X, y, groups, cfg["split"], seed)
    for name, idx in [("Train", train), ("Validation", val), ("Test", test)]:
        print(f"{name:11} {len(idx):5} crops from {len(set(groups[idx]))} flights")

    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    metric = cfg["selection_metric"]

    with mlflow.start_run(run_name="pipeline") as parent:
        mlflow.log_params({"seed": seed, "git_commit": commit,
                           "data_url": cfg["data"]["url"], "data_sha256": cfg["data"]["sha256"],
                           **{f"split_{k}": v for k, v in cfg["split"].items()},
                           **{f"features_{k}": v for k, v in cfg["features"].items()},
                           "n_train": len(train), "n_val": len(val), "n_test": len(test)})
        mlflow.log_artifact(str(ROOT / "config.yaml"))

        # 4. Train every model on the training flights, compare on the validation flights (one MLflow run each)
        print("\nValidation scores:")
        val_scores = {}
        for name, model in build_models(cfg, seed).items():
            with mlflow.start_run(run_name=name, nested=True):
                model.fit(X[train], y[train])
                val_scores[name] = scores(model, X[val], y[val], "val")
                mlflow.log_param("model", name)
                mlflow.log_params(cfg["models"].get(name, {}))
                mlflow.log_metrics(val_scores[name])
            print(f"  {name:20} accuracy {val_scores[name]['val_accuracy']:.3f}   macro F1 {val_scores[name]['val_macro_f1']:.3f}")
        best = max(val_scores, key=lambda n: val_scores[n][metric])
        print(f"Chosen model ({metric}): {best}")

        # 5. Refit the chosen model on train + validation, then score the test flights once
        train_val = np.concatenate([train, val])
        final = build_models(cfg, seed)[best].fit(X[train_val], y[train_val])
        test_scores = scores(final, X[test], y[test], "test")
        mlflow.log_param("chosen_model", best)
        mlflow.log_metrics({**val_scores[best], **test_scores})
        for path in evaluate_model(final, X[test], y[test], best, len(set(groups[test])), val_scores, test_scores):
            mlflow.log_artifact(str(path))
        mlflow.sklearn.log_model(final, name="model", input_example=X[test][:2],
                                 registered_model_name=cfg["mlflow"]["registered_model"])
        run_id = parent.info.run_id

    # 6. Save the model and what the API needs to use it
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / "model.joblib")
    metadata = {
        "model_name": best,
        "classes": [str(c) for c in final.classes_],
        "display": CLASS_LABELS,
        "features": cfg["features"],
        "default_frame": [int(labels["frame_w"].median()), int(labels["frame_h"].median())],
        "mlflow_run_id": run_id,
        "val_accuracy": round(val_scores[best]["val_accuracy"], 4),
        "test_accuracy": round(test_scores["test_accuracy"], 4),
        "test_macro_f1": round(test_scores["test_macro_f1"], 4),
        "data_sha256": cfg["data"]["sha256"],
        "git_commit": commit,
    }
    (MODELS / "metadata.json").write_text(json.dumps(metadata, indent=2))
    save_examples(data_dir, labels, test, seed)
    print(f"Saved models/model.joblib, models/metadata.json and {len(list(EXAMPLES.glob('*.png')))} front-end examples")
    print(f"Default frame size for the API: {metadata['default_frame'][0]} x {metadata['default_frame'][1]}")


if __name__ == "__main__":
    main()