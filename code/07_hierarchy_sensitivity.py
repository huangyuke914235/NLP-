"""Sensitivity checks for the QEvasion fine-to-coarse hierarchy.

Run from the repository's ``code`` directory, or pass ``--project-dir`` when
running a copy elsewhere. Only the requested output directory is written.
The nine-class probabilities are mutually exclusive, so their sum within a
coarse class is the model-implied coarse probability. Dividing by the number
of child classes is an alternative *decision score*, not a probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


SEED = 42
C = 0.1
MAX_ITER = 2000
COARSE = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
FINE = [
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Partial/half-answer",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
]
FINE_TO_COARSE = [0, 1, 1, 1, 1, 1, 2, 2, 2]
CHILD_COUNTS = [1, 5, 3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing data/, outputs/, deps/, and 05_hierarchy.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination for metrics_sensitivity.json (default: outputs/hierarchy/)",
    )
    args = parser.parse_args()
    project = args.project_dir.resolve()
    output = (args.output_dir or project / "outputs" / "hierarchy").resolve()

    sys.path.insert(0, str(project / "deps"))
    import datasets
    import numpy as np
    import pandas as pd
    import sklearn
    from datasets import load_from_disk
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import GroupShuffleSplit

    def fit(features, labels, class_weight):
        return LogisticRegression(
            C=C,
            max_iter=MAX_ITER,
            class_weight=class_weight,
            random_state=SEED,
        ).fit(features, labels)

    def summarize(gold, predicted):
        label_ids = list(range(len(COARSE)))
        return {
            "n": int(len(gold)),
            "accuracy": float(accuracy_score(gold, predicted)),
            "macro_f1": float(
                f1_score(gold, predicted, labels=label_ids, average="macro", zero_division=0)
            ),
            "gold_counts": np.bincount(gold, minlength=len(COARSE)).astype(int).tolist(),
            "predicted_counts": np.bincount(predicted, minlength=len(COARSE)).astype(int).tolist(),
            "confusion_matrix": confusion_matrix(gold, predicted, labels=label_ids).tolist(),
        }

    def coarse_scores(model, features):
        fine_probabilities = model.predict_proba(features)
        scores = np.zeros((len(features), len(COARSE)), dtype=float)
        for column, fine_id in enumerate(model.classes_):
            scores[:, FINE_TO_COARSE[int(fine_id)]] += fine_probabilities[:, column]
        return scores

    data_dir = project / "data" / "qevasion_raw"
    cache_path = project / "outputs" / "st_embeddings.npz"
    source_data = load_from_disk(str(data_dir))
    train = source_data["train"].to_pandas()
    test = source_data["test"].to_pandas()
    cache = np.load(cache_path)
    X_train, X_test = cache["X_train"], cache["X_test"]
    coarse_id = {name: i for i, name in enumerate(COARSE)}
    fine_id = {name: i for i, name in enumerate(FINE)}
    y_train = train.clarity_label.map(coarse_id).to_numpy()
    y_test = test.clarity_label.map(coarse_id).to_numpy()
    fine_train = train.evasion_label.map(fine_id).to_numpy()
    if pd.isna(y_train).any() or pd.isna(y_test).any() or pd.isna(fine_train).any():
        raise ValueError("Unexpected label in the saved QEvasion data")
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    fine_train = fine_train.astype(int)
    if not (
        len(X_train) == len(train)
        and len(X_test) == len(test)
        and np.array_equal(y_train, cache["y_train"])
        and np.array_equal(y_test, cache["y_test"])
        and np.array_equal(np.asarray(FINE_TO_COARSE)[fine_train], y_train)
    ):
        raise ValueError("Data order, cached labels, or fine-to-coarse mapping differs")

    fit_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(
            X_train, y_train, groups=train.title
        )
    )
    fit_titles = set(train.iloc[fit_idx].title)
    val_titles = set(train.iloc[val_idx].title)
    if fit_titles & val_titles:
        raise ValueError("Title-group leakage in validation split")

    direct_validation = fit(X_train[fit_idx], y_train[fit_idx], "balanced")
    direct_official = fit(X_train, y_train, "balanced")
    direct_reference = {
        "validation": summarize(y_train[val_idx], direct_validation.predict(X_train[val_idx])),
        "official_test": summarize(y_test, direct_official.predict(X_test)),
    }

    variants = []
    for class_weight, aggregations in [
        ("balanced", ["sum", "child_count_normalized"]),
        (None, ["sum"]),
    ]:
        val_model = fit(X_train[fit_idx], fine_train[fit_idx], class_weight)
        test_model = fit(X_train, fine_train, class_weight)
        val_sum = coarse_scores(val_model, X_train[val_idx])
        test_sum = coarse_scores(test_model, X_test)
        for aggregation in aggregations:
            divisor = np.asarray(CHILD_COUNTS) if aggregation == "child_count_normalized" else 1
            val_prediction = (val_sum / divisor).argmax(axis=1)
            test_prediction = (test_sum / divisor).argmax(axis=1)
            variants.append(
                {
                    "name": f"{'balanced' if class_weight else 'unweighted'}_{aggregation}",
                    "class_weight": class_weight,
                    "aggregation": aggregation,
                    "score_interpretation": (
                        "model-implied coarse probability"
                        if aggregation == "sum"
                        else "heuristic decision score; not a normalized probability"
                    ),
                    "validation": summarize(y_train[val_idx], val_prediction),
                    "official_test": summarize(y_test, test_prediction),
                }
            )

    input_paths = [
        cache_path,
        data_dir / "train" / "data-00000-of-00001.arrow",
        data_dir / "test" / "data-00000-of-00001.arrow",
        project / "05_hierarchy.py",
        project / "outputs" / "hierarchy" / "metrics_hierarchy.json",
    ]
    result = {
        "schema_version": 1,
        "purpose": "Post-hoc sensitivity analysis, not official-test model selection",
        "notes": [
            "Nine fine labels map exactly to three coarse labels; summing mutually exclusive fine-label probabilities is the standard coarse marginalization.",
            "Dividing coarse sums by the number of child labels (1, 5, 3) is an alternative decision rule, not a calibrated coarse-class probability.",
            "All three variants reuse the same frozen embeddings, C, split seed, and label mapping; only class weighting and decision rule vary.",
            "Official test scores are descriptive; do not choose a model or tune hyperparameters on this test set.",
        ],
        "seed": SEED,
        "C": C,
        "max_iter": MAX_ITER,
        "feature_dim": int(X_train.shape[1]),
        "coarse_class_order": COARSE,
        "fine_class_order": FINE,
        "fine_to_coarse": {name: COARSE[FINE_TO_COARSE[i]] for i, name in enumerate(FINE)},
        "child_counts": dict(zip(COARSE, CHILD_COUNTS)),
        "validation_split": {
            "method": "GroupShuffleSplit by title",
            "test_size": 0.2,
            "fit_n": int(len(fit_idx)),
            "validation_n": int(len(val_idx)),
            "fit_title_groups": int(len(fit_titles)),
            "validation_title_groups": int(len(val_titles)),
            "title_group_overlap": 0,
        },
        "official_train_n": int(len(train)),
        "official_test_n": int(len(test)),
        "direct_three_class_reference": direct_reference,
        "variants": variants,
        "input_sha256": {
            str(path.relative_to(project)).replace("\\", "/"): sha256(path)
            for path in input_paths
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "datasets": datasets.__version__,
        },
    }
    previous_path = project / "outputs" / "hierarchy" / "metrics_hierarchy.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    original = variants[0]
    for split, previous_key in [
        ("validation", "group_validation_hierarchy"),
        ("official_test", "official_hierarchy"),
    ]:
        for metric in ["accuracy", "macro_f1"]:
            if abs(original[split][metric] - previous[previous_key][metric]) > 1e-12:
                raise ValueError(f"Original {split} {metric} no longer reproduces")

    output.mkdir(parents=True, exist_ok=True)
    destination = output / "metrics_sensitivity.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)
    for variant in variants:
        print(
            variant["name"],
            "validation_macro_f1=",
            round(variant["validation"]["macro_f1"], 6),
            "test_accuracy=",
            round(variant["official_test"]["accuracy"], 6),
            "test_macro_f1=",
            round(variant["official_test"]["macro_f1"], 6),
            "test_prediction_counts=",
            variant["official_test"]["predicted_counts"],
        )


if __name__ == "__main__":
    main()
