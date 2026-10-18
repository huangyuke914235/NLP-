"""Paired uncertainty checks for the QEvasion test-set Macro-F1 comparisons.

Run from ``code/`` after the existing prediction CSVs have been produced::

    python 08_bootstrap_uncertainty.py --project . --output-dir outputs

The answer-cluster analysis resamples identical answer texts as one unit.
It is a sensitivity check rather than a source-level analysis. The official
test retains source URLs: 09_review_validation.py adds URL-cluster intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


LABELS = ("Clear Reply", "Ambivalent Reply", "Clear Non-Reply")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def read_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate test IDs in {path}")
    return by_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def macro_f1(gold: list[int], predicted: list[int], sampled_rows: list[int]) -> float:
    true_positive = [0, 0, 0]
    gold_count = [0, 0, 0]
    predicted_count = [0, 0, 0]
    for row_index in sampled_rows:
        true_label = gold[row_index]
        predicted_label = predicted[row_index]
        gold_count[true_label] += 1
        predicted_count[predicted_label] += 1
        if true_label == predicted_label:
            true_positive[true_label] += 1
    scores = []
    for label in range(len(LABELS)):
        denominator = gold_count[label] + predicted_count[label]
        scores.append(2 * true_positive[label] / denominator if denominator else 0.0)
    return sum(scores) / len(LABELS)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def ci95(values: list[float]) -> list[float]:
    return [percentile(values, 0.025), percentile(values, 0.975)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing the existing outputs/ prediction CSVs")
    parser.add_argument("--output-dir", type=Path,
                        help="Where to write metrics_uncertainty.json (default: PROJECT/outputs)")
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    project = args.project.resolve()
    output_dir = (args.output_dir or project / "outputs").resolve()
    input_paths = {
        "tfidf": project / "outputs" / "predictions_tfidf_lr.csv",
        "minilm": project / "outputs" / "predictions_st_lr.csv",
        "hierarchy": project / "outputs" / "hierarchy" / "predictions_hierarchy.csv",
    }
    predictions = {name: read_predictions(path) for name, path in input_paths.items()}
    ids = sorted(predictions["tfidf"], key=int)
    if not ids or any(set(rows) != set(ids) for rows in predictions.values()):
        raise ValueError("Prediction CSVs do not contain the same nonempty set of test IDs")

    gold: list[int] = []
    tfidf: list[int] = []
    minilm: list[int] = []
    direct: list[int] = []
    hierarchy: list[int] = []
    clusters: dict[str, list[int]] = defaultdict(list)
    for row_index, test_id in enumerate(ids):
        baseline = predictions["tfidf"][test_id]
        semantic = predictions["minilm"][test_id]
        layered = predictions["hierarchy"][test_id]
        if (baseline["gold"] != semantic["gold"]
                or baseline["question"] != semantic["question"]
                or baseline["answer"] != semantic["answer"]
                or LABEL_TO_ID[baseline["gold"]] != int(layered["gold"])):
            raise ValueError(f"Gold label or sample text differs for test ID {test_id}")
        gold.append(LABEL_TO_ID[baseline["gold"]])
        tfidf.append(LABEL_TO_ID[baseline["prediction"]])
        minilm.append(LABEL_TO_ID[semantic["prediction"]])
        direct.append(int(layered["direct"]))
        hierarchy.append(int(layered["hierarchical"]))
        clusters[baseline["answer"]].append(row_index)
    if any(label not in range(len(LABELS)) for label in direct + hierarchy):
        raise ValueError("Layered prediction contains a label outside the three coarse classes")

    n = len(ids)
    all_rows = list(range(n))
    models = {"tfidf": tfidf, "minilm": minilm, "direct": direct,
              "hierarchy": hierarchy}
    point_scores = {name: macro_f1(gold, predicted, all_rows)
                    for name, predicted in models.items()}
    comparisons = {
        "minilm_minus_tfidf": ("minilm", "tfidf"),
        "hierarchy_minus_direct": ("hierarchy", "direct"),
    }

    def differences(sampled_rows: list[int]) -> dict[str, float]:
        sampled_scores = {name: macro_f1(gold, predicted, sampled_rows)
                          for name, predicted in models.items()}
        return {name: sampled_scores[positive] - sampled_scores[negative]
                for name, (positive, negative) in comparisons.items()}

    sample_rng = random.Random(args.seed)
    sampled_differences = {name: [] for name in comparisons}
    for _ in range(args.repetitions):
        sampled_rows = sample_rng.choices(all_rows, k=n)
        for name, difference in differences(sampled_rows).items():
            sampled_differences[name].append(difference)

    answer_groups = list(clusters.values())
    cluster_rng = random.Random(args.seed)
    clustered_differences = {name: [] for name in comparisons}
    for _ in range(args.repetitions):
        sampled_rows = [row for group in cluster_rng.choices(answer_groups,
                                                             k=len(answer_groups))
                        for row in group]
        for name, difference in differences(sampled_rows).items():
            clustered_differences[name].append(difference)

    result = {
        "seed": args.seed,
        "bootstrap_repetitions": args.repetitions,
        "metric": "Macro-F1 with all three coarse labels; zero_division=0",
        "confidence_interval": "2.5th and 97.5th percentiles, linear interpolation",
        "labels": list(LABELS),
        "test_samples": n,
        "unique_answer_clusters": len(answer_groups),
        "repeated_answer_clusters": sum(len(group) > 1 for group in answer_groups),
        "rows_in_repeated_answer_clusters": sum(len(group) for group in answer_groups
                                                 if len(group) > 1),
        "largest_answer_cluster": max(map(len, answer_groups)),
        "direct_vs_minilm_prediction_mismatches": sum(a != b for a, b in zip(direct, minilm)),
        "point_macro_f1": point_scores,
        "comparisons": {
            name: {
                "point_difference": point_scores[positive] - point_scores[negative],
                "paired_sample_bootstrap_ci95": ci95(sampled_differences[name]),
                "paired_answer_cluster_bootstrap_ci95": ci95(clustered_differences[name]),
            }
            for name, (positive, negative) in comparisons.items()
        },
        "resampling_note": (
            "The sample bootstrap resamples paired test IDs. The cluster bootstrap "
            "resamples identical answer texts and preserves all questions sharing an "
            "answer. These two checks do not account for source-level dependence. "
            "Official test URLs are available; see 09_review_validation.py for "
            "URL-cluster intervals. Intervals are descriptive, not a claim of "
            "population-level significance."
        ),
        "sha256": {
            str(path.relative_to(project)).replace("\\", "/"): sha256(path)
            for path in input_paths.values()
        } | {"script": sha256(Path(__file__))},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics_uncertainty.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(output_path),
                      "comparisons": result["comparisons"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
