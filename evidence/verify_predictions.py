"""Recalculate classification metrics and compare independent runs from predictions."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def probability_column(row, label):
    variants = [f"prob_{label}", f"p_{label}", f"prob_{label.lower().replace(' ', '_')}"]
    for field in variants:
        if field in row:
            return field
    raise KeyError((label, list(row)))


def independent_scores(rows, labels):
    pairs = Counter((row["gold"], row["prediction"]) for row in rows)
    assert all(gold in labels and pred in labels for gold, pred in pairs), pairs
    matrix = [[pairs[g, p] for p in labels] for g in labels]
    classes = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        gold_count = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        classes.append({"label": label, "support": gold_count, "precision": tp / predicted_count if predicted_count else 0,
                        "recall": tp / gold_count if gold_count else 0,
                        "f1": 2 * tp / (gold_count + predicted_count) if (gold_count + predicted_count) else 0})
    return {"accuracy": sum(matrix[i][i] for i in range(len(labels))) / len(rows),
            "macro_precision": sum(row["precision"] for row in classes) / len(classes),
            "macro_recall": sum(row["recall"] for row in classes) / len(classes),
            "macro_f1": sum(row["f1"] for row in classes) / len(classes),
            "confusion_matrix": matrix, "per_class": classes, "count": len(rows)}


def equal(a, b):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9)


def keyed_rows(path):
    rows = read_csv(path)
    by_id = {row["id"]: row for row in rows}
    assert len(rows) == len(by_id), f"Duplicate ID: {path}"
    return by_id


def prediction_path(root, key, meta):
    folder_default = root / "predictions" / f"{key}.csv"
    if folder_default.is_file():
        return folder_default
    supplied = Path(meta["prediction_file"])
    for candidate in (root / supplied, root / "predictions" / supplied.name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError((key, supplied))


def compare_metric(key, rule, values):
    for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "count"):
        assert equal(values[metric], rule[metric]), (key, metric, values[metric], rule[metric])
    assert values["confusion_matrix"] == rule["confusion_matrix"], key
    reported = rule.get("per_class", {})
    if isinstance(reported, list):
        reported = {row["label"]: row for row in reported}
    for row in values["per_class"]:
        assert row["label"] in reported, (key, row["label"], list(reported))
        for field in ("support", "precision", "recall", "f1"):
            assert equal(row[field], reported[row["label"]][field]), (key, row["label"], field)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_set(root):
    rules = json.loads((root / "metrics.json").read_text(encoding="utf-8-sig"))
    output = {}
    for key, record in rules.items():
        labels = record["labels"]
        path = prediction_path(root, key, record)
        row_dict = keyed_rows(path)
        rows = list(row_dict.values())
        cols = [probability_column(rows[0], label) for label in labels]
        maximum_sum_error = 0
        for row in rows:
            probabilities = [float(row[col]) for col in cols]
            assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities), (key, row["id"])
            maximum_sum_error = max(maximum_sum_error, abs(sum(probabilities) - 1))
            assert probabilities[labels.index(row["prediction"])] >= max(probabilities) - 1e-7, (key, row["id"])
        assert maximum_sum_error <= 1e-6, (key, maximum_sum_error)
        stats = independent_scores(rows, labels)
        compare_metric(key, record, stats)
        output[key] = {"rows": row_dict, "columns": cols, "scores": stats,
                       "sha256": digest(path), "maximum_probability_sum_error": maximum_sum_error}
    return output


def compare_runs(first, second):
    assert first.keys() == second.keys()
    updates = {}
    for key, record in first.items():
        other = second[key]
        assert record["rows"].keys() == other["rows"].keys(), key
        diffs = []
        flips = []
        for ident, row in record["rows"].items():
            replay = other["rows"][ident]
            assert row["gold"] == replay["gold"], (key, ident)
            if row["prediction"] != replay["prediction"]:
                flips.append(ident)
            diffs.extend(abs(float(row[c]) - float(replay[c])) for c in record["columns"])
        updates[key] = {"count": len(record["rows"]), "hard_prediction_changed": len(flips), "changed_ids": flips,
                        "max_probability_abs_delta": max(diffs),
                        "macro_f1_abs_delta": abs(record["scores"]["macro_f1"] - other["scores"]["macro_f1"]),
                        "run_1_prediction_sha256": record["sha256"], "run_2_prediction_sha256": other["sha256"]}
    return updates


def validate_scope(first):
    source_gold = {i:r["gold"] for i,r in first["grouped_tfidf"]["rows"].items()}
    assert {i:r["gold"] for i,r in first["official_tfidf"]["rows"].items()} == source_gold
    target_gold = {i:r["gold"] for i,r in first["target_direct"]["rows"].items()}
    for name, item in first.items():
        if name.startswith("target_") or name.startswith("efficiency_"):
            assert {i:r["gold"] for i,r in item["rows"].items()} == target_gold, name
    return {"source_fixed_test": len(source_gold), "target_fixed_test": len(target_gold)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=Path("results/run_1"))
    parser.add_argument("--second", type=Path, default=Path("results/run_2"))
    parser.add_argument("--out", type=Path, default=Path("evidence/prediction_audit.json"))
    args = parser.parse_args()
    first = score_set(args.first)
    second = score_set(args.second)
    replay = compare_runs(first, second)
    summary = {"fixed_evaluation_sets": (validate_scope(first) if "grouped_tfidf" in first else {k:len(v["rows"]) for k,v in first.items()}), "predictions_verified": len(first),
               "recomputed": {k: v["scores"] for k, v in first.items()}, "independent_replay": replay,
               "max_probability_abs_delta": max(row["max_probability_abs_delta"] for row in replay.values()),
               "total_changed_hard_predictions": sum(row["hard_prediction_changed"] for row in replay.values())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"verified {len(first)} prediction files per run; hard-label changes={summary['total_changed_hard_predictions']};"
          f" max abs probability delta={summary['max_probability_abs_delta']:.10g}")


if __name__ == "__main__":
    main()
