"""Verify source truncation metrics and compare the original prefix checkpoint by length slice."""
import argparse
from collections import Counter
import json
from pathlib import Path

from verify_predictions import compare_metric, compare_runs, independent_scores, keyed_rows, score_set


def split_metrics(rows, tokens, labels):
    assert rows and set(rows).issubset(tokens)
    for ident, row in rows.items():
        tok = tokens[ident]
        assert row["gold"] == tok["gold"], ident
        assert (tok["truncated"] == "True") == (int(tok["answer_wordpieces"]) > int(tok["answer_wordpiece_budget"])), ident
    selections = {
        "all": list(rows.values()),
        "truncated": [row for ident, row in rows.items() if tokens[ident]["truncated"] == "True"],
        "not_truncated": [row for ident, row in rows.items() if tokens[ident]["truncated"] == "False"],
    }
    results = {}
    for name, subset in selections.items():
        info = independent_scores(subset, labels)
        pred = Counter(row["prediction"] for row in subset)
        info["prediction_counts"] = {label: pred[label] for label in labels}
        results[name] = info
    return results


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_baseline_validation(baseline, primary):
    original = read_json(baseline / "selection.json")["grouped_pair"]
    repeated = read_json(primary / "selection.json")["head"]
    for prev, now in zip(original["candidates"], repeated["candidates"], strict=True):
        assert prev["epoch"] == now["epoch"]
        assert prev["mean_batch_loss"] == now["mean_batch_loss"]
        observed = now["validation"]
        assert prev["validation_macro_f1"] == observed["macro_f1"]
        assert prev["validation_accuracy"] == observed["accuracy"]
        assert prev["validation_per_class"] == observed["per_class"]
    assert original["selected"]["epoch"] == repeated["selected"]["epoch"]
    return len(original["candidates"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary", type=Path, default=Path("results/truncation_run_1"))
    ap.add_argument("--replay", type=Path, default=None)
    ap.add_argument("--baseline", type=Path, default=Path("results/pair_run_1"))
    ap.add_argument("--output", type=Path, default=Path("evidence/source_truncation_audit.json"))
    args = ap.parse_args()
    first = score_set(args.primary)
    tokens = keyed_rows(args.primary / "tokenization_audit.csv")
    slice_record = read_json(args.primary / "slices.json")
    chosen = slice_record["selected"]
    metrics = read_json(args.primary / "metrics.json")
    labels = next(iter(metrics.values()))["labels"]
    computed = {}
    for candidate in slice_record["candidates"]:
        key = f"truncation_{candidate}_validation"
        subset = split_metrics(first[key]["rows"], tokens, labels)
        for section, record in subset.items():
            reported = slice_record["validation"][candidate][section]
            compare_metric(f"{candidate}/{section}", reported, record)
            assert record["prediction_counts"] == reported["prediction_counts"]
        computed[candidate] = subset
    chosen_test = split_metrics(first[f"truncation_{chosen}_test"]["rows"], tokens, labels)
    for section, record in chosen_test.items():
        reported = slice_record["selected_test"][chosen][section]
        compare_metric(f"selected/{section}", reported, record)
        assert record["prediction_counts"] == reported["prediction_counts"]
    base_run = score_set(args.baseline)
    old_test_rows = base_run["grouped_pair"]["rows"]
    assert old_test_rows.keys() == first[f"truncation_{chosen}_test"]["rows"].keys()
    original = split_metrics(old_test_rows, tokens, labels)
    summary = {
        "selected": chosen,
        "baseline_test": original,
        "validation": computed,
        "selected_test": chosen_test,
        "original_validation_history_matched_epochs": check_baseline_validation(args.baseline, args.primary),
        "replayed": False,
        "prediction_file_count": len(first),
    }
    if args.replay:
        second = score_set(args.replay)
        assert tokens == keyed_rows(args.replay / "tokenization_audit.csv")
        assert slice_record == read_json(args.replay / "slices.json")
        comparison = compare_runs(first, second)
        summary["independent_replay"] = comparison
        summary["replayed"] = True
        summary["max_probability_abs_delta"] = max(item["max_probability_abs_delta"] for item in comparison.values())
        summary["total_changed_hard_predictions"] = sum(item["hard_prediction_changed"] for item in comparison.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(first)} classifier prediction files; selected configuration: {chosen}")


if __name__ == "__main__":
    main()
