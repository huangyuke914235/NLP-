"""Recalculate and compare the 19 fixed-answer cases in two runs."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from verify_predictions import read_csv


LABELS = ("Clear Reply", "Ambivalent", "Clear Non-Reply")
VARIANTS = ("original", "paraphrase", "changed")
MODELS = ("grouped_tfidf", "grouped_minilm", "grouped_minilm_interactions", "constant_ambivalent")


def summarize(rows):
    counts = Counter()
    by_label = Counter()
    for row in rows:
        gold, expected = row["gold"], row["changed_expected"]
        original, para, changed = (row[f"{variant}_prediction"] for variant in VARIANTS)
        original_correct = original == gold
        changed_correct = changed == expected
        requires_change = gold != expected
        changed_as_expected = requires_change and changed_correct and changed != original
        counts["original_correct_n"] += original_correct
        counts["paraphrase_stable_n"] += original == para
        counts["paraphrase_both_correct_n"] += original_correct and para == gold
        counts["changed_expected_label_correct_n"] += changed_correct
        counts["gold_requires_change_n"] += requires_change
        counts["correct_and_changed_on_required_n"] += changed_as_expected
        counts["required_and_original_correct_n"] += requires_change and original_correct
        counts["correct_change_given_original_correct_n"] += changed_as_expected and original_correct
        counts["same_scoped_refusal_preserved_n"] += changed_correct and row["change_kind"] == "same_scoped_refusal"
        by_label[expected] += changed_correct
    return {"n": len(rows), **counts, "changed_correct_by_expected": {label: by_label[label] for label in LABELS}}


def read_run(path, cases, source, verify_text):
    rows = read_csv(path)
    keyed = {(row["model"], row["id"]): row for row in rows}
    assert len(rows) == len(keyed) and keyed.keys() == {(model, ident) for model in MODELS for ident in cases}, path

    for (_, ident), row in keyed.items():
        case, original = cases[ident], source[ident]
        for field in ("changed_expected", "change_kind"):
            assert row[field] == case[field], (path, ident, field)
        assert row["gold"] == original["label"], (path, ident, "gold")
        assert row["gold_requires_change"] == str(row["gold"] != row["changed_expected"]), (path, ident)
        if verify_text:
            assert row["original_question"] == original["question"] and row["fixed_answer"] == original["answer"], ident
            for field in ("paraphrase", "changed_question", "paraphrase_rationale", "change_rationale"):
                assert row[field] == case[field], (path, ident, field)
            assert len({row["original_question"], row["paraphrase"], row["changed_question"]}) == 3, ident
        for variant in VARIANTS:
            probabilities = {label: float(row[f"{variant}_prob_{label}"]) for label in LABELS}
            assert all(0 <= value <= 1 for value in probabilities.values()), (path, ident, variant)
            assert math.isclose(sum(probabilities.values()), 1, rel_tol=0, abs_tol=1e-6), (path, ident, variant)
            assert probabilities[row[f"{variant}_prediction"]] >= max(probabilities.values()) - 1e-7, (path, ident, variant)

    summary = {
        "changed_expected_support": dict(Counter(case["changed_expected"] for case in cases.values())),
        "change_kind_support": dict(Counter(case["change_kind"] for case in cases.values())),
        "models": {model: summarize([keyed[model, ident] for ident in cases]) for model in MODELS},
    }
    reported = json.loads(path.with_name("counterfactual.json").read_text(encoding="utf-8-sig"))
    assert summary == reported, f"Recomputed counts differ from {path.with_name('counterfactual.json')}"
    return keyed, summary


def compare_models(first, second, ids):
    comparison = {}
    for model in MODELS:
        flips = {variant: [] for variant in VARIANTS}
        maximum = 0.0
        for ident in ids:
            old, new = first[model, ident], second[model, ident]
            for variant in VARIANTS:
                if old[f"{variant}_prediction"] != new[f"{variant}_prediction"]:
                    flips[variant].append(ident)
                for label in LABELS:
                    field = f"{variant}_prob_{label}"
                    maximum = max(maximum, abs(float(old[field]) - float(new[field])))
        comparison[model] = {
            "changed_ids": flips,
            "hard_prediction_changes": sum(map(len, flips.values())),
            "max_probability_abs_delta": maximum,
        }
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True, help="submitted counterfactual prediction CSV")
    parser.add_argument("--second", type=Path, required=True, help="new counterfactual prediction CSV with question text")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("configs/counterfactual_reviewed.csv"))
    parser.add_argument("--source", type=Path, default=Path("data/processed/qevasion_grouped.csv"))
    args = parser.parse_args()

    case_rows = read_csv(args.cases)
    cases = {row["id"]: row for row in case_rows}
    assert len(case_rows) == len(cases) == 19, "Expected 19 unique counterfactual cases"
    source = {row["id"]: row for row in read_csv(args.source) if row["split"] == "test" and row["id"] in cases}
    assert source.keys() == cases.keys(), "Every case must belong to the fixed test set"
    first, first_summary = read_run(args.first, cases, source, verify_text=False)
    second, second_summary = read_run(args.second, cases, source, verify_text=True)
    comparison = compare_models(first, second, cases)
    changes = sum(item["hard_prediction_changes"] for item in comparison.values())
    maximum = max(item["max_probability_abs_delta"] for item in comparison.values())
    result = {
        "cases_verified": len(cases), "models_verified": len(MODELS),
        "prediction_rows_per_run": len(first), "prediction_slots_per_run": len(first) * len(VARIANTS),
        "submitted": first_summary, "reproduced": second_summary, "comparison": comparison,
        "model_summaries_identical": first_summary == second_summary,
        "total_changed_hard_predictions": changes, "max_probability_abs_delta": maximum,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(cases)} cases / {len(MODELS)} models; hard-label changes={changes}; max abs probability delta={maximum:.10g}")


if __name__ == "__main__":
    main()
