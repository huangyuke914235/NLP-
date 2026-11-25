"""Print live data-loading and paper-reproduction evidence for the reports."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def show_data():
    sys.path.insert(0, str(ROOT))
    from prepare_data import ensure_inputs, qevasion_rows, read_config

    sources = read_config()["qevasion"]
    adapter = read_json(ROOT / "configs" / "adapter.json")["source"]
    raw_dir = ROOT / "data" / "raw"
    checked = ensure_inputs(raw_dir, sources["files"], allow_download=False)
    rows, fields = qevasion_rows(raw_dir)
    by_partition = {
        split: Counter(row["label"] for row in rows if row["official_split"] == split)
        for split in fields
    }
    mapping = adapter["label_mapping"]
    if any(set(counts) != mapping.keys() for counts in by_partition.values()):
        raise ValueError("Raw labels and the configured adapter disagree")

    print(f"QEvasion release {sources['revision'][:12]} | verified Parquet SHA-256: {len(checked)}/2")
    print(f"Original files: {fields['train']['count']:,} train + {fields['test']['count']:,} test")
    print(f"{adapter['text_a_column']} -> text_a | {adapter['text_b_column']} -> text_b")
    print(f"{adapter['label_column']} -> label | {adapter['group_column']} -> interview group")
    print(f"{'Raw label':21} {'Mapped classifier label':24} {'Train':>6} {'Test':>5}")
    print("-" * 61)
    for raw, mapped in mapping.items():
        print(f"{raw:21} {mapped:24} {by_partition['train'][raw]:>6,} {by_partition['test'][raw]:>5,}")

    print("\nKey results loaded from completed evaluations")
    base = read_json(ROOT / "results" / "run_1" / "metrics.json")
    joint = read_json(ROOT / "results" / "pair_run_1" / "metrics.json")
    for name, item in [
        ("QEvasion grouped TF-IDF", base["grouped_tfidf"]),
        ("QNLI target TF-IDF", base["target_tfidf"]),
        ("QNLI target joint MiniLM", joint["target_pair"]),
    ]:
        print(f"{name:26}  Accuracy {item['accuracy']:.4f}  Macro-F1 {item['macro_f1']:.4f}")
    base_audit = read_json(ROOT / "evidence" / "prediction_audit.json")
    joint_audit = read_json(ROOT / "evidence" / "pair_prediction_audit.json")
    print(f"Prediction files compared per replay: {base_audit['predictions_verified']} base + "
          f"{joint_audit['predictions_verified']} joint")


def show_paper(author_repo, replay, compact):
    if replay.exists():
        raise FileExistsError(f"Use a new author reproduction output path: {replay}")
    print("Executing the released-author scoring component on the fixed snapshot.", flush=True)
    subprocess.run([
        sys.executable, "-u", str(ROOT / "paper_reproduction" / "reproduce_thomas.py"),
        "--repo", str(author_repo), "--output", str(replay),
    ], check=True, cwd=ROOT, stdout=subprocess.PIPE if compact else None)
    generated = read_json(replay)
    if compact:
        print(f"Upstream Git revision: {generated['upstream_commit']}")
        print("Annotator   Injected records   Matched items   Induced selections")
    for row in generated["counterfactual_annotation_check"]:
        if compact:
            print(f"{row['annotator']:11} {row['input_json_records']:>16} "
                  f"{row['matched_subquestions_evaluated']:>15} "
                  f"{row['selected_counterfactual_label']:>12} "
                  f"({row['counterfactual_selection_rate']:.2%})")
        else:
            print(f"{row['annotator']}: {row['input_json_records']} injected records -> "
                  f"{row['matched_subquestions_evaluated']} matched subquestions")
    original = ROOT / "paper_reproduction" / "author_artifacts.json"
    equal = replay.read_bytes() == original.read_bytes()
    print(f"Byte-identical to the archived reproduction: {equal}")
    if not equal and generated["counterfactual_annotation_check"] != (
        read_json(original)["counterfactual_annotation_check"]
    ):
        raise ValueError("Author scoring counts differ from the archived reproduction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="case", required=True)
    subparsers.add_parser("data")
    paper_parser = subparsers.add_parser("paper")
    paper_parser.add_argument("--author-repo", type=Path, required=True)
    paper_parser.add_argument("--replay", type=Path, required=True)
    paper_parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.case == "data":
        show_data()
    else:
        show_paper(args.author_repo.resolve(), args.replay.resolve(), args.compact)


if __name__ == "__main__":
    main()
