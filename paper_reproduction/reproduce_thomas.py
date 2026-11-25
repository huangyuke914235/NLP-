"""Re-run the released QEvasion annotation check and audit the evaluator inputs."""

import argparse
import ast
import csv
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import io
import json
from pathlib import Path
import pickle
import platform
import subprocess


REVISION = "ff78482fe3ee63b76ce427ea4b8ee2b690fe8896"
REPOSITORY = "https://github.com/konstantinosftw/Question-Evasion"
COUNTERFACTUAL_SCRIPT = "scripts/counterfactual_summaries_analysis.py"
TEST_FILE = "dataset/Inter-Annotator/test_set.csv"
ANNOTATION_COLUMNS = ("Annotator1", "Annotator2", "Annotator3")
EXPECTED_SOURCE_HASHES = {
    "scripts/utils.py": "c917bfb61da72ff8f6bf791520cb059ddb194f8c6d8b938e64305060fa531b56",
    COUNTERFACTUAL_SCRIPT: "adc23045098eb3d1aa3fe813a3a74f91f51f2f4011793b2a52d3831a309f3f1c",
    "scripts/results.py": "fdede244078a35db5af91a4ef6b3d36e67c580f306bcc7ab13975e98039f64ff",
}


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f"Unexpected named object in prediction file: {module}.{name}")

    def persistent_load(self, pid):
        raise ValueError(f"Unexpected persistent ID in prediction file: {pid!r}")


class Snapshot:
    def __init__(self, root):
        self.root = root
        self.hashes = {}

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args], capture_output=True, check=True
        ).stdout

    def read(self, relative):
        payload = self.git("show", f"{REVISION}:{relative}")
        digest = hashlib.sha256(payload).hexdigest()
        if relative in EXPECTED_SOURCE_HASHES and digest != EXPECTED_SOURCE_HASHES[relative]:
            raise ValueError(f"Unexpected author source hash: {relative} ({digest})")
        self.hashes[relative] = digest
        return payload


def load_functions(snapshot, relative, names, namespace):
    tree = ast.parse(snapshot.read(relative), filename=relative)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError(f"Missing expected functions in {relative}")
    source = ast.Module(body=functions, type_ignores=[])
    exec(compile(source, relative, "exec"), namespace)


def reproduce_counterfactual_check(snapshot):
    measured = []

    def accuracy_score(observed, assigned):
        matches = sum(left == right for left, right in zip(observed, assigned, strict=True))
        measured.append((matches, len(observed)))
        return matches / len(observed)

    # The released utility module also imports a pretrained sentence encoder.
    # The four functions used here only need SequenceMatcher and exact accuracy.
    namespace = {"SequenceMatcher": SequenceMatcher, "accuracy_score": accuracy_score}
    load_functions(snapshot, "scripts/utils.py", {"find_start_position"}, namespace)
    load_functions(
        snapshot,
        COUNTERFACTUAL_SCRIPT,
        {"extract_original_label", "extract_counterfactual_label", "find_disagreement"},
        namespace,
    )

    output = []
    for annotator in ANNOTATION_COLUMNS:
        path = f"dataset/Counterfactual-Summaries/{annotator}_countersummaries.json"
        samples = json.loads(snapshot.read(path))
        equality = namespace["find_disagreement"](samples)
        matches, evaluated = measured[-1]
        output.append({
            "annotator": annotator,
            "input_json_records": len(samples),
            "matched_subquestions_evaluated": evaluated,
            "selected_counterfactual_label": matches,
            "counterfactual_selection_rate": equality,
            "upstream_rounded_disagreement": round(1 - round(equality, 4), 4),
        })
    return output


def audit_release_evaluator(snapshot):
    tree = ast.parse(snapshot.read("scripts/results.py"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"mapping", "filenames"}
    }
    mapping = assignments["mapping"]
    raw_test = snapshot.read(TEST_FILE).decode("utf-8-sig")
    annotations = list(csv.DictReader(io.StringIO(raw_test)))
    label_counts = Counter(row[column] for row in annotations for column in ANNOTATION_COLUMNS)
    uncovered = {label: count for label, count in sorted(label_counts.items()) if label not in mapping}
    first_unmapped = next(
        (row[column] for row in annotations for column in ANNOTATION_COLUMNS if row[column] not in mapping),
        None,
    )

    missing_ids = {}
    paths = snapshot.git("ls-tree", "-r", "--name-only", REVISION, "results").decode().splitlines()
    for path in sorted(path for path in paths if path.endswith(".pickle")):
        indices, predictions = PrimitiveUnpickler(io.BytesIO(snapshot.read(path))).load()
        if len(indices) != len(predictions) or len(set(indices)) != len(indices):
            raise ValueError(f"Invalid prediction index at {path}")
        if missing := sorted(set(range(len(annotations))).difference(indices)):
            missing_ids[path] = missing

    duplicates = [key for key, count in Counter(assignments["filenames"]).items() if count > 1]
    return {
        "raw_test_rows": len(annotations),
        "raw_annotation_label_counts": dict(sorted(label_counts.items())),
        "unmapped_annotation_labels": uncovered,
        "first_unmapped_label": first_unmapped,
        "predictor_files_audited": sum(path.endswith(".pickle") for path in paths),
        "predictor_files_with_missing_test_ids": missing_ids,
        "duplicate_evaluator_inputs": duplicates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Clone of the public author repository")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = Snapshot(args.repo)
    check = reproduce_counterfactual_check(snapshot)
    evaluator = audit_release_evaluator(snapshot)
    output = {
        "upstream_repository": REPOSITORY,
        "upstream_commit": REVISION,
        "runtime": {"python": platform.python_version(), "os": platform.system()},
        "counterfactual_annotation_check": check,
        "evaluator_release_audit": evaluator,
        "input_sha256": snapshot.hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Upstream revision: {REVISION}")
    print(f"Runtime: Python {platform.python_version()} ({platform.system()})")
    for row in check:
        print(
            f"{row['annotator']}: {row['selected_counterfactual_label']}/"
            f"{row['matched_subquestions_evaluated']} selected the supplied counterfactual label, "
            f"rate={row['counterfactual_selection_rate']:.4f}; "
            f"released script 1-round(rate, 4)={row['upstream_rounded_disagreement']:.4f}"
        )
    print(f"Raw evaluator rows: {evaluator['raw_test_rows']}; "
          f"unmapped-label counts: {evaluator['unmapped_annotation_labels']}")
    print("Prediction files with missing IDs:", evaluator["predictor_files_with_missing_test_ids"])
    print(f"Wrote {args.output.name}")


if __name__ == "__main__":
    main()
