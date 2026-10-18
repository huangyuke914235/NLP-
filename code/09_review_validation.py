"""Read-only scientific checks and supplementary sensitivity analyses.

Only --out (default BASE/outputs/review) is written. Existing experiment outputs
are never replaced. No sentence encoder is loaded and no dependency is installed.
Example: python 09_review_validation.py --base E:/homework/nlp/code --out ./review
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
START = time.perf_counter()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--out", type=Path)
parser.add_argument("--repetitions", type=int, default=10_000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
if args.repetitions < 1:
    parser.error("--repetitions must be positive")
BASE = args.base.resolve()
OUT = (args.out or BASE / "outputs" / "review").resolve()
if OUT in {BASE, BASE / "outputs", BASE / "outputs" / "hierarchy"}:
    parser.error("--out must be a distinct review directory, not an original-output directory")

# 02/03 import numpy before common.py inserts the project's deps directory.
# Keep that order: otherwise a vendored numpy could silently replace the actual
# baseline runtime. The selected module paths are recorded below.
import numpy as np

sys.path.insert(0, str(BASE))
import common
import datasets
import pandas as pd
import sklearn
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split

LOG: list[str] = []
CHECKS: list[dict] = []
COARSE_RAW = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
COARSE = list(common.CLASS_ORDER)
COARSE_MAP = dict(zip(COARSE_RAW, range(3)))
FINE = ["Explicit", "Implicit", "Dodging", "General", "Deflection",
        "Partial/half-answer", "Declining to answer", "Claims ignorance", "Clarification"]
FINE_TO_COARSE = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2])
QNLI_NAMES = ["Entailment (contains answer)", "Not Entailment"]


def log(message: str) -> None:
    LOG.append(message)
    print(message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def check(name: str, passed, detail=None) -> None:
    item = {"name": name, "passed": bool(passed)}
    if detail is not None:
        item["detail"] = detail
    CHECKS.append(item)
    if not passed:
        log("FAILED: " + name + (" " + str(detail) if detail is not None else ""))


def read_json(relative: str):
    return json.loads((BASE / relative).read_text(encoding="utf-8"))


def metric(gold, pred, names) -> dict:
    gold, pred = np.asarray(gold, dtype=int), np.asarray(pred, dtype=int)
    ids = np.arange(len(names))
    p, r, f, n = precision_recall_fscore_support(gold, pred, labels=ids, zero_division=0)
    return {"n": int(len(gold)), "accuracy": float(accuracy_score(gold, pred)),
            "macro_p": float(p.mean()), "macro_r": float(r.mean()),
            "macro_f1": float(f.mean()), "gold_counts": n.astype(int).tolist(),
            "predicted_counts": np.bincount(pred, minlength=len(names)).tolist(),
            "confusion_matrix": confusion_matrix(gold, pred, labels=ids).tolist(),
            "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(n[i])}
                          for i, name in enumerate(names)}}


def compare_metrics(name: str, actual: dict, saved: dict) -> dict:
    differences = {key: abs(actual[key] - saved[key]) for key in
                   ["accuracy", "macro_p", "macro_r", "macro_f1"] if key in saved}
    cm_equal = actual["confusion_matrix"] == saved.get("confusion_matrix")
    per_class_differences = []
    for label, values in actual["per_class"].items():
        reference = saved.get("per_class", {}).get(label)
        if reference is None:
            continue
        for key in ["precision", "recall", "support"]:
            per_class_differences.append(abs(values[key] - reference[key]))
        per_class_differences.append(abs(values["f1"] - reference.get("f1", reference.get("f1-score"))))
    maximum = max([0.0] + list(differences.values()) + per_class_differences)
    check(name, maximum < 1e-12 and cm_equal, {"max_absolute_difference": maximum,
                                             "confusion_matrix_equal": cm_equal})
    return {"maximum_absolute_difference": maximum, "confusion_matrix_equal": cm_equal,
            "recomputed": actual}


def split_record(frame, fit, val, columns) -> dict:
    result = {"fit_n": int(len(fit)), "validation_n": int(len(val)),
              "fit_row_indices_zero_based": np.asarray(fit, dtype=int).tolist(),
              "validation_row_indices_zero_based": np.asarray(val, dtype=int).tolist(),
              "index_sha256_encoding": "UTF-8 JSON integer list, no spaces, original sklearn order",
              "fit_indices_sha256": canonical_hash(np.asarray(fit, dtype=int).tolist()),
              "validation_indices_sha256": canonical_hash(np.asarray(val, dtype=int).tolist()),
              "overlap": {}}
    for column in columns:
        left, right = frame.iloc[fit][column], frame.iloc[val][column]
        overlap = set(left) & set(right)
        result["overlap"][column] = {
            "shared_unique_values": len(overlap), "validation_unique_values": int(right.nunique()),
            "validation_rows_seen_in_fit": int(right.isin(overlap).sum())}
    return result


def read_qnli(path: Path) -> tuple[pd.DataFrame, dict]:
    rows, short, embedded_tabs = [], 0, 0
    with path.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\r\n").split("\t")
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                short += 1
                continue
            embedded_tabs += int(len(parts) > 4)
            rows.append({"index": parts[0], "question": parts[1],
                         "sentence": "\t".join(parts[2:-1]), "label": parts[-1]})
    frame = pd.DataFrame(rows, columns=header[:1] + ["question", "sentence", "label"])
    label_map = {"entailment": 0, "not_entailment": 1, "0": 0, "1": 1}
    frame["target"] = frame.label.astype(str).str.lower().map(label_map)
    if frame.target.isna().any():
        raise ValueError("Unknown QNLI label")
    frame["target"] = frame.target.astype(int)
    return frame, {"parsed_rows": len(rows), "skipped_short_lines": short,
                   "rows_with_sentence_tabs": embedded_tabs}


def lengths(frame: pd.DataFrame) -> dict:
    q, a = frame.question.str.split().str.len(), frame.sentence.str.split().str.len()
    return {"n": len(frame), "question_mean": float(q.mean()), "sentence_mean": float(a.mean()),
            "question_mean_1dp": round(float(q.mean()), 1),
            "sentence_mean_1dp": round(float(a.mean()), 1),
            "label_counts": np.bincount(frame.target, minlength=2).tolist()}


log("REVIEW START " + datetime.now(timezone(timedelta(hours=8))).isoformat())
log("READ-ONLY BASE: " + str(BASE))
log("NEW REVIEW OUTPUT: " + str(OUT))
environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
               "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
               "datasets": datasets.__version__, "module_paths": {
                   "numpy": np.__file__, "pandas": pd.__file__, "sklearn": sklearn.__file__,
                   "datasets": datasets.__file__, "common": common.__file__},
               "optional_modules_available": {name: importlib.util.find_spec(name) is not None
                                               for name in ["torch", "sentence_transformers", "transformers", "tokenizers"]}}
log("ENVIRONMENT " + json.dumps(environment, ensure_ascii=False))
input_files = ["configs/qevasion.json", "configs/qnli.json", "02_baselines.py", "03_plm_lr.py",
               "04_transfer_qnli.py", "05_hierarchy.py", "06_direct_transfer.py",
               "data/qevasion_raw/train/data-00000-of-00001.arrow",
               "data/qevasion_raw/test/data-00000-of-00001.arrow", "data/qnli_raw/train.tsv",
               "data/qnli_raw/dev.tsv", "outputs/st_embeddings.npz", "outputs/eda_stats.json",
               "outputs/metrics_baselines.json", "outputs/metrics_st_lr.json", "outputs/metrics_qnli_tfidf_lr.json",
               "outputs/predictions_tfidf_lr.csv", "outputs/predictions_st_lr.csv",
               "outputs/predictions_qnli_tfidf_lr.csv", "outputs/hierarchy/predictions_hierarchy.csv",
               "outputs/hierarchy/validation_hierarchy.csv", "outputs/hierarchy/metrics_hierarchy.json",
               "outputs/hierarchy/metrics_direct_transfer.json", "outputs/hierarchy/predictions_direct_transfer.csv"]
initial_hashes = {name: sha256(BASE / name) for name in input_files}
data = load_from_disk(str(BASE / "data" / "qevasion_raw"))
tr, te = data["train"].to_pandas(), data["test"].to_pandas()
y, yt = tr.clarity_label.map(COARSE_MAP).to_numpy(), te.clarity_label.map(COARSE_MAP).to_numpy()
fine = tr.evasion_label.map(dict(zip(FINE, range(9)))).to_numpy()
check("fine-to-coarse mapping exact", np.array_equal(FINE_TO_COARSE[fine], y))
url_overlap = set(tr.url) & set(te.url)
seen_url = te.url.isin(url_overlap).to_numpy()
source_audit = {
    "train_n": len(tr), "test_n": len(te), "train_unique_title_strings": int(tr.title.nunique()),
    "train_unique_source_urls": int(tr.url.nunique()), "test_unique_source_urls": int(te.url.nunique()),
    "shared_source_urls": len(url_overlap), "test_rows_source_url_seen_in_train": int(seen_url.sum()),
    "test_rows_source_url_seen_fraction": float(seen_url.mean()),
    "test_rows_source_url_unseen_in_train": int((~seen_url).sum()),
    "train_rows_source_url_shared_with_test": int(tr.url.isin(url_overlap).sum()),
    "max_unique_urls_per_title": int(tr.groupby("title").url.nunique().max()),
    "titles_with_multiple_urls": int((tr.groupby("title").url.nunique() > 1).sum()),
    "max_unique_titles_per_url": int(tr.groupby("url").title.nunique().max()),
    "test_blank_metadata_rows": {c: int(te[c].fillna("").astype(str).str.strip().eq("").sum())
                                 for c in ["title", "president", "url", "evasion_label"]},
    "exact_text_overlap_train_test": {c: len(set(tr[c]) & set(te[c]))
                                      for c in ["question", "interview_question", "interview_answer"]},
    "exact_question_answer_pair_overlap": len(set(zip(tr.question, tr.interview_answer)) & set(zip(te.question, te.interview_answer))),
    "train_duplicate_question_rows": int(tr.question.duplicated().sum()),
    "test_unique_answer_texts": int(te.interview_answer.nunique()),
    "interpretation": "175 title strings are not 175 interviews. Source URL overlap is observable; official test performance is not a fully source-independent evaluation."}
log("SOURCE AUDIT " + json.dumps(source_audit, ensure_ascii=False))

# Reconstruct the adapter's exact numeric labels from its config. Loading the
# raw snapshot with an absolute path avoids its cwd-dependent local_path fallback
# and any remote access; no chdir or temporary configuration file is needed.
adapter_cfg = read_json("configs/qevasion.json")
adapter_y = tr[adapter_cfg["label_column"]].astype(str).map(adapter_cfg["label_mapping"]).to_numpy()
adapter_yt = te[adapter_cfg["label_column"]].astype(str).map(adapter_cfg["label_mapping"]).to_numpy()
check("adapter numeric labels match raw-data mapping", np.array_equal(adapter_y, y) and np.array_equal(adapter_yt, yt))
numeric_fit, numeric_val = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=adapter_y, random_state=42)
string_fit, string_val = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=tr.clarity_label.to_numpy(), random_state=42)
split_audit = {
    "actual_02_03_numeric_stratification": split_record(tr, numeric_fit, numeric_val,
                                                      ["title", "url", "question", "interview_answer"]),
    "raw_string_stratification_counterexample": split_record(tr, string_fit, string_val,
                                                             ["title", "url", "question", "interview_answer"]),
    "explanation": "02/03 stratify the adapted numeric labels 0=Clear Reply, 1=Ambivalent, 2=Clear Non-Reply. sklearn sorts class values and consumes RNG class by class. Raw label strings sort in a different order, so an equal random seed produces different row indices. Numeric results 343/220/14 are applicable; raw-string 341/214/10 are not the original split.",
    "validation_coarse_counts": np.bincount(y[numeric_val], minlength=3).tolist()}
log("ACTUAL NUMERIC VALIDATION OVERLAP " + json.dumps(split_audit["actual_02_03_numeric_stratification"]["overlap"]))
log("RAW-STRING COUNTEREXAMPLE OVERLAP " + json.dumps(split_audit["raw_string_stratification_counterexample"]["overlap"]))

qt, qt_parse = read_qnli(BASE / "data" / "qnli_raw" / "train.tsv")
qd, qd_parse = read_qnli(BASE / "data" / "qnli_raw" / "dev.tsv")
qs = qt.groupby("target", group_keys=False).sample(n=6000, random_state=42)
qs = qs.sample(frac=1.0, random_state=42).reset_index(drop=True)
qnli_audit = {"parser": {"train": qt_parse, "dev": qd_parse},
              "full_train": lengths(qt), "sampled_train": lengths(qs), "dev": lengths(qd),
              "sampling": "Map raw labels to numeric 0/1 first; groupby numeric target.sample(n=6000,random_state=42); shuffle sample(frac=1,random_state=42). Matches 04_transfer_qnli.py.",
              "sample_id_order_sha256": canonical_hash(qs["index"].tolist()),
              "sample_dev_exact_overlap": {c: len(set(qs[c]) & set(qd[c])) for c in ["question", "sentence"]},
              "sample_dev_exact_pairs": len(set(zip(qs.question, qs.sentence)) & set(zip(qd.question, qd.sentence))),
              "note": "Split-local QNLI index values may overlap and are not global document IDs. Correct one-decimal sampled lengths are 9.8/26.5, not 9.9/26.4."}
log("QNLI SAMPLED LENGTHS " + json.dumps(qnli_audit["sampled_train"]))

recomputed = {}
predictions = {}
for model, filename, metric_path, key, names, gold in [
    ("tfidf", "predictions_tfidf_lr.csv", "outputs/metrics_baselines.json", "tfidf_lr_test", COARSE, yt),
    ("minilm", "predictions_st_lr.csv", "outputs/metrics_st_lr.json", None, COARSE, yt),
    ("qnli", "predictions_qnli_tfidf_lr.csv", "outputs/metrics_qnli_tfidf_lr.json", None, QNLI_NAMES, qd.target.to_numpy()),
]:
    frame = pd.read_csv(BASE / "outputs" / filename)
    mapping = dict(zip(names, range(len(names))))
    g, p = frame.gold.map(mapping).to_numpy(), frame.prediction.map(mapping).to_numpy()
    check(model + " gold equals raw data", np.array_equal(g, gold))
    raw_eval = qd if model == "qnli" else te
    raw_answer = raw_eval.sentence if model == "qnli" else raw_eval.interview_answer
    check(model + " row IDs/text equal raw data", frame.id.astype(str).tolist() == raw_eval["index"].astype(str).tolist()
          and frame.question.tolist() == raw_eval.question.tolist() and frame.answer.tolist() == raw_answer.tolist())
    check(model + " unique prediction IDs", frame.id.is_unique)
    probabilities = frame[["prob_" + name for name in names]].to_numpy()
    sum_error = float(np.abs(probabilities.sum(axis=1) - 1).max())
    check(model + " probabilities valid and argmax equals saved prediction", np.isfinite(probabilities).all()
          and (probabilities >= 0).all() and (probabilities <= 1).all()
          and sum_error < 2e-6 and np.array_equal(probabilities.argmax(axis=1), p), {"max_sum_error": sum_error})
    saved = read_json(metric_path)
    if key:
        saved = saved[key]
    recomputed[model] = compare_metrics(model + " saved metrics", metric(g, p, names), saved)
    predictions[model] = np.asarray(p, dtype=int)
majority = int(np.bincount(y).argmax())
recomputed["majority"] = compare_metrics("majority saved metrics", metric(yt, np.full(len(yt), majority), COARSE), read_json("outputs/metrics_baselines.json")["majority"])
h = pd.read_csv(BASE / "outputs" / "hierarchy" / "predictions_hierarchy.csv")
hm = read_json("outputs/hierarchy/metrics_hierarchy.json")
check("hierarchy raw ID/gold match", h.id.astype(str).tolist() == te["index"].astype(str).tolist() and np.array_equal(h.gold, yt))
check("hierarchy direct prediction equals main MiniLM", np.array_equal(h.direct, predictions["minilm"]))
for column, saved_key in [("direct", "official_direct"), ("hierarchical", "official_hierarchy")]:
    recomputed["hierarchy_" + column] = compare_metrics("hierarchy " + column + " saved metrics", metric(yt, h[column], COARSE_RAW), hm[saved_key])
probs = h[["prob_clear", "prob_ambivalent", "prob_nonreply"]].to_numpy()
check("hierarchy probability argmax", np.array_equal(probs.argmax(axis=1), h.hierarchical) and np.allclose(probs.sum(1), 1))
fit_title, val_title = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(tr, y, groups=tr.title))
vh = pd.read_csv(BASE / "outputs" / "hierarchy" / "validation_hierarchy.csv")
check("saved hierarchy validation row order", np.array_equal(vh.row, val_title))
check("saved hierarchy validation labels", np.array_equal(vh.gold_coarse, y[val_title]) and np.array_equal(vh.gold_fine, fine[val_title]))
for column, labels, target, saved_key in [("direct", COARSE_RAW, y[val_title], "group_validation_direct"),
                                           ("hierarchical", COARSE_RAW, y[val_title], "group_validation_hierarchy"),
                                           ("pred_fine", FINE, fine[val_title], "group_validation_fine")]:
    recomputed["validation_" + column] = compare_metrics("validation " + column + " saved metrics", metric(target, vh[column], labels), hm[saved_key])
original_title_split = split_record(tr, fit_title, val_title, ["title", "url", "question", "interview_answer"])
direct = pd.read_csv(BASE / "outputs" / "hierarchy" / "predictions_direct_transfer.csv")
dm = read_json("outputs/hierarchy/metrics_direct_transfer.json")
check("direct transfer raw ID/gold match", direct.id.astype(str).tolist() == qd["index"].astype(str).tolist() and np.array_equal(direct.gold, qd.target))
recomputed["direct_transfer"] = compare_metrics("direct transfer saved metrics", metric(direct.gold, direct.prediction, ["0", "1"]), dm["source_only_projection"])
tie_class = int(np.bincount(qs.target).argmax())
recomputed["target_train_tie_baseline"] = compare_metrics("target fixed tie baseline saved metrics", metric(qd.target, np.full(len(qd), tie_class), ["0", "1"]), dm["target_train_tie_majority"])
log("PREDICTION/METRIC AUDIT all current checks=" + str(all(c["passed"] for c in CHECKS)))

cache = np.load(BASE / "outputs" / "st_embeddings.npz")
X, T = cache["X_train"], cache["X_test"]
check("cached labels equal raw labels", np.array_equal(cache["y_train"], y) and np.array_equal(cache["y_test"], yt))
check("cached features finite and aligned in row counts", np.isfinite(X).all() and np.isfinite(T).all() and len(X) == len(tr) and len(T) == len(te))
refit_start = time.perf_counter()
refit = LogisticRegression(C=.1, max_iter=2000, class_weight="balanced", random_state=42).fit(X, y)
cached_prediction = refit.predict(T)
cached_elapsed = time.perf_counter() - refit_start
check("cached MiniLM LR refit prediction exact", np.array_equal(cached_prediction, predictions["minilm"]))
cache_audit = {"shapes": {name: list(cache[name].shape) for name in cache.files},
               "refit_config": {"C": .1, "max_iter": 2000, "class_weight": "balanced", "random_state": 42},
               "refit_elapsed_seconds": cached_elapsed,
               "refit_prediction_mismatches": int((cached_prediction != predictions["minilm"]).sum()),
               "meaning": "Fresh LR fit and prediction on the existing frozen feature cache, not a new sentence encoding or complete end-to-end rerun."}
for path, expected in hm["hashes"].items():
    normalized = path.replace("\\", "/")
    check("historical cache lineage " + normalized, sha256(BASE / normalized) == expected)
log(f"CACHE LR REFIT {cached_elapsed:.6f} seconds; prediction mismatches={cache_audit['refit_prediction_mismatches']}")

# One post-hoc URL-group holdout. It is intentionally not used to choose a model.
hold_start = time.perf_counter()
fit_url, val_url = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups=tr.url))
holdout = {"purpose": "Post-hoc source-group sensitivity; no model selection or official-test retuning",
           "method": "GroupShuffleSplit by exact source URL, test_size=0.2, random_state=42",
           "fixed_C": .1, "split": split_record(tr, fit_url, val_url, ["url", "title", "question", "interview_answer"]),
           "fit_source_urls": int(tr.iloc[fit_url].url.nunique()),
           "validation_source_urls": int(tr.iloc[val_url].url.nunique()), "variants": []}
check("new URL-group holdout has zero URL overlap", holdout["split"]["overlap"]["url"]["shared_unique_values"] == 0)
for label_space, class_weight, aggregations in [("coarse", "balanced", ["direct"]),
                                                ("fine", "balanced", ["sum", "child_count_mean"]),
                                                ("fine", None, ["sum"])]:
    fit_clock = time.perf_counter()
    target = y if label_space == "coarse" else fine
    model = LogisticRegression(C=.1, max_iter=2000, class_weight=class_weight, random_state=42).fit(X[fit_url], target[fit_url])
    fit_seconds = time.perf_counter() - fit_clock
    if label_space == "fine":
        fine_p = model.predict_proba(X[val_url])
        sums = np.zeros((len(val_url), 3), dtype=float)
        for column, fine_id in enumerate(model.classes_):
            sums[:, FINE_TO_COARSE[int(fine_id)]] += fine_p[:, column]
    for aggregation in aggregations:
        p = (model.predict(X[val_url]) if aggregation == "direct" else
             (sums / np.array([1, 5, 3]) if aggregation == "child_count_mean" else sums).argmax(1))
        name = "direct_balanced" if aggregation == "direct" else ("balanced" if class_weight else "unweighted") + "_" + aggregation
        entry = {"name": name, "class_weight": class_weight, "aggregation": aggregation,
                 "fit_elapsed_seconds": fit_seconds, "validation": metric(y[val_url], p, COARSE),
                 "score_note": "Heuristic decision score, not a normalized coarse probability" if aggregation == "child_count_mean" else "Standard classifier decision"}
        holdout["variants"].append(entry)
        log(f"URL HOLDOUT {name} n={len(val_url)} acc={entry['validation']['accuracy']:.6f} macro_f1={entry['validation']['macro_f1']:.6f} predicted={entry['validation']['predicted_counts']}")
holdout["elapsed_seconds"] = time.perf_counter() - hold_start

# Pair all models on the same sampled URLs, retaining every question in a URL.
bootstrap_start = time.perf_counter()
codes, unique_urls = pd.factorize(te.url, sort=False)
group_count = len(unique_urls)
rng = np.random.default_rng(args.seed)
all_preds = [predictions["tfidf"], predictions["minilm"], h.hierarchical.to_numpy()]
encoded = [yt * 3 + pred for pred in all_preds]
differences = np.empty((args.repetitions, 2), dtype=float)
resampled_n = np.empty(args.repetitions, dtype=int)
for iteration in range(args.repetitions):
    cluster_counts = np.bincount(rng.integers(0, group_count, size=group_count), minlength=group_count)
    weights = cluster_counts[codes]
    scores = []
    for label_pairs in encoded:
        cm = np.bincount(label_pairs, weights=weights, minlength=9).reshape(3, 3)
        den = cm.sum(1) + cm.sum(0)
        scores.append(float(np.divide(2 * cm.diagonal(), den, out=np.zeros(3), where=den > 0).mean()))
    differences[iteration] = [scores[1] - scores[0], scores[2] - scores[1]]
    resampled_n[iteration] = weights.sum()
bootstrap = {"metric": "Three-class Macro-F1, fixed labels 0/1/2, zero_division=0",
             "unit": "Exact raw QEvasion source URL; sampled with replacement, retain all rows per URL",
             "url_group_order": "first appearance in original official test row order (pandas.factorize sort=False)",
             "test_rows": len(te), "url_clusters": group_count, "repetitions": args.repetitions,
             "seed": args.seed, "rng": "numpy.random.default_rng / Generator(PCG64)",
             "confidence_interval": "2.5th and 97.5th percentiles, numpy.quantile method=linear",
             "elapsed_seconds": time.perf_counter() - bootstrap_start,
             "resampled_rows_min_max": [int(resampled_n.min()), int(resampled_n.max())],
             "comparisons": {},
             "limitation": "Conditional uncertainty for this saved single-run model on these source clusters. Does not measure training-seed variability, remove train/test source overlap, or establish population-level generalization."}
for col, name, positive, negative in [(0, "minilm_minus_tfidf", "minilm", "tfidf"),
                                        (1, "hierarchy_minus_minilm", "hierarchy_hierarchical", "minilm")]:
    bootstrap["comparisons"][name] = {"point_difference": recomputed[positive]["recomputed"]["macro_f1"] - recomputed[negative]["recomputed"]["macro_f1"],
                                     "ci95": np.quantile(differences[:, col], [.025, .975], method="linear").tolist()}
    log("URL CLUSTER BOOTSTRAP " + name + " " + json.dumps(bootstrap["comparisons"][name]))

source_subgroups = {}
for name, mask in [("url_seen_in_train", seen_url), ("url_unseen_in_train", ~seen_url)]:
    source_subgroups[name] = {"n": int(mask.sum()), "source_urls": int(te.loc[mask, "url"].nunique()),
                             "models": {model: metric(yt[mask], pred[mask], COARSE) for model, pred in
                                        [("tfidf", predictions["tfidf"]), ("minilm", predictions["minilm"]), ("hierarchy", all_preds[2])]}}

final_hashes = {name: sha256(BASE / name) for name in input_files}
check("all original inputs unchanged during review", initial_hashes == final_hashes)
elapsed = time.perf_counter() - START
result = {"schema_version": 1, "created_at_asia_shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
          "purpose": "Independent read-only experimental audit plus post-hoc, non-selection sensitivity checks",
          "base": str(BASE), "environment": environment, "total_elapsed_seconds": elapsed,
          "source_audit": source_audit, "original_validation_split_audit": split_audit,
          "original_hierarchy_title_split_audit": original_title_split, "qnli_audit": qnli_audit,
          "prediction_metric_recalculation": recomputed, "cache_lr_reproduction": cache_audit,
          "new_url_group_holdout": holdout, "url_cluster_bootstrap": bootstrap,
          "descriptive_official_test_source_subgroups": source_subgroups,
          "checks": CHECKS, "all_checks_passed": all(c["passed"] for c in CHECKS),
          "input_sha256": initial_hashes, "script_sha256": sha256(Path(__file__))}
log(f"REVIEW COMPLETE: checks={len(CHECKS)}, passed={sum(c['passed'] for c in CHECKS)}, elapsed={elapsed:.6f}s")
log("No existing metrics, predictions, data, or embeddings were modified. No sentence embeddings were recomputed.")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "review_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "review_validation.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("JSON: " + str(OUT / "review_validation.json"))
print("LOG: " + str(OUT / "review_validation.log"))
if not result["all_checks_passed"]:
    raise SystemExit(2)
