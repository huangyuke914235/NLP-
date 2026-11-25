"""Fit and evaluate the course baselines, transfer and controlled extensions."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
METADATA_COLUMNS = ["id", "group", "answer_length", "question_length"]
FINE_TO_COARSE = {
    "Explicit": "Clear Reply",
    "Dodging": "Ambivalent",
    "Implicit": "Ambivalent",
    "General": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_data(path, labels):
    frame = pd.read_csv(path, keep_default_na=False, dtype={"id": str, "label": str})
    required = {"question", "answer", "label", "split", *METADATA_COLUMNS}
    if required - set(frame) or frame["id"].duplicated().any():
        raise ValueError(f"Malformed input or duplicate IDs in {path}")
    if set(frame["label"]) != set(labels):
        raise ValueError(f"Unexpected label set in {path}: {set(frame['label'])}")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError(f"Unexpected split set in {path}: {set(frame['split'])}")
    return frame


def index_split(frame, name):
    return np.flatnonzero(frame["split"].to_numpy() == name)


def score(y, pred, labels):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0
    )
    return {
        "labels": list(labels),
        "count": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def ordered_probabilities(model, x, labels):
    positions = {label: index for index, label in enumerate(model.classes_)}
    probabilities = model.predict_proba(x)
    return np.column_stack(
        [
            probabilities[:, positions[label]] if label in positions else np.zeros(len(x))
            for label in labels
        ]
    )


def predicted(probabilities, labels):
    return np.asarray(labels)[probabilities.argmax(axis=1)]


def aggregate(probabilities, source_labels, target_labels, mapping):
    return np.column_stack(
        [
            probabilities[:, [mapping[s] == t for s in source_labels]].sum(axis=1)
            for t in target_labels
        ]
    )


class Experiment:
    def __init__(self, output, config):
        self.output = output
        (output / "predictions").mkdir(parents=True, exist_ok=True)
        self.config = config
        self.started = perf_counter()
        self.metrics = {}
        self.selection = {}

    def log(self, text):
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} [{perf_counter() - self.started:8.2f}s] {text}"
        print(line, flush=True)
        with (self.output / "run.log").open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def evaluate(self, name, frame, y, probabilities, labels, parameters, split="test"):
        predictions = predicted(probabilities, labels)
        record = score(y, predictions, labels)
        result = frame[METADATA_COLUMNS].reset_index(drop=True).copy()
        result["gold"] = np.asarray(y)
        result["prediction"] = predictions
        for i, label in enumerate(labels):
            result[f"prob_{label}"] = probabilities[:, i]
        result = result[["id", "gold", "prediction", "group", "answer_length", "question_length", *[f"prob_{label}" for label in labels]]]
        relative_path = f"predictions/{name}.csv"
        result.to_csv(self.output / relative_path, index=False, encoding="utf-8", float_format="%.12g")
        record.update({"prediction_file": relative_path, "parameters": parameters, "split": split})
        self.metrics[name] = record
        write_json(self.output / "metrics.json", self.metrics)
        self.log(f"{name} n={len(y)} accuracy={record['accuracy']:.6f} macro_f1={record['macro_f1']:.6f}")
        return record

    def add_selection(self, name, candidates, selected):
        self.selection[name] = {"candidates": candidates, "selected": selected}
        write_json(self.output / "selection.json", self.selection)

    def logistic(self, x, y, c, weight=None):
        model = LogisticRegression(
            C=c,
            class_weight=weight,
            solver="lbfgs",
            max_iter=self.config["max_lr_iterations"],
            random_state=self.config["seed"],
            tol=1e-5,
        )
        model.fit(x, y)
        return model

    def tune_lr(self, name, x, y, train, validation, labels, weight=None):
        candidates = []
        for c in self.config["candidates_c"]:
            model = self.logistic(x[train], y[train], c, weight)
            p = ordered_probabilities(model, x[validation], labels)
            candidates.append({
                "c": c,
                "class_weight": weight,
                "validation_macro_f1": f1_score(y[validation], predicted(p, labels), labels=labels, average="macro", zero_division=0),
                "iterations": model.n_iter_.tolist(),
            })
        best = max(candidates, key=lambda row: row["validation_macro_f1"])
        self.add_selection(name, candidates, best)
        return best


def pair_text(frame):
    return (frame["question"] + " [SEP] " + frame["answer"]).to_numpy()


def run_sparse(exp, name, frame, labels, ablations=True):
    y = frame["label"].to_numpy()
    train, validation, test = (index_split(frame, split) for split in ("train", "validation", "test"))
    majority = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), y[train])
    exp.evaluate(
        f"{name}_majority", frame.iloc[test], y[test],
        ordered_probabilities(majority, np.zeros((len(test), 1)), labels),
        labels, {"strategy": "prior"},
    )
    corpus = pair_text(frame)
    candidates = []
    vectors = {}
    for top in (1, 2):
        vectorizer = TfidfVectorizer(
            ngram_range=(1, top), min_df=exp.config["tfidf_min_document_frequency"],
            max_features=exp.config["tfidf_max_features"], sublinear_tf=True,
            token_pattern=r"(?u)\[sep\]|\b\w\w+\b",
        ).fit(corpus[train])
        x = vectorizer.transform(corpus)
        vectors[top] = (vectorizer, x)
        for c, weight in itertools.product(exp.config["candidates_c"], (None, "balanced")):
            model = exp.logistic(x[train], y[train], c, weight)
            p = ordered_probabilities(model, x[validation], labels)
            candidates.append({
                "ngram_top": top, "c": c, "class_weight": weight,
                "validation_macro_f1": f1_score(y[validation], predicted(p, labels), labels=labels, average="macro", zero_division=0),
                "iterations": model.n_iter_.tolist(),
                "feature_count": x.shape[1],
            })
    best = max(candidates, key=lambda row: row["validation_macro_f1"])
    exp.add_selection(f"{name}_tfidf", candidates, best)
    chosen = [("tfidf", best["ngram_top"], best["class_weight"])]
    if ablations:
        chosen.extend([
            ("tfidf_alternate_ngram", 3 - best["ngram_top"], best["class_weight"]),
            ("tfidf_alternate_weight", best["ngram_top"], None if best["class_weight"] else "balanced"),
        ])
    primary = None
    for suffix, top, weight in chosen:
        vectorizer, x = vectors[top]
        model = exp.logistic(x[train], y[train], best["c"], weight)
        candidate = next(row for row in candidates if (row["ngram_top"], row["c"], row["class_weight"]) == (top, best["c"], weight))
        exp.evaluate(
            f"{name}_{suffix}", frame.iloc[test], y[test],
            ordered_probabilities(model, x[test], labels), labels, candidate,
        )
        if suffix == "tfidf":
            primary = vectorizer, model
    return primary


class FrozenEncoder:
    def __init__(self, exp, device):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.exp = exp
        self.config = exp.config
        torch.manual_seed(self.config["seed"])
        torch.set_num_threads(self.config["threads"])
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        name = self.config["minilm_model"]
        revision = self.config["minilm_revision"]
        try:
            model_file = hf_hub_download(name, "model.safetensors", revision=revision, local_files_only=True)
            local = True
        except FileNotFoundError:
            model_file = hf_hub_download(name, "model.safetensors", revision=revision)
            local = False
        with Path(model_file).open("rb") as weights:
            self.model_hash = hashlib.file_digest(weights, "sha256").hexdigest()
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=local)
        self.model = AutoModel.from_pretrained(name, revision=revision, local_files_only=local, attn_implementation="eager").to(self.device).eval()
        self.memory = {}
        exp.log(f"Encoder {name}@{revision}, torch={torch.__version__}, device={self.device}, sha256={self.model_hash}")

    def pairs(self, frame):
        return self.encode(frame["question"], frame["answer"])

    def encode(self, questions, answers):
        torch = self.torch
        questions, answers = list(questions), list(answers)
        missing = sorted(set(questions + answers).difference(self.memory), key=lambda t: (len(t), t))
        size = self.config["minilm_batch_size"] if self.device == "cuda" else min(32, self.config["minilm_batch_size"])
        max_tokens = self.config["minilm_max_tokens_per_text"]
        self.exp.log(f"Encoding {len(missing)} unseen texts; batch={size}, max_wordpieces={max_tokens}")
        with torch.inference_mode():
            for i in range(0, len(missing), size):
                batch = missing[i:i + size]
                tokens = self.tokenizer(batch, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                states = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
                self.memory.update(zip(batch, vectors))
                if i and i % (size * 100) == 0:
                    self.exp.log(f"Encoder progress {i}/{len(missing)}")
        return np.stack([self.memory[t] for t in questions]), np.stack([self.memory[t] for t in answers])


def pair_features(embeddings, interactions):
    question, answer = embeddings
    parts = [question, answer]
    if interactions:
        parts.extend([np.abs(question - answer), question * answer])
    return np.concatenate(parts, axis=1)


def run_dense(exp, name, frame, labels, embeddings):
    y = frame["label"].to_numpy()
    train, validation, test = (index_split(frame, split) for split in ("train", "validation", "test"))
    base_x = pair_features(embeddings, False)
    selection = exp.tune_lr(f"{name}_minilm_c", base_x, y, train, validation, labels)
    candidates = []
    for interactions in (False, True):
        suffix = "minilm_interactions" if interactions else "minilm"
        x = pair_features(embeddings, True) if interactions else base_x
        model = exp.logistic(x[train], y[train], selection["c"])
        p_val = ordered_probabilities(model, x[validation], labels)
        params = {
            "c": selection["c"],
            "class_weight": None,
            "interactions": interactions,
            "feature_count": x.shape[1],
            "iterations": model.n_iter_.tolist(),
            "validation_macro_f1": f1_score(y[validation], predicted(p_val, labels), labels=labels, average="macro", zero_division=0),
        }
        exp.evaluate(f"{name}_{suffix}", frame.iloc[test], y[test],
                     ordered_probabilities(model, x[test], labels), labels, params)
        candidates.append((suffix, model, params))
    best = max(candidates, key=lambda row: row[2]["validation_macro_f1"])
    exp.add_selection(f"{name}_minilm_representation", [{"variant": s, **p} for s, _, p in candidates], {"variant": best[0], **best[2]})
    return best, candidates


def run_hierarchy(exp, frame, labels, embeddings, source):
    _, _, selected = source
    x = pair_features(embeddings, selected["interactions"])
    cleaned = frame["fine_label"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    lookup = {label.casefold(): label for label in FINE_TO_COARSE}
    fine = cleaned.str.casefold().map(lookup).fillna("").to_numpy()
    coarse = frame["label"].to_numpy()
    fine_labels = list(FINE_TO_COARSE)
    have_fine = fine != ""
    train = index_split(frame, "train")
    validation = index_split(frame, "validation")
    test = index_split(frame, "test")
    fine_train = train[have_fine[train]]
    fine_validation = validation[have_fine[validation]]
    if not fine_train.size or not fine_validation.size:
        raise ValueError(f"Fine-label input not usable: {cleaned.value_counts().to_dict()}")
    fine_model = exp.logistic(x[fine_train], fine[fine_train], selected["c"])
    coarse_model = exp.logistic(x[fine_train], coarse[fine_train], selected["c"])
    p_fine = ordered_probabilities(fine_model, x[test], fine_labels)
    p_coarse = ordered_probabilities(coarse_model, x[test], labels)
    p_aggregated = aggregate(p_fine, fine_labels, labels, FINE_TO_COARSE)
    parameters = {
        "c": selected["c"], "class_weight": None, "interactions": selected["interactions"],
        "training_count": len(fine_train), "fine_labels": fine_labels,
    }
    exp.evaluate("hierarchy_direct_coarse", frame.iloc[test], coarse[test], p_coarse, labels, parameters)
    exp.evaluate("hierarchy_fine_to_coarse", frame.iloc[test], coarse[test], p_aggregated, labels, parameters)
    exp.evaluate(
        "hierarchy_fine", frame.iloc[fine_validation], fine[fine_validation],
        ordered_probabilities(fine_model, x[fine_validation], fine_labels),
        fine_labels, parameters, split="validation",
    )
    mismatch = [
        {"id": sample_id, "coarse": c, "fine": f, "mapped": FINE_TO_COARSE[f]}
        for sample_id, c, f in zip(frame["id"], coarse, fine)
        if f and FINE_TO_COARSE[f] != c
    ]
    write_json(exp.output / "hierarchy.json", {
        "taxonomy": FINE_TO_COARSE, "training_fine_count": len(fine_train),
        "validation_fine_count": len(fine_validation), "test_fine_count": int(have_fine[test].sum()),
        "fine_raw_counts_by_split": {
            split: dict(Counter(cleaned.iloc[index_split(frame, split)]))
            for split in ("train", "validation", "test")
        },
        "coarse_fine_disagreements": mismatch,
    })


def stratified_nested(y, train, seed, fractions):
    rng = np.random.default_rng(seed)
    shuffled = [rng.permutation(train[y[train] == label]) for label in sorted(set(y[train]))]
    for fraction in fractions:
        yield fraction, np.sort(np.concatenate([
            group[:max(1, round(fraction * len(group)))] for group in shuffled
        ]))


def run_transfer(exp, frame, labels, source_labels, embeddings, source):
    source_name, source_model, source_parameters = source
    x = pair_features(embeddings, source_parameters["interactions"])
    train, validation, test = (index_split(frame, split) for split in ("train", "validation", "test"))
    y = frame["label"].to_numpy()
    mapping = exp.config["transfer_label_mapping"]
    p_source = ordered_probabilities(source_model, x, source_labels)
    p_direct = aggregate(p_source, source_labels, labels, mapping)
    exp.evaluate("target_direct", frame.iloc[test], y[test], p_direct[test], labels, {
        "source_variant": source_name, "label_mapping": mapping,
        "target_labels_used_for_fitting_or_selection": False,
    })
    raw_log_odds = np.log(np.clip(p_direct[:, 0], 1e-8, 1.0)) - np.log(np.clip(p_direct[:, 1], 1e-8, 1.0))
    full_sel = exp.tune_lr("target_minilm_c", x, y, train, validation, labels)
    rows = []
    draw_ids = []
    full_result = {}
    for seed in exp.config["subset_seeds"]:
        for fraction, selected in stratified_nested(y, train, seed, exp.config["fractions"]):
            key = f"{round(100 * fraction):03d}_seed{seed}"
            draw_ids.extend({"seed": seed, "fraction": fraction, "id": frame.iloc[idx]["id"]} for idx in selected)
            if fraction == 1.0 and full_result:
                rows.extend({**record, "seed": seed} for record in full_result.values())
                continue
            tuning = full_sel if fraction == 1.0 else exp.tune_lr(f"efficiency_{key}_c", x, y, selected, validation, labels)
            train_odds = raw_log_odds[selected]
            mean, scale = float(train_odds.mean()), max(float(train_odds.std()), 1e-8)
            x_adapted = np.column_stack([x, (raw_log_odds - mean) / scale])
            for suffix, features in (("minilm", x), ("adapted", x_adapted)):
                model = exp.logistic(features[selected], y[selected], tuning["c"])
                p = ordered_probabilities(model, features[test], labels)
                val_p = ordered_probabilities(model, features[validation], labels)
                fit_params = {
                    "c": tuning["c"], "class_weight": None, "feature_count": features.shape[1],
                    "train_n": len(selected), "variance_seed": seed, "iterations": model.n_iter_.tolist(),
                    "source_variant": source_name if suffix == "adapted" else None,
                    "validation_macro_f1": f1_score(y[validation], predicted(val_p, labels), labels=labels, average="macro", zero_division=0),
                }
                if suffix == "adapted":
                    fit_params.update({
                        "source_log_odds_training_mean": mean, "source_log_odds_training_sd": scale,
                        "source_log_odds_coefficient_for_positive_class": float(model.coef_[0, -1]),
                        "positive_class": model.classes_[1],
                    })
                name = f"target_{suffix}" if fraction == 1.0 else f"efficiency_{suffix}_{key}"
                result = exp.evaluate(name, frame.iloc[test], y[test], p, labels, fit_params)
                row = {
                    "method": suffix, "fraction": fraction, "seed": seed, "train_n": len(selected),
                    "train_label_counts": dict(Counter(y[selected])),
                    "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
                    "validation_macro_f1": fit_params["validation_macro_f1"], "c": tuning["c"],
                    "prediction_file": result["prediction_file"],
                }
                rows.append(row)
                if fraction == 1.0:
                    full_result[suffix] = row
    summaries = []
    for fraction, method in itertools.product(exp.config["fractions"], ("minilm", "adapted")):
        selected_rows = [row for row in rows if row["fraction"] == fraction and row["method"] == method]
        values = [row["macro_f1"] for row in selected_rows]
        summaries.append({
            "fraction": fraction, "method": method, "mean_macro_f1": statistics.mean(values),
            "sd_macro_f1": statistics.stdev(values), "repeats": len(values), "train_n": selected_rows[0]["train_n"],
            "unique_training_draws": 1 if fraction == 1.0 else len(values),
        })
    write_json(exp.output / "data_efficiency.json", {"runs": rows, "summary": summaries})
    pd.DataFrame(draw_ids).to_csv(exp.output / "data_efficiency_train_ids.csv", index=False)


def run_counterfactual(exp, path, frame, labels, tfidf, dense, encoder):
    pairs = pd.read_csv(path, keep_default_na=False, dtype=str)
    fields = ["id", "paraphrase", "unrelated", "unrelated_expected", "paraphrase_rationale", "unrelated_rationale"]
    if set(fields).difference(pairs) or pairs[fields].apply(lambda s: s.str.strip().eq("")).any().any() or pairs["id"].duplicated().any():
        raise ValueError(f"Missing fields, duplicated IDs or blank values in {path}")
    if not set(pairs["unrelated_expected"]).issubset(labels):
        raise ValueError(f"Unexpected manually assigned counterfactual label in {path}")
    test = frame[frame["split"].eq("test")].set_index("id")
    original = test.loc[pairs["id"]].reset_index()
    if (pairs["paraphrase"] == original["question"]).any() or (pairs["unrelated"] == original["question"]).any() or (pairs["paraphrase"] == pairs["unrelated"]).any():
        raise ValueError("Counterfactual questions must be distinct")
    other = []
    for variant in ("paraphrase", "unrelated"):
        copy = original.copy()
        copy["question"] = pairs[variant]
        other.append(copy)
    combined = pd.concat([original, *other], ignore_index=True)
    embeddings = encoder.pairs(combined)
    vectorizer, model = tfidf
    outputs = {"grouped_tfidf": ordered_probabilities(model, vectorizer.transform(pair_text(combined)), labels)}
    for name, model, settings in dense:
        x = pair_features(embeddings, settings["interactions"])
        outputs[f"grouped_{name}"] = ordered_probabilities(model, x, labels)
    records, summaries = [], {}
    for model_name, probabilities in outputs.items():
        n = len(pairs)
        call = predicted(probabilities, labels)
        original_call, para_call, mismatch_call = (call[i * n:(i + 1) * n] for i in range(3))
        gold = original["label"].to_numpy()
        expected = pairs["unrelated_expected"].to_numpy()
        should_change = gold != expected
        original_correct = original_call == gold
        para_stable = original_call == para_call
        mismatch_good = mismatch_call == expected
        label_changed = mismatch_call != original_call
        summaries[model_name] = {
            "n": n,
            "original_correct_n": int(original_correct.sum()),
            "paraphrase_stable_n": int(para_stable.sum()),
            "paraphrase_gold_correct_n": int((para_call == gold).sum()),
            "unrelated_expected_label_correct_n": int(mismatch_good.sum()),
            "unrelated_gold_requires_change_n": int(should_change.sum()),
            "unrelated_changed_on_required_n": int((label_changed & should_change).sum()),
            "unrelated_correct_and_changed_on_required_n": int((mismatch_good & label_changed & should_change).sum()),
            "unrelated_required_and_original_correct_n": int((should_change & original_correct).sum()),
            "unrelated_correct_and_changed_on_required_and_original_correct_n": int((mismatch_good & label_changed & should_change & original_correct).sum()),
        }
        for i, (_, sample) in enumerate(pairs.iterrows()):
            row = sample.to_dict()
            row.update({
                "model": model_name, "original_question": original.iloc[i]["question"],
                "fixed_answer": original.iloc[i]["answer"], "gold": gold[i], "original_prediction": original_call[i],
                "paraphrase_prediction": para_call[i], "unrelated_prediction": mismatch_call[i],
                "gold_requires_change": bool(should_change[i]),
            })
            for j, variant in enumerate(("original", "paraphrase", "unrelated")):
                row.update({f"{variant}_prob_{label}": float(probabilities[j * n + i, k]) for k, label in enumerate(labels)})
            records.append(row)
    write_json(exp.output / "counterfactual.json", summaries)
    pd.DataFrame(records).to_csv(exp.output / "counterfactual.csv", index=False, float_format="%.12g")
    exp.log(f"Counterfactual fixed-answer audit: {len(pairs)} examples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--counterfactual", type=Path, default=ROOT / "configs" / "counterfactual.csv")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    conf = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {args.output}")
    exp = Experiment(args.output, conf)
    random.seed(conf["seed"])
    np.random.seed(conf["seed"])
    write_json(args.output / "configuration.json", conf)
    metadata = {
        "python": sys.version, "platform": platform.platform(), "executable": sys.executable,
        "command": [sys.executable, *sys.argv], "started": datetime.now().astimezone().isoformat(),
        "versions": {name: version(name) for name in ("torch", "numpy", "pandas", "transformers", "scikit-learn", "scipy", "huggingface-hub")},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "input_sha256": {},
    }
    write_json(args.output / "environment.json", metadata)
    exp.log(f"Started, Python={sys.version.split()[0]}, config_seed={conf['seed']}")
    source_labels = conf["source_labels"]
    target_labels = conf["target_labels"]
    with threadpool_limits(limits=conf["threads"]):
        encoder = FrozenEncoder(exp, args.device)
        metadata.update({
            "device": encoder.device, "weights_sha256": encoder.model_hash,
            "cuda_build": encoder.torch.version.cuda,
            "gpu_name": encoder.torch.cuda.get_device_name(0) if encoder.device == "cuda" else "",
        })
        for name in ("grouped", "official"):
            source_path = args.data / f"qevasion_{name}.csv"
            metadata["input_sha256"][source_path.name] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            source = read_data(source_path, source_labels)
            exp.log(f"Read {name}: {source['split'].value_counts().to_dict()}, labels={source['label'].value_counts().to_dict()}")
            sparse = run_sparse(exp, name, source, source_labels)
            embeddings = encoder.pairs(source)
            best, dense = run_dense(exp, name, source, source_labels, embeddings)
            if name == "grouped":
                grouped_best = best
                run_hierarchy(exp, source, source_labels, embeddings, grouped_best)
                if args.counterfactual.is_file():
                    metadata["input_sha256"][args.counterfactual.name] = hashlib.sha256(args.counterfactual.read_bytes()).hexdigest()
                    run_counterfactual(exp, args.counterfactual, source, source_labels, sparse, dense, encoder)
                else:
                    exp.log(f"Counterfactual input absent: {args.counterfactual}")

        target_path = args.data / "qnli.csv"
        metadata["input_sha256"][target_path.name] = hashlib.sha256(target_path.read_bytes()).hexdigest()
        target = read_data(target_path, target_labels)
        exp.log(f"Read QNLI: {target['split'].value_counts().to_dict()}, labels={target['label'].value_counts().to_dict()}")
        run_sparse(exp, "target", target, target_labels, ablations=False)
        run_transfer(exp, target, target_labels, source_labels, encoder.pairs(target), grouped_best)
    metadata.update({"finished": datetime.now().astimezone().isoformat(), "elapsed_seconds": round(perf_counter() - exp.started, 3)})
    write_json(args.output / "environment.json", metadata)
    exp.log("Completed")


if __name__ == "__main__":
    main()
