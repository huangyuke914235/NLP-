"""Audit paired fixed-test model differences by resampling whole source groups."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {
    "qevasion": {
        "data": "data/processed/qevasion_grouped.csv",
        "labels": ("Clear Reply", "Ambivalent", "Clear Non-Reply"),
        "group_unit": "interview URL",
    },
    "qnli": {
        "data": "data/processed/qnli.csv",
        "labels": ("entailment", "not_entailment"),
        "group_unit": "Wikipedia article with preprocessing grouping key",
    },
}

RUN = "results/run_1/predictions/"
PAIR = "results/pair_run_1/predictions/"
COMPARISONS = (
    ("qevasion", "sparse_vs_frozen", RUN+"grouped_tfidf.csv", RUN+"grouped_minilm.csv"),
    ("qevasion", "sparse_vs_interactions", RUN+"grouped_tfidf.csv", RUN+"grouped_minilm_interactions.csv"),
    ("qevasion", "sparse_vs_joint", RUN+"grouped_tfidf.csv", PAIR+"grouped_pair.csv"),
    ("qevasion", "frozen_vs_interactions", RUN+"grouped_minilm.csv", RUN+"grouped_minilm_interactions.csv"),
    ("qevasion", "hierarchy_direct_vs_two_stage", RUN+"hierarchy_direct_coarse.csv", RUN+"hierarchy_fine_to_coarse.csv"),
    ("qnli", "direct_vs_target_frozen", RUN+"target_direct.csv", RUN+"target_minilm.csv"),
    ("qnli", "target_frozen_vs_adapted", RUN+"target_minilm.csv", RUN+"target_adapted.csv"),
    ("qnli", "target_sparse_vs_joint", RUN+"target_tfidf.csv", PAIR+"target_pair.csv"),
)


def csv(path):
    return pd.read_csv(path, keep_default_na=False, dtype={"id": str}).set_index("id", verify_integrity=True)


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def scores(matrix):
    """Return accuracy, Macro-F1 and recall per fixed label over the last axes."""
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    support = matrix.sum(axis=-1)
    precision_denominator = matrix.sum(axis=-2)
    accuracy = ratio(diagonal.sum(axis=-1), support.sum(axis=-1))
    macro_f1 = ratio(2 * diagonal, support + precision_denominator).mean(axis=-1)
    recall = ratio(diagonal, support)
    return np.concatenate((accuracy[..., None], macro_f1[..., None], recall), axis=-1)


class GroupedEvaluation:
    def __init__(self, settings, seed, replications):
        self.dataset = ROOT / settings["data"]
        self.labels = settings["labels"]
        self.label_code = {label: i for i, label in enumerate(self.labels)}
        frame = csv(self.dataset)
        self.test = frame.loc[frame["split"].eq("test"), ["label", "group"]]
        if self.test.empty or self.test["group"].eq("").any():
            raise ValueError(f"Missing test data or test group labels: {self.dataset}")
        self.gold = self.encode(self.test["label"])
        groups, self.cluster_code = np.unique(self.test["group"], return_inverse=True)
        self.group_unit = settings["group_unit"]
        self.group_count = len(groups)
        weights = np.full(self.group_count, 1 / self.group_count)
        self.draws = np.random.default_rng(seed).multinomial(self.group_count, weights, size=replications)

    def encode(self, labels):
        return np.fromiter((self.label_code[label] for label in labels), dtype=np.int64)

    def read_prediction(self, relpath):
        path = ROOT / relpath
        source = csv(path)
        if set(source.index) != set(self.test.index):
            raise ValueError(f"Prediction IDs differ from the fixed test set: {path}")
        aligned = source.reindex(self.test.index)
        if not aligned["gold"].eq(self.test["label"]).all() or not aligned["group"].eq(self.test["group"]).all():
            raise ValueError(f"Gold or group values differ from the fixed test set: {path}")
        return self.encode(aligned["prediction"]), sha256(path.read_bytes()).hexdigest()

    def matrices(self, prediction):
        k = len(self.labels)
        bins = (self.cluster_code * k + self.gold) * k + prediction
        by_cluster = np.bincount(bins, minlength=self.group_count * k * k).reshape(self.group_count, k, k)
        samples = (self.draws @ by_cluster.reshape(self.group_count, -1)).reshape(-1, k, k)
        return by_cluster.sum(0), samples

    def compare(self, baseline, candidate):
        old, old_hash = self.read_prediction(baseline)
        new, new_hash = self.read_prediction(candidate)
        old_total, old_boot = self.matrices(old)
        new_total, new_boot = self.matrices(new)
        old_score, new_score = scores(old_total), scores(new_total)
        delta = scores(new_boot) - scores(old_boot)
        lo, hi = np.quantile(delta, (0.025, 0.975), axis=0)
        metric_names = ["accuracy", "macro_f1", *(f"recall:{label}" for label in self.labels)]
        metrics = {
            name: {"baseline": float(old_score[i]), "candidate": float(new_score[i]),
                   "difference": float(new_score[i] - old_score[i]), "percentile_95_interval": [float(lo[i]), float(hi[i])]}
            for i, name in enumerate(metric_names)
        }
        old_correct, new_correct = old == self.gold, new == self.gold
        by_label = {}
        for label, code in self.label_code.items():
            selected = self.gold == code
            by_label[label] = {"support": int(selected.sum()), "wrong_to_right": int((selected & ~old_correct & new_correct).sum()),
                               "right_to_wrong": int((selected & old_correct & ~new_correct).sum())}
        k = len(self.labels)
        transition = np.bincount(old * k + new, minlength=k * k).reshape(k, k)
        return {
            "baseline_predictions": baseline, "baseline_predictions_sha256": old_hash,
            "candidate_predictions": candidate, "candidate_predictions_sha256": new_hash,
            "n": len(old), "group_unit": self.group_unit, "groups": self.group_count, "label_order": list(self.labels),
            "changed_prediction": int((old != new).sum()), "wrong_to_right": int((~old_correct & new_correct).sum()),
            "right_to_wrong": int((old_correct & ~new_correct).sum()),
            "wrong_to_different_wrong": int((~old_correct & ~new_correct & (old != new)).sum()),
            "transitions_by_gold": by_label, "prediction_transition_matrix_baseline_to_candidate": transition.tolist(),
            "baseline_confusion_matrix": old_total.tolist(), "candidate_confusion_matrix": new_total.tolist(), "metrics": metrics,
        }


def efficiency_effects(audit):
    path = ROOT / "results" / "run_1" / "data_efficiency.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["runs"]
    paths = {(row["fraction"], row["method"], row["seed"]): "results/run_1/" + row["prediction_file"] for row in rows}
    seeds = sorted({row["seed"] for row in rows})
    saved = {}

    def results(fraction, method, seed):
        p = paths[fraction, method, seed]
        if p not in saved:
            prediction, _ = audit.read_prediction(p)
            full, samples = audit.matrices(prediction)
            saved[p] = (scores(full)[1], scores(samples)[:, 1])
        return p, *saved[p]

    planned = {
        "target_frozen_10_to_30": ((0.1, "minilm"), (0.3, "minilm")),
        "target_frozen_30_to_100": ((0.3, "minilm"), (1.0, "minilm")),
        "target_frozen_10_to_100": ((0.1, "minilm"), (1.0, "minilm")),
        "target_frozen_vs_adapted_10": ((0.1, "minilm"), (0.1, "adapted")),
        "target_frozen_vs_adapted_30": ((0.3, "minilm"), (0.3, "adapted")),
    }
    output = {}
    for key, (baseline, candidate) in planned.items():
        full_deltas, boot_deltas, comparisons = [], [], []
        for seed in seeds:
            old_path, old_full, old_boot = results(*baseline, seed)
            new_path, new_full, new_boot = results(*candidate, seed)
            full_deltas.append(new_full - old_full)
            boot_deltas.append(new_boot - old_boot)
            comparisons.append({"seed": seed, "baseline_predictions": old_path,
                                "candidate_predictions": new_path, "macro_f1_difference": float(new_full - old_full)})
        interval = np.quantile(np.mean(boot_deltas, axis=0), (0.025, 0.975))
        output[key] = {"mean_macro_f1_difference": float(np.mean(full_deltas)),
                       "percentile_95_interval": interval.tolist(), "fixed_seed_comparisons": comparisons}
    return {
        "interpretation": "average paired Macro-F1 differences over the three fixed training draws, using the same article resamples across all draws; no resampling of the training draws; the single full-budget model is reused",
        "comparisons": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--replications", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "paired_group_bootstrap.json")
    parser.add_argument("--pair", nargs=4, action="append", default=[], metavar=("DOMAIN", "NAME", "BASELINE_CSV", "CANDIDATE_CSV"),
                        help="add a fixed-set comparison using paths relative to the project root")
    args = parser.parse_args()
    if args.replications < 1:
        parser.error("--replications must be positive")
    comparisons = {}
    data = {}
    efficiency = {}
    for offset, (domain, settings) in enumerate(DOMAINS.items()):
        seed = args.seed + offset
        audit = GroupedEvaluation(settings, seed, args.replications)
        test_group_sizes = audit.test.groupby("group").size()
        data[domain] = {"data_path": settings["data"], "data_sha256": sha256(audit.dataset.read_bytes()).hexdigest(),
                        "seed": seed, "groups": audit.group_count, "group_unit": audit.group_unit, "n": len(audit.test),
                        "min_group_size": int(test_group_sizes.min()), "max_group_size": int(test_group_sizes.max())}
        for name, key, old, new in (*COMPARISONS, *args.pair):
            if name == domain:
                if key in comparisons:
                    parser.error(f"duplicate comparison name: {key}")
                comparisons[key] = {"domain": domain, **audit.compare(old, new)}
        if domain == "qnli":
            efficiency = efficiency_effects(audit)
    unknown_domains = {domain for domain, *_ in args.pair} - DOMAINS.keys()
    if unknown_domains:
        parser.error(f"unknown domain: {sorted(unknown_domains)}")
    payload = {
        "method": {"replications": args.replications, "interval": "two-sided 95% percentile, 2.5th and 97.5th percentiles",
                   "resampling": "sample the observed number of whole groups uniformly with replacement, applying the same draws to both fixed models",
                   "differences": "candidate minus baseline; 1.0 = 100 percentage points; no retraining; zero if class denominator is zero",
                   "interpretation": "exploratory fixed-model test-group uncertainty; no model selection, training variation or multiplicity correction"},
        "datasets": data, "comparisons": comparisons, "data_efficiency": efficiency,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for key, record in comparisons.items():
        metric = record["metrics"]["macro_f1"]
        print(f"{key}: ΔF1={100*metric['difference']:+.3f} pp, 95% [{100*metric['percentile_95_interval'][0]:+.3f}, "
              f"{100*metric['percentile_95_interval'][1]:+.3f}] pp, flips={record['changed_prediction']}")


if __name__ == "__main__":
    main()
