"""Generate report figures from pinned data, source logs and audited predictions."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ("#234D75", "#317C86", "#8798A8")
LABELS = ("Clear Reply", "Ambivalent", "Clear Non-Reply")
FIELDS = ("question_length", "answer_length")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dataset(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(figure, folder, name):
    figure.savefig(folder / name, dpi=225, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def distributions(root):
    rows = dataset(root / "processed/qevasion_official.csv")
    folder = root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    train = [r for r in rows if r["official_split"] == "train"]
    test = [r for r in rows if r["official_split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.55), gridspec_kw={"width_ratios": [1, 1.22]}, layout="constrained")
    x = np.arange(3)
    for i, (part, label) in enumerate(((train, "Official train"), (test, "Official test"))):
        counts = Counter(row["label"] for row in part)
        bars = axes[0].bar(x + (i - .5) * .35, [counts[k] for k in LABELS], .35, color=COLORS[i], label=label)
        axes[0].bar_label(bars, fontsize=8, padding=1)
    axes[0].set_xticks(x, ("Clear\nReply", "Ambivalent", "Clear\nNon-Reply"))
    axes[0].set_ylim(0, 2380)
    axes[0].set_ylabel("Rows")
    axes[0].set_title("A  Response clarity", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    c = Counter(row["fine_label"] for row in train)
    ordered = sorted(c, key=c.get)
    bars = axes[1].barh(ordered, [c[k] for k in ordered], color=COLORS[0])
    axes[1].bar_label(bars, fontsize=8, padding=3)
    axes[1].set_title("B  Evasion labels (training only)", loc="left", fontweight="bold")
    axes[1].set_xlim(0, max(c.values()) * 1.17)
    axes[1].set_xlabel("Rows")
    save(fig, folder, "label_distribution.png")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.45), layout="constrained")
    for axis, field, label in zip(axes, FIELDS, ("A  Question length", "B  Answer length")):
        for value, color in zip(LABELS, COLORS):
            data = np.sort([int(r[field]) for r in rows if r["label"] == value])
            axis.step(data, np.arange(1, len(data) + 1) / len(data), where="post", lw=1.75, color=color, label=value)
        axis.set_xscale("log")
        axis.set_xlabel("Normalized words (log scale)")
        axis.set_ylabel("Cumulative proportion")
        axis.set_title(label, loc="left", fontweight="bold")
        axis.grid(alpha=.17)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    save(fig, folder, "length_distribution.png")


def result_figures(run, folder, audit, pair_run):
    metrics = load(run / "metrics.json")
    variants = [("grouped_tfidf", "TF-IDF + LR"), ("grouped_minilm", "MiniLM + LR"), ("grouped_minilm_interactions", "MiniLM + interactions")]
    if pair_run is not None:
        metrics.update(load(pair_run / "metrics.json"))
        variants.append(("grouped_pair", "Joint MiniLM"))
    fig, axes = plt.subplots(1, len(variants), figsize=(10.4, 3.15), layout="constrained")
    for axis, (key, title) in zip(axes, variants):
        values = np.asarray(metrics[key]["confusion_matrix"])
        axis.imshow(values, cmap="Blues", vmin=0, vmax=206)
        for y, row in enumerate(values):
            for x, n in enumerate(row):
                axis.text(x, y, str(n), ha="center", va="center", color="white" if n > 115 else "#0C2743", fontsize=13)
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.set_xticks(range(3), ("CR", "AM", "NR"))
        axis.set_yticks(range(3), ("CR", "AM", "NR"))
        axis.set_xlabel("Prediction")
    axes[0].set_ylabel("Gold")
    save(fig, folder, "report_confusion.png" if pair_run is not None else "source_confusion.png")
    fig, ax = plt.subplots(figsize=(8.4, 3.15), layout="constrained")
    summaries = load(run / "data_efficiency.json")["summary"]
    for method, color, label in zip(("minilm", "adapted"), COLORS, ("Target MiniLM + LR", "Target + source score")):
        items = sorted((r for r in summaries if r["method"] == method), key=lambda r: r["fraction"])
        ax.errorbar([r["fraction"] * 100 for r in items], [r["mean_macro_f1"] for r in items],
                    yerr=[r["sd_macro_f1"] for r in items], label=label, color=color, linewidth=2, capsize=4, marker="o")
    ax.set_xlabel("Target training labels (%)")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks([10, 30, 100])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.22)
    save(fig, folder, "data_efficiency.png")
    lines = (run / "run.log").read_text(encoding="utf-8-sig").splitlines()
    wanted = ("Started", "device=", "Read grouped", "grouped_tfidf n=", "grouped_minilm_interactions n=", "Read QNLI", "target_direct n=", "target_adapted n=", "Completed")
    sample = [line.split("] ", 1)[1] for line in lines if "] " in line and any(item in line for item in wanted)]
    sample += [f"Independent replay: {audit['predictions_verified']} files per run, {audit['total_changed_hard_predictions']} changed hard predictions."]
    sample = [short for line in sample for short in textwrap.wrap(line, width=97)]
    fig = plt.figure(figsize=(10.1, .28 * len(sample) + .35), facecolor="#F3F6F9")
    fig.text(.02, .95, "\n".join(sample), fontfamily="monospace", color="#12314D", fontsize=8.5, linespacing=1.35, va="top")
    save(fig, folder, "run_log_excerpt.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--run", type=Path, default=Path("results/run_1"))
    parser.add_argument("--audit", type=Path, default=Path("evidence/prediction_audit.json"))
    parser.add_argument("--out", type=Path, default=Path("evidence/figures"))
    parser.add_argument("--pair-run", type=Path, default=None)
    parser.add_argument("--eda-only", action="store_true")
    args = parser.parse_args()
    distributions(args.data)
    if not args.eda_only:
        args.out.mkdir(parents=True, exist_ok=True)
        result_figures(args.run, args.out, load(args.audit), args.pair_run)


if __name__ == "__main__":
    main()
