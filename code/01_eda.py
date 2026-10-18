# -*- coding: utf-8 -*-
"""QEvasion EDA with complete length plots and explicit source-overlap audit.

Run in the code directory as before: python 01_eda.py
Optional isolated output: python 01_eda.py --base PATH_TO_CODE --out PATH_TO_FIGURES
The original eda_stats.json fields remain; misleading legacy names are documented.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent,
                    help="Project code directory containing common.py, configs and data")
parser.add_argument("--out", type=Path, help="Figure directory (default: BASE/figures)")
parser.add_argument("--stats-out", type=Path, help="JSON path (default: sibling outputs/eda_stats.json)")
args = parser.parse_args()
BASE = args.base.resolve()
FIG = args.out.resolve() if args.out else BASE / "figures"
STATS = args.stats_out.resolve() if args.stats_out else (
    FIG.parent / "outputs" / "eda_stats.json" if args.out else BASE / "outputs" / "eda_stats.json"
)
sys.path.insert(0, str(BASE))
from common import SEED, load_adapter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from datasets import load_from_disk

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
FIG.mkdir(parents=True, exist_ok=True)
STATS.parent.mkdir(parents=True, exist_ok=True)
LABEL_NAMES = ["Clear Reply", "Ambivalent Reply", "Clear Non-Reply"]
COLORS = ["#3978AD", "#8266A6", "#C75B52"]

# Resolve the source config's relative data path without changing the working directory
# or writing to the original project. The existing adapter remains the only field mapper.
cfg = json.loads((BASE / "configs" / "qevasion.json").read_text(encoding="utf-8-sig"))
if cfg.get("local_path") and not Path(cfg["local_path"]).is_absolute():
    cfg["local_path"] = str(BASE / cfg["local_path"])
with tempfile.TemporaryDirectory(prefix="qevasion-eda-config-", dir=FIG.parent) as resolved_dir:
    resolved_cfg = Path(resolved_dir) / "qevasion.json"
    resolved_cfg.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    train, test, meta = load_adapter(str(resolved_cfg))
raw = load_from_disk(str(BASE / "data" / "qevasion_raw"))
tr_raw, te_raw = raw["train"].to_pandas(), raw["test"].to_pandas()
print(f"train={len(train)}, test={len(test)}, label_names={meta['label_names']}")
stats = {"seed": SEED, "train_n": int(len(train)), "test_n": int(len(test))}

for name, df, original in [("train", train, tr_raw), ("test", test, te_raw)]:
    na = df.isna().mean().round(4).to_dict()
    empty = {"text_a_empty": int((df["text_a"].str.strip() == "").sum()),
             "text_b_empty": int((df["text_b"].str.strip() == "").sum()),
             "label_empty": int(df["label"].isna().sum())}
    stats[f"{name}_missing_ratio"] = na
    stats[f"{name}_empty"] = empty
    stats[f"{name}_raw_field_dtypes"] = {c: str(t) for c, t in original.dtypes.items()}
    stats[f"{name}_raw_missing_ratio"] = original.isna().mean().round(4).to_dict()
    print(name, "missing:", {k: v for k, v in na.items() if v > 0}, "empty:", empty)
    for field, display in [("text_a", "question"), ("text_b", "answer")]:
        counts = df[field].str.split().str.len()
        stats[f"{name}_{display}_len"] = {"mean": round(float(counts.mean()), 1),
            "median": float(counts.median()), "p95": float(counts.quantile(0.95)),
            "max": int(counts.max())}
print("length:", {k: v for k, v in stats.items() if k.endswith("_len")})

tr_cnt = train["label"].value_counts().reindex(range(3), fill_value=0)
te_cnt = test["label"].value_counts().reindex(range(3), fill_value=0)
for name, count, frame in [("train", tr_cnt, train), ("test", te_cnt, test)]:
    stats[f"{name}_label_counts"] = {LABEL_NAMES[i]: int(count[i]) for i in range(3)}
    stats[f"{name}_label_ratio"] = {LABEL_NAMES[i]: round(float(count[i]) / len(frame), 4)
                                   for i in range(3)}
print("label counts:", stats["train_label_counts"], stats["test_label_counts"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={"width_ratios": [1, 1.2]})
x, width = np.arange(3), 0.37
for positions, values, label, color in [(x - width / 2, tr_cnt.values, "训练集", "#3978AD"),
                                      (x + width / 2, te_cnt.values, "官方测试集", "#D08A4C")]:
    axes[0].bar(positions, values, width, label=label, color=color)
    for pos, value in zip(positions, values):
        axes[0].text(pos, value + 22, str(value), ha="center", fontsize=9)
axes[0].set_xticks(x, ["Clear\nReply", "Ambivalent\nReply", "Clear\nNon-Reply"])
axes[0].set_ylim(0, float(tr_cnt.max()) * 1.17)
axes[0].set_ylabel("样本数")
axes[0].set_title("（a）回答清晰度三分类")
axes[0].legend(frameon=False)
ev_order = ["Explicit", "Implicit", "Dodging", "General", "Deflection",
            "Partial/half-answer", "Declining to answer", "Claims ignorance", "Clarification"]
ev = tr_raw["evasion_label"].astype(str).value_counts().reindex(ev_order, fill_value=0)
ev_colors = [COLORS[0] if v == "Explicit" else COLORS[2] if v in ev_order[-3:] else COLORS[1]
             for v in ev_order]
axes[1].barh(np.arange(9), ev.values, color=ev_colors)
axes[1].set_yticks(np.arange(9), ev.index, fontsize=9)
axes[1].invert_yaxis()
for i, value in enumerate(ev.values):
    axes[1].text(value + 12, i, str(value), va="center", fontsize=9)
axes[1].set_xlim(0, float(ev.max()) * 1.15)
axes[1].set_xlabel("样本数（训练集；测试集细标签全部缺失）")
axes[1].set_title("（b）九类回避策略")
fig.tight_layout(w_pad=2.5)
fig.savefig(FIG / "fig1_label_distribution.png", dpi=220)
plt.close(fig)

# All classes use the SAME bins. No clipping or omitted tail observations.
train["q_len"] = train["text_a"].str.split().str.len()
train["a_len"] = train["text_b"].str.split().str.len()
answer_min, answer_max = int(train["a_len"].min()), int(train["a_len"].max())
if answer_min <= 0:
    raise ValueError("Log-length plotting requires non-empty answer text; inspect missing-text audit.")
answer_bins = np.geomspace(max(0.5, answer_min * 0.8), answer_max * 1.04, 42)
question_max = int(train["q_len"].max())
question_bins = np.arange(-0.5, question_max + 2.5, 2)
tail = train.loc[train["a_len"] >= 600]
tail_counts = tail["label"].value_counts().reindex(range(3), fill_value=0)
stats["train_answer_tail_ge600"] = {"n": int(len(tail)), "ratio": float(len(tail) / len(train)),
    "label_counts": {LABEL_NAMES[i]: int(tail_counts[i]) for i in range(3)},
    "ambivalent_share": float(tail_counts[1] / len(tail)) if len(tail) else None}
stats["length_figure_policy"] = {
    "word_count": "str.split() on whitespace", "sample_scope": f"all {len(train)} training samples",
    "answer_x_axis": "logarithmic; 41 shared log-spaced bins; no clipping",
    "question_x_axis": "linear; shared two-word bins over the full observed range",
    "answer_histogram_count": int(sum(np.histogram(train["a_len"], answer_bins)[0])),
    "question_histogram_count": int(sum(np.histogram(train["q_len"], question_bins)[0]))}
assert stats["length_figure_policy"]["answer_histogram_count"] == len(train)
assert stats["length_figure_policy"]["question_histogram_count"] == len(train)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for i, name in enumerate(LABEL_NAMES):
    subset = train.loc[train["label"] == i]
    legend = f"{name}（n={len(subset):,}）"
    axes[0].hist(subset["a_len"], bins=answer_bins, histtype="step", linewidth=1.6,
                 label=legend, color=COLORS[i])
    axes[1].hist(subset["q_len"], bins=question_bins, histtype="step", linewidth=1.6,
                 label=legend, color=COLORS[i])
axes[0].set_xscale("log")
axes[0].set_xlim(answer_bins[0], answer_bins[-1])
axes[0].set_xticks([1, 3, 10, 30, 100, 300, 1000, 2117])
axes[0].xaxis.set_major_formatter(ScalarFormatter())
axes[0].set_xlabel("回答词数（对数横轴；全部样本）")
axes[0].set_ylabel("每个对数分箱内的样本数")
axes[0].set_title("（a）回答长度：长尾完整保留")
axes[0].legend(fontsize=8.1, frameon=False, loc="upper left")
tail_note = f"≥600词：{len(tail)}条（{len(tail) / len(train):.1%}），已包含于图中\n" + \
    f"明确回答 {tail_counts[0]}；模糊回答 {tail_counts[1]}；明确不答 {tail_counts[2]}"
axes[0].text(0.02, 0.72, tail_note, transform=axes[0].transAxes, ha="left", va="top",
             fontsize=8.4, bbox={"facecolor": "white", "edgecolor": "#B9C6CF", "alpha": 0.94})
axes[1].set_xlim(0, question_max + 1)
axes[1].set_xlabel("问题词数（线性横轴；全部样本）")
axes[1].set_ylabel("每个两词分箱内的样本数")
axes[1].set_title("（b）问题长度：使用相同分箱边界")
axes[1].legend(fontsize=8.1, frameon=False)
for ax in axes:
    ax.grid(axis="y", alpha=0.16)
fig.tight_layout(w_pad=2.2)
fig.savefig(FIG / "fig2_length_distribution.png", dpi=220)
plt.close(fig)
print("tail >=600 words:", stats["train_answer_tail_ge600"])
print("length histogram sample coverage:", stats["length_figure_policy"])

tr_pres = train["group"].replace({"None": None}).dropna()
pres_cnt = tr_pres.value_counts()
stats["train_president_counts"] = {str(k): int(v) for k, v in pres_cnt.items()}
stats["qa_text_overlap_train_test"] = {
    "question_exact_overlap": len(set(train["text_a"]) & set(test["text_a"])),
    "pair_exact_overlap": len(set(zip(train["text_a"], train["text_b"])) &
                              set(zip(test["text_a"], test["text_b"]))),
    "answer_exact_overlap": len(set(train["text_b"]) & set(test["text_b"]))}
title_n = int(tr_raw["title"].nunique(dropna=True))
stats["train_unique_title_texts"] = title_n
# Kept only for backwards-compatible consumers. A title is NOT a unique interview identifier.
stats["train_unique_interviews"] = title_n
stats["legacy_field_notes"] = {"train_unique_interviews":
    "Deprecated: this field counts distinct title strings, not distinct interviews. Use train_unique_title_texts."}
stats["train_dup_question_rows"] = int(tr_raw["question"].duplicated().sum())
url_sets = []
for name, frame in [("train", tr_raw), ("test", te_raw)]:
    series = frame["url"].dropna().astype(str).str.strip()
    url_sets.append(set(series[series != ""]))
    stats[f"{name}_unique_source_urls"] = len(url_sets[-1])
shared_urls = url_sets[0] & url_sets[1]
stats["source_url_overlap_train_test"] = {"shared_url_count": len(shared_urls),
    "train_rows_on_shared_urls": int(tr_raw["url"].isin(shared_urls).sum()),
    "test_rows_on_shared_urls": int(te_raw["url"].isin(shared_urls).sum()),
    "test_row_fraction": float(te_raw["url"].isin(shared_urls).mean()),
    "shared_urls": sorted(shared_urls),
    "interpretation": "Exact URL overlap demonstrates shared source pages; it does not measure exact duplicate text or prove model memorization."}
stats["title_interpretation"] = f"{title_n} distinct non-null title strings; titles may recur across source URLs and are not unique interview IDs."
tab = []
for president in pres_cnt.index:
    counts = train.loc[train["group"] == president, "label"].value_counts().reindex(range(3), fill_value=0)
    tab.append([president] + [round(float(v / counts.sum()), 3) for v in counts.values])
stats["president_label_ratio"] = tab
print("distinct title strings:", title_n, "duplicate question rows:", stats["train_dup_question_rows"])
print("source URLs:", stats["train_unique_source_urls"], stats["test_unique_source_urls"],
      "shared:", len(shared_urls), "test rows:", stats["source_url_overlap_train_test"]["test_rows_on_shared_urls"])
print("overlap train/test:", stats["qa_text_overlap_train_test"])
with STATS.open("w", encoding="utf-8") as handle:
    json.dump(stats, handle, ensure_ascii=False, indent=2)
print("stats saved:", STATS)
print("figures saved:", FIG)
