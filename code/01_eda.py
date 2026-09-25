# -*- coding: utf-8 -*-
"""
任务一：探索性数据分析（EDA）
- 样本数、字段类型、缺失值比例
- clarity_label / evasion_label 类别分布（图1）
- question / interview_answer 长度分布（图2）
- president / title 分组与跨划分泄漏检查
输出：outputs/eda_stats.json + figures/fig1..fig3
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SEED, load_adapter

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

LABEL_NAMES = ["Clear Reply", "Ambivalent Reply", "Clear Non-Reply"]

train, test, meta = load_adapter(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "qevasion.json"))
print(f"train={len(train)}, test={len(test)}, label_names={meta['label_names']}")

# 细粒度 evasion_label 不在统一适配器的 label_raw 中，直接从原始数据读取。
from datasets import load_from_disk
raw = load_from_disk(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qevasion_raw"))
tr_raw = raw["train"].to_pandas()

stats = {"seed": SEED, "train_n": int(len(train)), "test_n": int(len(test))}

# ---------- 缺失与字段类型 ----------
for name, df in [("train", train), ("test", test)]:
    na = df.isna().mean().round(4).to_dict()
    empty = {
        "text_a_empty": int((df["text_a"].str.strip() == "").sum()),
        "text_b_empty": int((df["text_b"].str.strip() == "").sum()),
        "label_empty": int((df["label"].isna()).sum()),
    }
    stats[f"{name}_missing_ratio"] = na
    stats[f"{name}_empty"] = empty
    print(name, "missing:", {k: v for k, v in na.items() if v > 0}, "empty:", empty)

# ---------- 长度统计（词数） ----------
for name, df in [("train", train), ("test", test)]:
    qa = df["text_a"].str.split().str.len()
    ab = df["text_b"].str.split().str.len()
    stats[f"{name}_question_len"] = {"mean": round(float(qa.mean()), 1), "median": float(qa.median()),
                                     "p95": float(qa.quantile(0.95)), "max": int(qa.max())}
    stats[f"{name}_answer_len"] = {"mean": round(float(ab.mean()), 1), "median": float(ab.median()),
                                   "p95": float(ab.quantile(0.95)), "max": int(ab.max())}
print("length:", {k: v for k, v in stats.items() if "len" in k})

# ---------- 类别分布 ----------
tr_cnt = train["label"].value_counts().reindex(range(3), fill_value=0)
te_cnt = test["label"].value_counts().reindex(range(3), fill_value=0)
stats["train_label_counts"] = {LABEL_NAMES[i]: int(tr_cnt[i]) for i in range(3)}
stats["test_label_counts"] = {LABEL_NAMES[i]: int(te_cnt[i]) for i in range(3)}
stats["train_label_ratio"] = {LABEL_NAMES[i]: round(tr_cnt[i] / len(train), 4) for i in range(3)}
stats["test_label_ratio"] = {LABEL_NAMES[i]: round(te_cnt[i] / len(test), 4) for i in range(3)}
print("label counts:", stats["train_label_counts"], stats["test_label_counts"])

# ---------- 图1：标签分布 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(3)
w = 0.38
axes[0].bar(x - w / 2, tr_cnt.values, w, label="train", color="#4C72B0")
axes[0].bar(x + w / 2, te_cnt.values, w, label="test", color="#DD8452")
for i, v in enumerate(tr_cnt.values):
    axes[0].text(i - w / 2, v + 15, str(v), ha="center", fontsize=9)
for i, v in enumerate(te_cnt.values):
    axes[0].text(i + w / 2, v + 15, str(v), ha="center", fontsize=9)
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Clear Reply", "Ambivalent\nReply", "Clear\nNon-Reply"])
axes[0].set_ylabel("样本数")
axes[0].set_title("(a) clarity_label 高层三分类分布")
axes[0].legend()

ev = tr_raw["evasion_label"].astype(str).value_counts()
ev_order = ["Explicit", "Implicit", "Dodging", "General", "Deflection",
            "Partial/half-answer", "Declining to answer", "Claims ignorance", "Clarification"]
ev = ev.reindex(ev_order, fill_value=0)
colors = ["#55A868" if n == "Explicit" else "#C44E52" if n in ("Declining to answer", "Claims ignorance", "Clarification") else "#8172B3" for n in ev.index]
axes[1].barh(range(len(ev)), ev.values, color=colors)
axes[1].set_yticks(range(len(ev)))
axes[1].set_yticklabels(ev.index, fontsize=9)
axes[1].invert_yaxis()
for i, v in enumerate(ev.values):
    axes[1].text(v + 8, i, str(v), va="center", fontsize=8.5)
axes[1].set_xlabel("样本数（仅训练集标注，测试集该字段为空）")
axes[1].set_title("(b) evasion_label 九类细粒度分布（train）")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_label_distribution.png"), dpi=200)
plt.close()
print("fig1 saved")

# ---------- 图2：长度分布（按类别） ----------
train["q_len"] = train["text_a"].str.split().str.len()
train["a_len"] = train["text_b"].str.split().str.len()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
colors3 = ["#4C72B0", "#8172B3", "#C44E52"]
for i, name in enumerate(LABEL_NAMES):
    sub = train.loc[train["label"] == i, "a_len"].clip(upper=600)
    axes[0].hist(sub, bins=40, alpha=0.55, label=name, color=colors3[i])
axes[0].set_xlabel("回答词数（截断于600）")
axes[0].set_ylabel("样本数")
axes[0].set_title("(a) interview_answer 长度分布（按 clarity_label）")
axes[0].legend(fontsize=8.5)
for i, name in enumerate(LABEL_NAMES):
    sub = train.loc[train["label"] == i, "q_len"].clip(upper=60)
    axes[1].hist(sub, bins=30, alpha=0.55, label=name, color=colors3[i])
axes[1].set_xlabel("问题词数（截断于60）")
axes[1].set_ylabel("样本数")
axes[1].set_title("(b) question 长度分布（按 clarity_label）")
axes[1].legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_length_distribution.png"), dpi=200)
plt.close()
print("fig2 saved")

# ---------- president 分组（仅训练集有该字段） ----------
tr_pres = train["group"].replace({"None": None}).dropna()
pres_cnt = tr_pres.value_counts()
stats["train_president_counts"] = pres_cnt.to_dict()
print("president:", pres_cnt.to_dict())

# title 在测试集全为空 → 用文本重叠检查泄漏
q_overlap = len(set(train["text_a"]) & set(test["text_a"]))
pair_overlap = len(set(zip(train["text_a"], train["text_b"])) & set(zip(test["text_a"], test["text_b"])))
stats["qa_text_overlap_train_test"] = {"question_exact_overlap": int(q_overlap), "pair_exact_overlap": int(pair_overlap)}
print("overlap train/test:", stats["qa_text_overlap_train_test"])

# 训练集内部：同一场采访多个子问题
tr_title = train["group"].astype(str)
n_titles_train = None

stats["train_unique_interviews"] = int(tr_raw["title"].nunique())
stats["train_dup_question_rows"] = int(tr_raw["question"].duplicated().sum())
print("unique interviews(train):", stats["train_unique_interviews"], "dup questions:", stats["train_dup_question_rows"])

# 按 president 的标签比例（分组差异）
tab = []
for p in pres_cnt.index:
    sub = train.loc[train["group"] == p, "label"].value_counts().reindex(range(3), fill_value=0)
    tab.append([p] + [round(v / sub.sum(), 3) for v in sub.values])
stats["president_label_ratio"] = tab
print("president label ratio:", tab)

with open(os.path.join(OUT, "eda_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("eda_stats.json saved")
