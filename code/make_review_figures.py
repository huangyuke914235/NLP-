# -*- coding: utf-8 -*-
"""Render verified QEvasion-only results and the actual frozen MiniLM+LR pipeline."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--out", type=Path)
args = parser.parse_args()
base = args.base.resolve()
out = args.out.resolve() if args.out else base / "figures"
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

baseline = read_json(base / "outputs" / "metrics_baselines.json")
st = read_json(base / "outputs" / "metrics_st_lr.json")
hierarchy = read_json(base / "outputs" / "hierarchy" / "metrics_hierarchy.json")
results = [baseline["majority"], baseline["tfidf_lr_test"], st, hierarchy["official_hierarchy"]]
names = ["多数类常数预测", "TF-IDF\n＋逻辑回归", "冻结 MiniLM\n＋逻辑回归", "九细类逻辑回归\n＋概率求和"]
acc = [m["accuracy"] for m in results]
f1 = [m["macro_f1"] for m in results]
cm = np.asarray(st["confusion_matrix"], dtype=int)
assert cm.shape == (3, 3) and int(cm.sum()) == 308

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
x, width = np.arange(4), 0.35
for offset, values, title, color in [(-width / 2, acc, "准确率", "#8EACBD"),
                                     (width / 2, f1, "宏平均 F1（主指标）", "#2E6087")]:
    bars = axes[0].bar(x + offset, values, width, color=color, label=title)
    axes[0].bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4, fontsize=9.2)
axes[0].set_xticks(x, names, fontsize=9.5)
axes[0].set_ylim(0, 0.81)
axes[0].set_ylabel("指标值")
axes[0].set_title("（a）相同官方测试集上的总体结果", fontsize=12)
axes[0].legend(loc="upper center", frameon=False, fontsize=9.5, ncol=2)
axes[0].grid(axis="y", alpha=0.16)
axes[0].set_axisbelow(True)

im = axes[1].imshow(cm, cmap="Blues", vmin=0, vmax=int(cm.max()))
tick_names = ["Clear\nReply", "Ambivalent\nReply", "Clear\nNon-Reply"]
axes[1].set_xticks(np.arange(3), tick_names, fontsize=9.4)
axes[1].set_yticks(np.arange(3), tick_names, fontsize=9.4)
axes[1].set_xlabel("预测标签")
axes[1].set_ylabel("真实标签")
axes[1].set_title("（b）冻结 MiniLM＋逻辑回归混淆矩阵", fontsize=12)
for (i, j), value in np.ndenumerate(cm):
    axes[1].text(j, i, str(value), ha="center", va="center", fontsize=13,
                 color="white" if value > cm.max() * 0.55 else "#182A38")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="样本数")
fig.suptitle("QEvasion 官方测试集（308条；三分类；单次运行）", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.2)
fig.savefig(out / "fig3_results.png", dpi=240)
plt.close(fig)

# This diagram depicts the executed 03_plm_lr.py feature construction, not a new model.
fig, ax = plt.subplots(figsize=(13, 4.0))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4)
ax.axis("off")

def box(x, y, w, h, text, face="#EEF3F7", fontsize=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                                edgecolor="#5E7C93", facecolor=face, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

def arrow(start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "->", "color": "#49677E", "lw": 1.6})

box(0.15, 2.50, 1.50, 0.68, "目标问题 q")
box(0.15, 1.15, 1.50, 0.68, "完整回答 a")
box(2.15, 2.35, 2.20, 0.98, "冻结 MiniLM\n归一化句向量", fontsize=10.7)
box(2.15, 1.00, 2.20, 0.98, "冻结 MiniLM\n归一化句向量", fontsize=10.7)
ax.text(3.25, 2.14, "共享编码器参数；不更新", ha="center", va="center", fontsize=9.2, color="#5B6770")
arrow((1.70, 2.84), (2.09, 2.84))
arrow((1.70, 1.49), (2.09, 1.49))
ax.text(4.79, 2.98, "e_q：384维", ha="center", fontsize=9)
ax.text(4.79, 1.15, "e_a：384维", ha="center", fontsize=9)
box(5.25, 1.25, 2.65, 1.90,
    "四部分特征拼接\n\n[e_q；e_a；|e_q−e_a|；e_q⊙e_a]\n\n384 × 4 ＝ 1,536维", fontsize=10)
arrow((4.42, 2.84), (5.19, 2.66))
arrow((4.42, 1.49), (5.19, 1.72))
box(8.45, 1.40, 1.70, 1.60, "逻辑回归\n\nC＝0.1\nbalanced", face="#E4EFF5", fontsize=10.5)
arrow((7.96, 2.19), (8.39, 2.19))
box(10.70, 1.10, 2.10, 2.20, "清晰度三分类\n\nClear Reply\nAmbivalent Reply\nClear Non-Reply", face="#EDF2EA", fontsize=10.2)
arrow((10.22, 2.19), (10.64, 2.19))
ax.text(6.5, 0.38,
        "每段默认最多256个 word pieces；编码批大小64；CPU编码；仅训练逻辑回归（迭代上限2,000）",
        ha="center", va="center", fontsize=10, color="#344B5A")
ax.text(6.5, 3.78, "实际运行的冻结句向量＋逻辑回归流程", ha="center", va="center", fontsize=14)
fig.tight_layout(pad=0.8)
fig.savefig(out / "fig4_frozen_lr_pipeline.png", dpi=240)
plt.close(fig)
print("Saved:", out / "fig3_results.png")
print("Saved:", out / "fig4_frozen_lr_pipeline.png")
