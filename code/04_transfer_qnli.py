# -*- coding: utf-8 -*-
"""任务三：把统一的文本对流程迁移到 QNLI。

QNLI 的 question/sentence 对被适配为 text_a/text_b，标签为二分类：
0=Entailment (contains answer)，1=Not Entailment。为控制运行时间，
从官方 GLUE train.tsv 按固定种子抽取 12,000 条训练样本，dev.tsv 全量
作为迁移测试集。输出格式与前两个实验一致。
"""
import json
import os
import sys
import zipfile
from urllib.request import Request, urlopen

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
# 允许在 pytorch 环境中使用项目内、无需写入 Conda 的依赖目录。
sys.path.insert(0, os.path.join(BASE, "deps"))
sys.path.insert(0, BASE)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common import SEED, evaluate, make_pair_text, save_metrics, save_predictions, set_seed

OUT = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data", "qnli_raw")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
set_seed(SEED)

LABEL_NAMES = ["Entailment (contains answer)", "Not Entailment"]
URL = "https://dl.fbaipublicfiles.com/glue/data/QNLI.zip"
ZIP_PATH = os.path.join(DATA, "QNLI.zip")


def ensure_qnli():
    """下载并解压官方 GLUE QNLI 文件；已有文件时不重复下载。"""
    train_path = os.path.join(DATA, "train.tsv")
    dev_path = os.path.join(DATA, "dev.tsv")
    if not (os.path.exists(train_path) and os.path.exists(dev_path)):
        if not os.path.exists(ZIP_PATH):
            print("downloading", URL)
            req = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=120) as src, open(ZIP_PATH, "wb") as dst:
                total = 0
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    total += len(chunk)
            print(f"downloaded {total / 1024 / 1024:.1f} MB")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            names = {os.path.basename(n): n for n in zf.namelist()}
            for name in ("train.tsv", "dev.tsv"):
                if name not in names:
                    raise FileNotFoundError(f"{name} not found in QNLI.zip")
                with zf.open(names[name]) as src, open(os.path.join(DATA, name), "wb") as dst:
                    dst.write(src.read())
    return train_path, dev_path


def load_qnli():
    train_path, dev_path = ensure_qnli()
    def read_tsv_robust(path):
        # 个别句子内部含有制表符；按首列 index、第二列 question、末列
        # label 解析，中间所有列重新拼回 sentence，避免丢样本。
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n\r").split("\t")
            for line in f:
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) < 4:
                    continue
                rows.append({"index": parts[0], "question": parts[1],
                             "sentence": "\t".join(parts[2:-1]), "label": parts[-1]})
        return pd.DataFrame(rows, columns=header[:1] + ["question", "sentence", "label"])

    tr = read_tsv_robust(train_path)
    te = read_tsv_robust(dev_path)
    # GLUE 的标签有版本差异（字符串或数字），统一为配置中的 0/1。
    label_map = {"entailment": 0, "not_entailment": 1, "0": 0, "1": 1, 0: 0, 1: 1}

    def adapt(df):
        out = pd.DataFrame()
        out["text_a"] = df["question"].astype(str)
        out["text_b"] = df["sentence"].astype(str)
        out["label"] = df["label"].map(lambda x: label_map.get(str(x).lower(), label_map.get(x)))
        if out["label"].isna().any():
            raise ValueError(f"unknown QNLI labels: {df.loc[out['label'].isna(), 'label'].unique()}")
        out["label"] = out["label"].astype(int)
        out["group"] = df["index"].astype(str) if "index" in df.columns else ""
        out["uid"] = df["index"] if "index" in df.columns else np.arange(len(df))
        return out

    tr = adapt(tr)
    te = adapt(te)
    # 固定种子抽样，保持配置说明中的可复现运行成本。
    tr = tr.groupby("label", group_keys=False).sample(n=6000, random_state=SEED)
    tr = tr.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return tr, te


def build(ngram=(1, 2), class_weight="balanced"):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram, sublinear_tf=True, min_df=2, max_df=0.95)),
        ("lr", LogisticRegression(max_iter=2000, class_weight=class_weight, C=1.0, random_state=SEED)),
    ])


train, test = load_qnli()
label_names = LABEL_NAMES
y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()
X_train = make_pair_text(train["text_a"], train["text_b"]).to_numpy(dtype=object)
X_test = make_pair_text(test["text_a"], test["text_b"]).to_numpy(dtype=object)

# 用固定验证切分选择 unigram / unigram+bigram 与类别权重。
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
ablation = {}
for ngram_name, ngram in (("unigram", (1, 1)), ("uni+bigram", (1, 2))):
    for cw_name, cw in (("no_cw", None), ("balanced", "balanced")):
        model = build(ngram, cw)
        model.fit(X_fit, y_fit)
        pred = model.predict(X_val)
        ablation[f"tfidf_{ngram_name}_{cw_name}"] = {
            "val_accuracy": float((pred == y_val).mean()),
            "val_macro_f1": float(f1_score(y_val, pred, average="macro")),
        }
        print(f"{ngram_name:10s} cw={str(cw):8s} "
              f"val_acc={ablation[f'tfidf_{ngram_name}_{cw_name}']['val_accuracy']:.4f} "
              f"val_macroF1={ablation[f'tfidf_{ngram_name}_{cw_name}']['val_macro_f1']:.4f}")

best_key = max(ablation, key=lambda k: ablation[k]["val_macro_f1"])
best_ngram = (1, 2) if "bigram" in best_key else (1, 1)
best_cw = "balanced" if "balanced" in best_key else None
model = build(best_ngram, best_cw)
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)
res = evaluate(y_test, pred, proba, label_names)
res.update({
    "model": "TF-IDF(question + [SEP] + sentence) + Logistic Regression",
    "dataset": "nyu-mll/glue/qnli (official GLUE TSV)",
    "train_n": int(len(train)),
    "test_n": int(len(test)),
    "best_config": best_key,
    "val_selection": ablation,
    "seed": SEED,
    "note": "train 按类别各抽样 6000 条；QNLI dev 全量作为迁移测试集。",
})
print("QNLI test:", res["accuracy"], res["macro_f1"])

save_predictions(
    test["uid"].to_numpy(),
    (test["text_a"].to_numpy(), test["text_b"].to_numpy()),
    y_test, pred, proba, label_names,
    os.path.join(OUT, "predictions_qnli_tfidf_lr.csv"),
)
save_metrics(res, os.path.join(OUT, "metrics_qnli_tfidf_lr.json"))
with open(os.path.join(OUT, "qnli_data_meta.json"), "w", encoding="utf-8") as f:
    json.dump({"source_url": URL, "train_n": int(len(train)), "test_n": int(len(test)),
               "label_names": LABEL_NAMES, "seed": SEED}, f, ensure_ascii=False, indent=2)
print("saved QNLI metrics, predictions and metadata")
