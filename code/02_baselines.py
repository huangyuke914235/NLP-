# -*- coding: utf-8 -*-
"""
任务二：传统基线
- 多数类基线（Majority）
- TF-IDF + Logistic Regression（必做）
  消融：unigram vs unigram+bigram；class_weight=None vs balanced
  模型选择在固定种子的验证集（train 90/10 分层切分）上进行，
  选定配置后在完整训练集重训并在官方 test 上评估。
输出：outputs/metrics_baselines.json + outputs/predictions_tfidf_lr.csv
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (SEED, evaluate, load_adapter, make_pair_text,
                    save_metrics, save_predictions, set_seed, timer)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)
set_seed(SEED)
tick = timer()

train, test, meta = load_adapter(os.path.join(BASE, "configs", "qevasion.json"))
label_names = meta["label_names"]
print(f"loaded in {tick()}s | train={len(train)} test={len(test)}")

X_train_pair = make_pair_text(train["text_a"], train["text_b"]).to_numpy(dtype=object)
y_train = train["label"].to_numpy()
X_test_pair = make_pair_text(test["text_a"], test["text_b"]).to_numpy(dtype=object)
y_test = test["label"].to_numpy()

# 固定种子 90/10 分层切分出验证集（仅用于模型选择）
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_pair, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
print(f"fit={len(X_fit)} val={len(X_val)} ({tick()}s)")

results = {}

# ---------- 多数类基线 ----------
majority = int(np.bincount(y_train).argmax())
y_majority = np.full_like(y_test, majority)
results["majority"] = evaluate(y_test, y_majority, None, label_names)
results["majority"]["majority_class"] = label_names[majority]
print("majority:", results["majority"]["accuracy"], results["majority"]["macro_f1"])


def build_tfidf_lr(ngram=(1, 1), class_weight=None, C=1.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram, sublinear_tf=True, min_df=2, max_df=0.95)),
        ("lr", LogisticRegression(max_iter=2000, class_weight=class_weight, C=C, random_state=SEED)),
    ])


# ---------- 消融（验证集选择） ----------
ablation = {}
for ngram_name, ngram in [("unigram", (1, 1)), ("uni+bigram", (1, 2))]:
    for cw_name, cw in [("no_cw", None), ("balanced", "balanced")]:
        pipe = build_tfidf_lr(ngram, cw)
        pipe.fit(X_fit, y_fit)
        val_pred = pipe.predict(X_val)
        acc = float((val_pred == y_val).mean())
        mf1 = float(f1_score(y_val, val_pred, average="macro"))
        ablation[f"tfidf_{ngram_name}_{cw_name}"] = {"val_accuracy": acc, "val_macro_f1": mf1}
        print(f"{ngram_name:10s} cw={str(cw):8s} val_acc={acc:.4f} val_macroF1={mf1:.4f}")
results["tfidf_ablation_val"] = ablation

best_key = max(ablation, key=lambda k: ablation[k]["val_macro_f1"])
print("best config:", best_key)
results["tfidf_best_config"] = best_key

# ---------- 用最佳配置在完整训练集重训 → 官方测试集 ----------
best_ngram = (1, 2) if "bigram" in best_key else (1, 1)
best_cw = "balanced" if "balanced" in best_key else None
final = build_tfidf_lr(best_ngram, best_cw)
final.fit(X_train_pair, y_train)
test_pred = final.predict(X_test_pair)
test_proba = final.predict_proba(X_test_pair)
res_test = evaluate(y_test, test_pred, test_proba, label_names)
n_feat = len(final.named_steps["tfidf"].vocabulary_)
res_test["vocab_size"] = n_feat
results["tfidf_lr_test"] = res_test
print(f"TF-IDF+LR test: acc={res_test['accuracy']:.4f} macroF1={res_test['macro_f1']:.4f} vocab={n_feat} ({tick()}s)")

save_predictions(
    ids=test["uid"].values,
    texts=(test["text_a"].values, test["text_b"].values),
    gold=y_test, pred=test_pred, proba=test_proba,
    label_names=label_names,
    path=os.path.join(OUT, "predictions_tfidf_lr.csv"))

save_metrics(results, os.path.join(OUT, "metrics_baselines.json"))
print(f"saved metrics + predictions ({tick()}s total)")
