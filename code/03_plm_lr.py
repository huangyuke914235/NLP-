# -*- coding: utf-8 -*-
"""
任务二（续）：预训练表示基线 B-A：Sentence Transformer 表示 + Logistic Regression
- all-MiniLM-L6-v2（384 维）分别编码 question 与 answer
- 特征拼接 [e_q; e_a; |e_q-e_a|; e_q*e_a] -> 1536 维
- LR（class_weight=balanced），C 在验证集 {0.1,1,10} 中选择
输出：outputs/metrics_st_lr.json + outputs/predictions_st_lr.csv + outputs/st_embeddings.npz
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (SEED, evaluate, load_adapter, save_metrics,
                    save_predictions, set_seed)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)
set_seed(SEED)

train, test, meta = load_adapter(os.path.join(BASE, "configs", "qevasion.json"))
label_names = meta["label_names"]
y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()

# ---------- 编码 ----------
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
t0 = time.time()
st = SentenceTransformer(MODEL, device="cpu")  # 无 GPU 时使用 CPU
print("model loaded", MODEL, f"{time.time()-t0:.1f}s")

t0 = time.time()
def encode(texts, bs=64):
    return st.encode(list(texts), batch_size=bs, show_progress_bar=False,
                     normalize_embeddings=True)
E_q_tr = encode(train["text_a"]); E_a_tr = encode(train["text_b"])
E_q_te = encode(test["text_a"]);  E_a_te = encode(test["text_b"])
print(f"encoded in {time.time()-t0:.1f}s | dim={E_q_tr.shape[1]}")


def feats(q, a):
    return np.hstack([q, a, np.abs(q - a), q * a]).astype(np.float32)

X_tr = feats(E_q_tr, E_a_tr)
X_te = feats(E_q_te, E_a_te)
np.savez_compressed(os.path.join(OUT, "st_embeddings.npz"),
                    X_train=X_tr, y_train=y_train, X_test=X_te, y_test=y_test)
print("feature dim:", X_tr.shape[1])

# ---------- 验证集选择 C ----------
X_fit, X_val, y_fit, y_val = train_test_split(
    X_tr, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
val_res = {}
best = None
for C in [0.1, 1.0, 10.0]:
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=C, random_state=SEED)
    lr.fit(X_fit, y_fit)
    p = lr.predict(X_val)
    mf1 = float(f1_score(y_val, p, average="macro"))
    val_res[f"C={C}"] = mf1
    print(f"C={C}: val_macroF1={mf1:.4f}")
    if best is None or mf1 > best[1]:
        best = (C, mf1)
best_C = best[0]
print("best C:", best_C)

# ---------- 全量重训 -> 测试 ----------
lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=best_C, random_state=SEED)
lr.fit(X_tr, y_train)
pred = lr.predict(X_te)
proba = lr.predict_proba(X_te)
res = evaluate(y_test, pred, proba, label_names)
res.update({
    "model": "SentenceTransformer all-MiniLM-L6-v2 (frozen) + LR",
    "embedding_dim": int(E_q_tr.shape[1]),
    "feature_dim": int(X_tr.shape[1]),
    "best_C": best_C,
    "val_selection": val_res,
})
print("ST+LR test:", res["accuracy"], res["macro_f1"])

save_predictions(test["uid"].to_numpy(),
                 (test["text_a"].to_numpy(), test["text_b"].to_numpy()),
                 y_test, pred, proba, label_names,
                 os.path.join(OUT, "predictions_st_lr.csv"))
save_metrics(res, os.path.join(OUT, "metrics_st_lr.json"))
print("saved")
