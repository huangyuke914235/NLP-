# -*- coding: utf-8 -*-
"""
统一实验框架：数据适配器 + 评估 + 输出
所有数据集通过 adapter config 统一为 text_a / text_b / label / group 四元组，
训练、评估、可视化代码在不同数据集之间完全复用。
"""
import json
import os
import random
import sys
import time

import numpy as np

# 项目内依赖目录：避免要求用户修改 Conda 环境或全局 PYTHONPATH。
_DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps")
if os.path.isdir(_DEPS) and _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

SEED = 42
CLASS_ORDER = ["Clear Reply", "Ambivalent Reply", "Clear Non-Reply"]  # 0/1/2 展示顺序


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_adapter(config_path: str):
    """按 adapter config 读取数据集，返回 (train_df, test_df, meta)。

    统一后字段：text_a, text_b, label(int), group, 以及原始标签 label_raw。
    """
    import pandas as pd
    from datasets import load_dataset, load_from_disk

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    source = cfg.get("local_path")
    if source and os.path.exists(source):
        ds = load_from_disk(source)
    else:
        load_kwargs = {"path": cfg["dataset_name"]}
        if cfg.get("config_name"):
            load_kwargs["name"] = cfg["config_name"]
        ds = load_dataset(**load_kwargs)

    lm = cfg["label_mapping"]  # 原始标签值 -> int
    frames = {}
    for split, target in [("train", "train"), ("test", "test"), ("validation", "test")]:
        if split not in ds or target in frames:
            continue
        df = ds[split].to_pandas()
        out = pd.DataFrame()
        out["text_a"] = df[cfg["text_a_column"]].astype(str)
        out["text_b"] = df[cfg["text_b_column"]].astype(str)
        raw = df[cfg["label_column"]]
        # 字符串标签按映射转 int；数值标签直接 int
        if pd.api.types.is_numeric_dtype(raw):
            out["label"] = raw.astype(int)
        else:
            out["label"] = raw.astype(str).map({k: v for k, v in lm.items()})
        out["label_raw"] = raw.astype(str)
        out["group"] = df[cfg["group_column"]].astype(str) if cfg.get("group_column") and cfg["group_column"] in df.columns else ""
        # 保留 id（若有）
        out["uid"] = df[cfg.get("id_column", "index")] if cfg.get("id_column") and cfg.get("id_column") in df.columns else np.arange(len(df))
        frames[target] = out

    meta = {
        "dataset_name": cfg.get("dataset_name"),
        "label_names": cfg["label_names"],
        "label_mapping": cfg["label_mapping"],
        "text_a_column": cfg["text_a_column"],
        "text_b_column": cfg["text_b_column"],
        "label_column": cfg["label_column"],
        "group_column": cfg.get("group_column"),
        "num_labels": len(cfg["label_names"]),
    }
    return frames.get("train"), frames.get("test"), meta


def make_pair_text(a, b, sep=" [SEP] "):
    """文本对拼接，与报告中的 question [SEP] answer 一致。"""
    return a + sep + b


def evaluate(y_true, y_pred, y_proba=None, label_names=None):
    """统一评估：Accuracy / Macro-P / Macro-R / Macro-F1 / per-class。"""
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 f1_score, precision_recall_fscore_support)

    res = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(label_names))), zero_division=0)
    res["macro_p"] = float(np.mean(p))
    res["macro_r"] = float(np.mean(r))
    res["per_class"] = {
        name: {"precision": float(p[i]), "recall": float(r[i]),
               "f1": float(f[i]), "support": int(support[i])}
        for i, name in enumerate(label_names)
    }
    res["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names)))).tolist()
    return res


def save_predictions(ids, texts, gold, pred, proba, label_names, path):
    """统一格式预测文件：id, gold, prediction, probability（各类别概率展开）。"""
    import pandas as pd
    out = pd.DataFrame({
        "id": ids,
        "question": texts[0],
        "answer": texts[1],
        "gold": [label_names[i] for i in gold],
        "prediction": [label_names[i] for i in pred],
    })
    for i, name in enumerate(label_names):
        out[f"prob_{name}"] = np.round(proba[:, i], 6)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_metrics(res: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


def timer():
    t0 = time.time()
    return lambda: round(time.time() - t0, 1)
