"""Diagnose truncation and test same-budget answer packing on fixed QEvasion splits."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
from time import perf_counter

from huggingface_hub import hf_hub_download
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers

from run_experiments import Experiment, index_split, read_data, score, write_json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenize_pairs(frame, tokenizer, max_wordpieces, audit_file):
    questions = frame["question"].tolist()
    answers = frame["answer"].tolist()
    question_ids = tokenizer(questions, add_special_tokens=False, verbose=False)["input_ids"]
    answer_ids = tokenizer(answers, add_special_tokens=False, verbose=False)["input_ids"]
    special = tokenizer.num_special_tokens_to_add(pair=True)
    # Compare the manual prefix path to the unmodified trainer's native truncation.
    native_prefix = tokenizer(questions, answers, truncation="only_second", max_length=max_wordpieces)
    records = []
    prefixes = []
    head_tails = []
    for i, (question, answer) in enumerate(zip(question_ids, answer_ids, strict=True)):
        budget = max_wordpieces - len(question) - special
        if budget <= 0:
            raise ValueError(f"Question does not leave an answer budget for {frame.iloc[i]['id']}")
        truncated = len(answer) > budget
        prefix = answer[:budget]
        if truncated:
            tail = budget // 2
            mixed = answer[:budget - tail] + (answer[-tail:] if tail else [])
        else:
            mixed = answer
        assert len(mixed) == len(prefix)
        options = []
        for tokens in (prefix, mixed):
            options.append({
                "input_ids": tokenizer.build_inputs_with_special_tokens(question, tokens),
                "token_type_ids": tokenizer.create_token_type_ids_from_sequences(question, tokens),
            })
        assert options[0]["input_ids"] == native_prefix["input_ids"][i]
        assert options[0]["token_type_ids"] == native_prefix["token_type_ids"][i]
        prefixes.append(options[0])
        head_tails.append(options[1])
        row = frame.iloc[i]
        records.append({
            "id": row["id"], "split": row["split"], "gold": row["label"],
            "question_wordpieces": len(question), "answer_wordpieces": len(answer),
            "answer_wordpiece_budget": budget, "truncated": truncated,
        })
    with audit_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return {"prefix": prefixes, "prefix_and_suffix": head_tails}, np.array([r["truncated"] for r in records])


def by_truncation(gold, pred, labels, truncated):
    result = {}
    for key, mask in (("all", np.ones(len(gold), dtype=bool)), ("truncated", truncated), ("not_truncated", ~truncated)):
        record = score(gold[mask], pred[mask], labels)
        counts = Counter(pred[mask])
        record["prediction_counts"] = {label: counts[label] for label in labels}
        result[key] = record
    return result


def make_model(options, labels, device, local):
    torch.manual_seed(options["seed"])
    return AutoModelForSequenceClassification.from_pretrained(
        options["model"], revision=options["revision"], local_files_only=local,
        num_labels=len(labels), attn_implementation="eager",
    ).to(device)


def input_batch(tokenizer, packed, indices, device):
    tensors = tokenizer.pad([packed[i] for i in indices], padding=True, return_tensors="pt", verbose=False)
    return {key: val.to(device) for key, val in tensors.items()}


@torch.inference_mode()
def predict(model, tokenizer, packed, indices, device, batch_size, use_bfloat):
    model.eval()
    probs = []
    for start in range(0, len(indices), batch_size):
        batch = input_batch(tokenizer, packed, indices[start:start + batch_size], device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bfloat):
            logits = model(**batch).logits
        probs.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(probs)


def fit(exp, spec, opts, tokenizer, packed, train_idx, val_idx, labels, encoded, gold, device, local):
    model = make_model(opts, labels, device, local)
    counts = torch.bincount(encoded[train_idx], minlength=len(labels)).float()
    weight = counts.rsqrt() if spec["loss_weighting"] == "inverse_square_root" else counts.reciprocal()
    weight = (weight / weight.mean()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts["learning_rate"], weight_decay=opts["weight_decay"])
    num_epochs = opts["epochs"]["grouped"]
    batch_size = opts["batch_size"] if device == "cuda" else 12
    total_steps = num_epochs * math.ceil(len(train_idx) / batch_size)
    warmup = max(1, round(opts["warmup_fraction"] * total_steps))

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(.08, (total_steps - step) / (total_steps - warmup))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    use_bfloat = device == "cuda" and torch.cuda.is_bf16_supported()
    rng = np.random.default_rng(opts["seed"])
    history = []
    best = None
    state = None
    best_probs = None
    for epoch in range(1, num_epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            batch = input_batch(tokenizer, packed, ids, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bfloat):
                logits = model(**batch).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), encoded[ids].to(device), weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            losses.append(float(loss.detach()))
        probs = predict(model, tokenizer, packed, val_idx, device, batch_size, use_bfloat)
        validation = score(gold[val_idx], np.asarray(labels)[probs.argmax(axis=1)], labels)
        record = {"epoch": epoch, "mean_batch_loss": float(np.mean(losses)), "validation": validation}
        history.append(record)
        exp.log(f"{spec['key']} epoch={epoch} validation_macro_f1={validation['macro_f1']:.6f} "
                f"validation_clear_recall={validation['per_class']['Clear Reply']['recall']:.6f}")
        if best is None or validation["macro_f1"] > best["validation"]["macro_f1"]:
            best = record
            best_probs = probs
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    exp.add_selection(spec["key"], history, best)
    params = {
        **spec, "best_epoch": best["epoch"], "validation_macro_f1": best["validation"]["macro_f1"],
        "class_weights": dict(zip(labels, weight.cpu().tolist())), "learning_rate": opts["learning_rate"],
        "batch_size": batch_size, "max_wordpieces": opts["max_wordpieces"], "used_bfloat16": use_bfloat,
        "seed": opts["seed"], "train_count": len(train_idx), "validation_count": len(val_idx),
    }
    return best_probs, params, state, use_bfloat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = perf_counter()
    cfg_path = ROOT / "configs" / "source_truncation.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    options_path = ROOT / cfg["base_config"]
    opts = json.loads(options_path.read_text(encoding="utf-8"))
    base = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    random.seed(opts["seed"])
    np.random.seed(opts["seed"])
    torch.set_num_threads(base["threads"])
    torch.use_deterministic_algorithms(True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    exp = Experiment(args.output, base)
    try:
        weight_path = hf_hub_download(opts["model"], "model.safetensors", revision=opts["revision"], local_files_only=True)
        local = True
    except FileNotFoundError:
        weight_path = hf_hub_download(opts["model"], "model.safetensors", revision=opts["revision"])
        local = False
    tokenizer = AutoTokenizer.from_pretrained(opts["model"], revision=opts["revision"], local_files_only=local)
    data_path = args.data / cfg["source_data"]
    labels = base["source_labels"]
    frame = read_data(data_path, labels)
    train_idx, val_idx, test_idx = (index_split(frame, split) for split in ("train", "validation", "test"))
    gold = frame["label"].to_numpy()
    encoded = torch.tensor([labels.index(label) for label in gold], dtype=torch.long)
    packed, truncated = tokenize_pairs(frame, tokenizer, opts["max_wordpieces"], args.output / "tokenization_audit.csv")
    opts_results = {}
    slices = {"validation": {}}

    def run_candidate(spec):
        probs, params, state, bf16 = fit(exp, spec, opts, tokenizer, packed[spec["answer_policy"]],
                                         train_idx, val_idx, labels, encoded, gold, device, local)
        key = spec["key"]
        result = exp.evaluate(f"truncation_{key}_validation", frame.iloc[val_idx], gold[val_idx], probs, labels, params,
                              split="validation")
        prediction = np.asarray(labels)[probs.argmax(axis=1)]
        slices["validation"][key] = by_truncation(gold[val_idx], prediction, labels, truncated[val_idx])
        opts_results[key] = {"params": params, "state": state, "bf16": bf16}
        return result

    prefix, head_tail = [run_candidate(spec) for spec in cfg["candidates"]]
    if not (head_tail["macro_f1"] > prefix["macro_f1"] and
            head_tail["per_class"]["Clear Reply"]["recall"] > prefix["per_class"]["Clear Reply"]["recall"]):
        run_candidate(cfg["conditional_candidate"])
    chosen = max(opts_results, key=lambda key: opts_results[key]["params"]["validation_macro_f1"])
    selected = opts_results[chosen]
    chosen_param = selected["params"]
    exp.log(f"Validation selection frozen before test: {chosen}")
    model = make_model(opts, labels, device, local)
    model.load_state_dict(selected["state"])
    test_probs = predict(model, tokenizer, packed[chosen_param["answer_policy"]], test_idx, device,
                         chosen_param["batch_size"], selected["bf16"])
    exp.evaluate(f"truncation_{chosen}_test", frame.iloc[test_idx], gold[test_idx], test_probs, labels, chosen_param)
    test_pred = np.asarray(labels)[test_probs.argmax(axis=1)]
    slices["selected_test"] = {chosen: by_truncation(gold[test_idx], test_pred, labels, truncated[test_idx])}
    summary = {
        "selected": chosen, "selection_rule": cfg["selection_rule"], "candidates": list(opts_results),
        "conditional_candidate_triggered": cfg["conditional_candidate"]["key"] in opts_results,
        "native_prefix_equivalence": {"matched_all_rows": True, "row_count": len(frame)},
        **slices,
    }
    write_json(args.output / "slices.json", summary)
    write_json(args.output / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "base_weight_sha256": sha(Path(weight_path)), "code_sha256": sha(Path(__file__)),
        "source_config_sha256": sha(cfg_path), "base_config_sha256": sha(options_path), "data_sha256": sha(data_path),
        "elapsed_seconds": perf_counter() - started,
    })
    exp.log("Completed")


if __name__ == "__main__":
    main()
