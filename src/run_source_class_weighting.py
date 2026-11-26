"""Compare two source loss weightings with the same head-tail inputs and training budget."""
import argparse
import json
import os
from pathlib import Path
import platform
import random
from time import perf_counter

from huggingface_hub import hf_hub_download
import numpy as np
import torch
from transformers import AutoTokenizer
import transformers

from run_experiments import Experiment, index_split, read_data, write_json
from run_source_truncation import ROOT, by_truncation, fit, sha, tokenize_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    started = perf_counter()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config_path = ROOT / "configs" / "source_class_weighting.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    truncation_path = ROOT / config["truncation_config"]
    truncation = json.loads(truncation_path.read_text(encoding="utf-8"))
    options_path = ROOT / truncation["base_config"]
    options = json.loads(options_path.read_text(encoding="utf-8"))
    experiment_path = ROOT / "configs" / "experiment.json"
    base = json.loads(experiment_path.read_text(encoding="utf-8"))
    candidate = next(spec for spec in truncation["candidates"] if spec["key"] == config["candidate_key"])
    torch.set_num_threads(base["threads"])
    torch.use_deterministic_algorithms(True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    exp = Experiment(args.output, base)
    try:
        weight_path = hf_hub_download(options["model"], "model.safetensors", revision=options["revision"],
                                      local_files_only=True)
        local = True
    except FileNotFoundError:
        weight_path = hf_hub_download(options["model"], "model.safetensors", revision=options["revision"])
        local = False
    tokenizer = AutoTokenizer.from_pretrained(options["model"], revision=options["revision"], local_files_only=local)
    data_path = args.data / truncation["source_data"]
    labels = base["source_labels"]
    frame = read_data(data_path, labels)
    frame = frame[frame["split"].isin(("train", "validation"))].reset_index(drop=True)
    train_idx, val_idx = (index_split(frame, split) for split in ("train", "validation"))
    gold = frame["label"].to_numpy()
    encoded = torch.tensor([labels.index(label) for label in gold], dtype=torch.long)
    packed, truncated = tokenize_pairs(frame, tokenizer, options["max_wordpieces"], args.output / "tokenization_audit.csv")
    packed_pairs = packed[candidate["answer_policy"]]
    exp.log(f"Head-tail inputs kept fixed: train={len(train_idx)} validation={len(val_idx)} device={device}")
    slices = {}
    for seed in config["seeds"]:
        for weighting in config["weightings"]:
            random.seed(seed)
            np.random.seed(seed)
            key = f"weighting_{seed}_{weighting}"
            spec = {**candidate, "key": key, "loss_weighting": weighting}
            probs, params, state, _ = fit(
                exp, spec, {**options, "seed": seed}, tokenizer, packed_pairs,
                train_idx, val_idx, labels, encoded, gold, device, local,
            )
            del state
            exp.evaluate(key, frame.iloc[val_idx], gold[val_idx], probs, labels, params, split="validation")
            prediction = np.asarray(labels)[probs.argmax(axis=1)]
            slices[key] = by_truncation(gold[val_idx], prediction, labels, truncated[val_idx])
            write_json(args.output / "slices.json", slices)

    write_json(args.output / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "selection_rule": config["selection_rule"], "base_weight_sha256": sha(Path(weight_path)),
        "data_sha256": sha(data_path),
        "input_code_and_config_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
            for path in (
                Path(__file__), Path(__file__).with_name("run_source_truncation.py"),
                Path(__file__).with_name("run_experiments.py"),
                config_path, truncation_path, options_path, experiment_path,
            )
        },
        "elapsed_seconds": perf_counter() - started,
    })
    exp.log("Completed")


if __name__ == "__main__":
    main()
