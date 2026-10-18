"""Check the current environment and cached model without running experiments.

Run from code/: python 10_check_environment.py
The default model check is offline. Use --allow-download only when the pinned
model revision is not already cached. The JSON records this check's environment;
it is not a historical lock file or evidence of all experiment outputs matching.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone, timedelta

BASE = Path(__file__).resolve().parent
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, default=BASE / "outputs/environment_verified.json")
    args = parser.parse_args()
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import common

    result = {
        "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "purpose": "Current environment preflight; not the historical experiment lock",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "common_module": common.__file__,
        "path_priority": sys.path[:6],
        "model_dependencies_directory": str(BASE / "model_deps"),
        "legacy_dependencies_directory": str(BASE / "deps"),
        "modules": {},
        "model_check": {},
    }
    for name in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "datasets", "torch",
                 "sentence_transformers", "transformers", "tokenizers", "safetensors", "huggingface_hub", "regex"]:
        try:
            module = importlib.import_module(name)
            result["modules"][name] = {
                "version": getattr(module, "__version__", None),
                "loaded_from": getattr(module, "__file__", None),
            }
        except Exception as exc:
            result["modules"][name] = {"error": repr(exc)}
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu",
                                    local_files_only=not args.allow_download)
        result["model_check"] = {
            "success": True, "model": MODEL, "revision": MODEL_REVISION,
            "device": str(model.device), "offline": not args.allow_download,
            "embedding_dim": model.get_sentence_embedding_dimension(),
            "max_seq_length": model.max_seq_length,
            "note": "Loaded model only; did not regenerate embeddings or train classifiers.",
        }
    except Exception as exc:
        result["model_check"] = {"success": False, "error": repr(exc)}
    result["success"] = all("error" not in record for record in result["modules"].values()) and result["model_check"].get("success", False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("ENVIRONMENT_CHECK", "PASS" if result["success"] else "FAIL", str(args.output))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
