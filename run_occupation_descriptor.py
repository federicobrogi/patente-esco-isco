"""Launcher della pipeline professionale ESCO, portabile su Azure."""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.occupation_descriptor_pipeline import MODELS


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def combine(outputs, destination):
    writer = None
    with destination.open("w", newline="", encoding="utf-8") as target:
        for source_path in outputs:
            with source_path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)


def main():
    parser = argparse.ArgumentParser(description="Matching brevetti -> professioni ESCO")
    parser.add_argument("--config", default="occupation_descriptor_config.yaml")
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--limit", type=int, help="Smoke test sui primi N brevetti")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = args.models or config["embedding"]["models"]
    output_dir = resolve(root, config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    esco_path = resolve(root, config["esco"]["path"])
    patents_path = resolve(root, config["patents"]["path"])
    gte_path = resolve(root, config["runtime"].get("gte_compatibility_path", "vendor_gte"))
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "models": [], "errors": []}
    completed_files = []
    for model in selected:
        command = [sys.executable, "-m", "src.occupation_descriptor_pipeline",
            "--esco", str(esco_path), "--patents", str(patents_path),
            "--output-dir", str(output_dir), "--models", model,
            "--id-col", config["patents"]["id_col"], "--title-col", config["patents"]["title_col"],
            "--abstract-col", config["patents"]["abstract_col"],
            "--first-claim-col", config["patents"]["first_claim_col"],
            "--patent-batch-size", str(config["matching"]["patent_batch_size"])]
        if args.limit:
            command += ["--limit", str(args.limit)]
        if args.overwrite:
            command.append("--overwrite")
        environment = os.environ.copy()
        if model == "gte_multilingual":
            if not gte_path.exists():
                manifest["errors"].append({"model": model, "error": f"Directory GTE mancante: {gte_path}"})
                continue
            environment["PYTHONPATH"] = str(gte_path) + os.pathsep + environment.get("PYTHONPATH", "")
        print(f"\nAvvio modello: {model}", flush=True)
        result = subprocess.run(command, cwd=root, env=environment)
        combined = output_dir / f"{model}_two_variants_top10.csv"
        if result.returncode == 0 and combined.exists():
            completed_files.append(combined)
            manifest["models"].append(model)
        else:
            manifest["errors"].append({"model": model, "returncode": result.returncode})
    if completed_files:
        combine(completed_files, output_dir / "all_models_two_variants_top10.csv")
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if manifest["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
