from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

try:
    from src.systemB_pipeline_common import DATASET_SPECS, project_root
except ImportError:
    from systemB_pipeline_common import DATASET_SPECS, project_root


VARIANT_MAP = {
    ("base", "sig"): "base_sig",
    ("base", "log"): "base_log",
    ("ll", "sig"): "ll_sig",
    ("ll", "log"): "ll_log",
}

FEATURE_LABELS = {"sig": "signature", "log": "logsignature"}
PATH_LABELS = {"base": "base", "ll": "lead-lag"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full System B experiment pipeline.")
    parser.add_argument("--dataset", choices=["cev", "shifted_cev", "sin"])
    parser.add_argument("--path", choices=["base", "ll"])
    parser.add_argument("--feature", choices=["sig", "log"])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--run_all", action="store_true")
    return parser.parse_args()


def variant_for(path_kind: str, feature_kind: str) -> str:
    return VARIANT_MAP[(path_kind, feature_kind)]


def model_dir_for(dataset: str) -> str:
    return "models/systemB_variants" if dataset == "cev" else f"models/{dataset}_systemB_variants"


def run_command(command: list[str], root: Path) -> None:
    subprocess.run(command, cwd=root, check=True)


def load_metrics(summary_path: Path, variant: str) -> dict:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return summary["variants"][variant]["metrics"]["test"]


def print_result(result: dict) -> None:
    print("\n=== Experiment Result ===")
    print(f"Dataset: {result['dataset']}")
    print(f"Path:    {PATH_LABELS[result['path']]}")
    print(f"Feature: {FEATURE_LABELS[result['feature']]}")
    print(f"Depth:   {result['depth']}")
    print(f"Variant: {result['variant']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    if result["roc_auc"] is None:
        print("ROC-AUC:  n/a")
    else:
        print(f"ROC-AUC:  {result['roc_auc']:.4f}")
    print("Confusion matrix:")
    for row in result["confusion_matrix"]:
        print(row)


def run_experiment(
    dataset: str,
    path_kind: str,
    feature_kind: str,
    depth: int,
    device: str,
    batch_size: int,
) -> dict:
    root = project_root()
    spec = DATASET_SPECS[dataset]
    variant = variant_for(path_kind, feature_kind)
    paths_dir = spec.paths_dir
    features_dir = spec.features_dir
    save_dir = model_dir_for(dataset)

    run_command(
        [
            sys.executable,
            "src/build_systemB_paths_all.py",
            "--dataset",
            dataset,
            "--in_dir",
            spec.raw_dir,
            "--out_dir",
            paths_dir,
            "--depth",
            str(depth),
            "--device",
            device,
            "--batch_size",
            str(batch_size),
        ],
        root,
    )
    run_command(
        [
            sys.executable,
            "src/compute_systemB_features_all.py",
            "--dataset",
            dataset,
            "--in_dir",
            paths_dir,
            "--out_dir",
            features_dir,
            "--depth",
            str(depth),
            "--device",
            device,
            "--batch_size",
            str(batch_size),
            "--variants",
            variant,
        ],
        root,
    )
    run_command(
        [
            sys.executable,
            "src/train_systemB_classifier_variants.py",
            "--depth",
            str(depth),
            "--sig_dir",
            features_dir,
            "--save_dir",
            save_dir,
            "--variants",
            variant,
        ],
        root,
    )

    metrics = load_metrics(root / save_dir / f"summary_depth{depth}.json", variant)
    result = {
        "dataset": dataset,
        "path": path_kind,
        "feature": feature_kind,
        "depth": depth,
        "variant": variant,
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    print_result(result)
    return result


def write_summary(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "path", "feature", "depth", "accuracy", "roc_auc"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def run_all(depth: int, device: str, batch_size: int) -> list[dict]:
    rows = []
    for dataset in ("cev", "shifted_cev", "sin"):
        for path_kind in ("base", "ll"):
            for feature_kind in ("sig", "log"):
                rows.append(run_experiment(dataset, path_kind, feature_kind, depth, device, batch_size))
    summary_path = project_root() / "outputs" / "systemB_experiments_summary.csv"
    write_summary(rows, summary_path)
    print(f"\nSaved experiment summary -> {summary_path}")
    return rows


def main() -> None:
    args = parse_args()
    if args.run_all:
        run_all(args.depth, args.device, args.batch_size)
        return
    if None in (args.dataset, args.path, args.feature):
        raise SystemExit("--dataset, --path, and --feature are required unless --run_all is used.")
    run_experiment(args.dataset, args.path, args.feature, args.depth, args.device, args.batch_size)


if __name__ == "__main__":
    main()
