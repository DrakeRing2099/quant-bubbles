from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.multiscale_xgb_common import (
        default_model_dir,
        feature_split_filename,
        load_feature_split,
        multiscale_feature_dir,
        project_root,
        require_xgboost,
        save_xgb_model,
    )
except ImportError:
    from multiscale_xgb_common import (
        default_model_dir,
        feature_split_filename,
        load_feature_split,
        multiscale_feature_dir,
        project_root,
        require_xgboost,
        save_xgb_model,
    )

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost classifier on multiscale logsignature features.")
    parser.add_argument("--dataset", required=True, choices=["cev", "shifted_cev", "sin"])
    parser.add_argument("--in_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_split(features_dir: Path, split: str, depth: int) -> tuple[np.ndarray, np.ndarray]:
    X, payload = load_feature_split(features_dir / feature_split_filename(split, depth))
    return X, payload["labels"]


def evaluate_block(clf, X: np.ndarray, y: np.ndarray) -> dict:
    pred = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def main() -> None:
    args = parse_args()
    xgb = require_xgboost()
    root = project_root()
    features_dir = root / args.in_dir if args.in_dir else multiscale_feature_dir(root, args.dataset)
    out_dir = default_model_dir(args.dataset, out_dir=args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(features_dir, "train", args.depth)
    X_val, y_val = load_split(features_dir, "val", args.depth)
    X_test, y_test = load_split(features_dir, "test", args.depth)

    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    metrics = {
        "train": evaluate_block(clf, X_train, y_train),
        "val": evaluate_block(clf, X_val, y_val),
        "test": evaluate_block(clf, X_test, y_test),
    }

    print("\n=== TRAIN ===")
    print(metrics["train"])
    print("\n=== VAL ===")
    print(metrics["val"])
    print("\n=== TEST ===")
    print(metrics["test"])
    print("\nClassification report (TEST):")
    print(classification_report(y_test, clf.predict(X_test)))

    model_path = out_dir / f"model_depth{args.depth}.joblib"
    save_xgb_model(model_path, clf)

    feature_meta_path = features_dir / f"metadata_depth{args.depth}.json"
    feature_meta = json.loads(feature_meta_path.read_text(encoding="utf-8")) if feature_meta_path.exists() else {}
    summary = {
        "model_family": "multiscale_xgb",
        "dataset": args.dataset,
        "depth": args.depth,
        "features_dir": str(features_dir.relative_to(root)),
        "model_path": str(model_path.relative_to(root)),
        "classifier": {
            "type": "XGBClassifier",
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "seed": args.seed,
        },
        "feature_metadata": feature_meta,
        "dims": {
            "X_dim": int(X_train.shape[1]),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
        },
        "metrics": metrics,
    }
    summary_json = out_dir / f"summary_depth{args.depth}.json"
    summary_csv = out_dir / f"summary_depth{args.depth}.csv"
    metadata_json = out_dir / f"metadata_depth{args.depth}.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "depth": args.depth,
                "accuracy": metrics["test"]["accuracy"],
                "roc_auc": metrics["test"]["roc_auc"],
                "train_accuracy": metrics["train"]["accuracy"],
                "val_accuracy": metrics["val"]["accuracy"],
                "test_accuracy": metrics["test"]["accuracy"],
            }
        ]
    ).to_csv(summary_csv, index=False)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved summary JSON to: {summary_json}")
    print(f"Saved summary CSV to: {summary_csv}")


if __name__ == "__main__":
    main()
