import argparse
import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_split(sig_dir: Path, split_name: str, depth: int):
    path = sig_dir / f"{split_name}_signatures_depth{depth}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing signatures file: {path}")

    data = np.load(path)
    X_sig = data["X_sig"]
    labels = data["labels"]
    model_id = data["model_id"]
    return X_sig, labels, model_id


def eval_split(name: str, clf, X, y):
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print("Confusion matrix:\n", cm)

    return {"accuracy": acc, "roc_auc": auc, "confusion_matrix": cm.tolist()}


def main():
    parser = argparse.ArgumentParser(description="Train System B classifier on signature features.")
    parser.add_argument("--depth", type=int, default=3, help="Signature depth used to load features.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for logistic regression.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for optimizer.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/systemB",
        help="Directory (relative to project root) to save scaler/model/metadata.",
    )
    args = parser.parse_args()

    root = project_root()
    data_dir = root / "data"
    sig_dir = data_dir / "processed" / "systemB_signatures"
    save_dir = root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    depth = args.depth

    # Load splits
    X_train, y_train, _ = load_split(sig_dir, "train", depth)
    X_val, y_val, _ = load_split(sig_dir, "val", depth)
    X_test, y_test, _ = load_split(sig_dir, "test", depth)

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # Standardize features (fit on train, apply everywhere)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train logistic regression
    # NOTE: sklearn default penalty is L2; setting it explicitly is deprecated in newer versions.
    clf = LogisticRegression(
        solver="lbfgs",
        C=args.C,
        max_iter=args.max_iter,
    )
    clf.fit(X_train_s, y_train)

    # Evaluate
    metrics = {}
    metrics["train"] = eval_split("TRAIN", clf, X_train_s, y_train)
    metrics["val"] = eval_split("VAL", clf, X_val_s, y_val)
    metrics["test"] = eval_split("TEST", clf, X_test_s, y_test)

    y_test_pred = clf.predict(X_test_s)
    print("\nClassification report (TEST):")
    print(classification_report(y_test, y_test_pred))

    # Save artifacts
    scaler_path = save_dir / f"scaler_depth{depth}.joblib"
    model_path = save_dir / f"logreg_depth{depth}.joblib"
    meta_path = save_dir / f"metadata_depth{depth}.json"

    dump(scaler, scaler_path)
    dump(clf, model_path)

    metadata = {
        "system": "B",
        "features": "signature",
        "channels": ["time", "log_price_norm"],
        "signature_depth": depth,
        "model": {
            "type": "LogisticRegression",
            "solver": "lbfgs",
            "C": args.C,
            "max_iter": args.max_iter,
        },
        "data": {
            "signature_dir": str(sig_dir.relative_to(root)),
            "train_file": f"train_signatures_depth{depth}.npz",
            "val_file": f"val_signatures_depth{depth}.npz",
            "test_file": f"test_signatures_depth{depth}.npz",
            "X_dim": int(X_train.shape[1]),
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
        },
        "metrics": metrics,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved scaler to: {scaler_path}")
    print(f"Saved model  to: {model_path}")
    print(f"Saved meta   to: {meta_path}")


if __name__ == "__main__":
    main()
