import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_variant_split(sig_dir: Path, split: str, variant: str, depth: int):
    """
    Expects files saved by compute_systemB_signatures_mod.py:
      {split}_{variant}_depth{depth}.npz
    with keys: X_sig, labels, and optionally model_id
    """
    p = sig_dir / f"{split}_{variant}_depth{depth}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing features file: {p}")
    d = np.load(p, allow_pickle=True)
    X = d["X_sig"]
    y = d["labels"]
    mid = d["model_id"] if "model_id" in d.files else None
    return X, y, mid


def eval_split(name: str, clf, X, y):
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    # ROC-AUC only defined if both classes exist
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, y_proba)
    else:
        auc = None

    return {
        "accuracy": float(acc),
        "roc_auc": None if auc is None else float(auc),
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train & compare System B classifiers across feature variants.")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--sig_dir",
        type=str,
        default="data/processed/systemB_signatures_mod",
        help="Directory containing {split}_{variant}_depth{depth}.npz",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/systemB_variants",
        help="Where to save per-variant scalers/models and summary.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base_sig", "base_log", "ll_sig", "ll_log"],
        help="Which variants to train (must match filenames).",
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    root = project_root()
    sig_dir = root / args.sig_dir
    save_dir = root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    depth = args.depth
    variants = args.variants

    rows = []
    summary = {
        "system": "B",
        "depth": depth,
        "sig_dir": str(sig_dir.relative_to(root)),
        "model": {"type": "LogisticRegression", "solver": "lbfgs", "C": args.C, "max_iter": args.max_iter},
        "variants": {},
    }

    for v in variants:
        # Load
        Xtr, ytr, midtr = load_variant_split(sig_dir, "train", v, depth)
        Xva, yva, midva = load_variant_split(sig_dir, "val", v, depth)
        Xte, yte, midte = load_variant_split(sig_dir, "test", v, depth)

        print(f"\n==============================")
        print(f"Variant: {v} | depth={depth}")
        print("Train:", Xtr.shape, "Val:", Xva.shape, "Test:", Xte.shape)

        # Scale
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xte)

        # Train
        clf = LogisticRegression(solver="lbfgs", C=args.C, max_iter=args.max_iter)
        clf.fit(Xtr_s, ytr)

        # Eval
        m_tr = eval_split("TRAIN", clf, Xtr_s, ytr)
        m_va = eval_split("VAL", clf, Xva_s, yva)
        m_te = eval_split("TEST", clf, Xte_s, yte)

        print("TRAIN:", m_tr)
        print("VAL  :", m_va)
        print("TEST :", m_te)

        print("\nClassification report (TEST):")
        print(classification_report(yte, clf.predict(Xte_s)))

        # Save artifacts per variant
        v_dir = save_dir / f"{v}_depth{depth}"
        v_dir.mkdir(parents=True, exist_ok=True)
        dump(scaler, v_dir / "scaler.joblib")
        dump(clf, v_dir / "model.joblib")

        # Record
        summary["variants"][v] = {
            "dims": {"X_dim": int(Xtr.shape[1]), "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte))},
            "metrics": {"train": m_tr, "val": m_va, "test": m_te},
            "artifacts": {
                "scaler": str((v_dir / "scaler.joblib").relative_to(root)),
                "model": str((v_dir / "model.joblib").relative_to(root)),
            },
        }

        rows.append({
            "variant": v,
            "depth": depth,
            "X_dim": int(Xtr.shape[1]),
            "train_acc": m_tr["accuracy"],
            "train_auc": m_tr["roc_auc"],
            "val_acc": m_va["accuracy"],
            "val_auc": m_va["roc_auc"],
            "test_acc": m_te["accuracy"],
            "test_auc": m_te["roc_auc"],
        })

    # Save combined summary
    summary_path = save_dir / f"summary_depth{depth}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary JSON to: {summary_path}")

    df = pd.DataFrame(rows).sort_values(by=["test_auc", "test_acc"], ascending=False)
    csv_path = save_dir / f"summary_depth{depth}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to: {csv_path}")

    # Also print the ranking
    print("\n=== Variant ranking (by test_auc then test_acc) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
