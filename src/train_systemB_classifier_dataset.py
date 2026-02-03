import argparse
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_split(sig_dir: Path, split: str, variant: str, depth: int):
    p = sig_dir / f"{split}_{variant}_signatures_depth{depth}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    d = np.load(p, allow_pickle=True)
    return d["X_sig"], d["labels"]


def eval_block(name: str, clf, X, y):
    pred = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else None
    cm = confusion_matrix(y, pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"ROC-AUC:  {auc:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nReport:\n", classification_report(y, pred))
    return acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig_dir", type=str, default="data/processed/shifted_cev_systemB_signatures")
    ap.add_argument("--out_dir", type=str, default="models/shifted_cev_systemB")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--variant", type=str, default="base", choices=["base", "ll"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=1000)
    args = ap.parse_args()

    root = project_root()
    sig_dir = root / args.sig_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr = load_split(sig_dir, "train", args.variant, args.depth)
    Xva, yva = load_split(sig_dir, "val", args.variant, args.depth)
    Xte, yte = load_split(sig_dir, "test", args.variant, args.depth)

    print("Train:", Xtr.shape, "Val:", Xva.shape, "Test:", Xte.shape)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    clf = LogisticRegression(solver="lbfgs", C=args.C, max_iter=args.max_iter)
    clf.fit(Xtr_s, ytr)

    eval_block("TRAIN", clf, Xtr_s, ytr)
    eval_block("VAL", clf, Xva_s, yva)
    eval_block("TEST", clf, Xte_s, yte)

    dump(scaler, out_dir / f"scaler_{args.variant}_depth{args.depth}.joblib")
    dump(clf, out_dir / f"logreg_{args.variant}_depth{args.depth}.joblib")
    print(f"\nSaved scaler+model to: {out_dir}")


if __name__ == "__main__":
    main()
