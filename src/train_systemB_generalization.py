import numpy as np
from pathlib import Path 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SIG_DIR = DATA_DIR / "processed" / "systemB_signatures"

DEPTH = 3

def load_split(split_name: str, depth: int = DEPTH):
    path = SIG_DIR / f"{split_name}_signatures_depth{depth}.npz"
    data = np.load(path)
    X_sig = data["X_sig"]
    labels = data["labels"]
    model_id = data["model_id"]
    return X_sig, labels, model_id

def eval_split(name, clf, X, y):
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print("Confusion matrix:\n", cm)

    return acc, auc, cm


def main():
    # Load full splits
    X_train, y_train, mid_train = load_split("train", depth=DEPTH)
    X_val, y_val, mid_val = load_split("val", depth=DEPTH)
    X_test, y_test, mid_test = load_split("test", depth=DEPTH)

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # Restricting training model to not see the 4th one
    mask_train = mid_train != 4
    X_train_r = X_train[mask_train]
    y_train_r = y_train[mask_train]
    mid_train_r = mid_train[mask_train]

    print(f"Restricted train size: {X_train_r.shape[0]}")

    # standardise features
    scaler = StandardScaler()
    X_train_r_scaled = scaler.fit_transform(X_train_r)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Also scale the model id 4 features
    mask_test_model4 = mid_test == 4
    X_test_model4 = X_test[mask_test_model4]
    y_test_model4 = y_test[mask_test_model4]
    X_test_model4_scaled = scaler.transform(X_test_model4)

    print(f"Test paths with model_id 4 (unseen bubble regime): {X_test_model4.shape[0]}")

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )
    clf.fit(X_train_r_scaled, y_train_r)

    # Evaluate on full TRAIN/VAL/TEST
    eval_split("TRAIN (restricted)", clf, X_train_r_scaled, y_train_r)
    eval_split("VAL (all models)", clf, X_val_scaled, y_val)
    eval_split("TEST (all models)", clf, X_test_scaled, y_test)

    # Evaluation on remaining

    y_pred_4 = clf.predict(X_test_model4_scaled)
    y_proba_4 = clf.predict_proba(X_test_model4_scaled)[:, 1]

    acc_4 = accuracy_score(y_test_model4, y_pred_4)
    auc_4 = roc_auc_score(y_test_model4, y_proba_4)
    cm_4 = confusion_matrix(y_test_model4, y_pred_4)

    print("\n=== TEST on unseen bubble regime (model_id = 4) ===")
    print(f"Accuracy: {acc_4:.4f}")
    print(f"ROC-AUC:  {auc_4:.4f}")
    print("Confusion matrix:\n", cm_4)
    print("\nClassification report (model_id 4 only):")
    print(classification_report(y_test_model4, y_pred_4))


if __name__ == "__main__":
    main()

