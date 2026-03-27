import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from systemB_unsupervised import fit_sig_maha_knn, score_windows


def load_split(sig_dir: Path, split: str, depth: int) -> tuple[np.ndarray, np.ndarray]:
    path = sig_dir / f"{split}_signatures_depth{depth}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing signatures file: {path}")
    data = np.load(path, allow_pickle=True)
    for k in ("X_sig", "labels"):
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}")
    x = np.asarray(data["X_sig"], dtype=float)
    y = np.asarray(data["labels"], dtype=int).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"{path} X_sig must be 2D, got {x.shape}")
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"{path} label length mismatch: X={x.shape[0]}, y={y.shape[0]}")
    return x, y


def percentile_rank(scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1)
    r = np.asarray(ref_scores, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.full(s.shape, np.nan, dtype=float)
    rs = np.sort(r)
    return np.searchsorted(rs, s, side="right") / float(rs.size)


def sharpen_tail(prob: np.ndarray, p0: float) -> np.ndarray:
    if not (0.0 <= p0 < 1.0):
        raise ValueError("p0 must satisfy 0 <= p0 < 1")
    denom = max(1.0 - p0, 1e-12)
    return np.clip((prob - p0) / denom, 0.0, 1.0)


def score_corpus_excluding_self(model, x_corpus: np.ndarray, k: int) -> np.ndarray:
    if not hasattr(model, "mahal_distance") or model.mahal_distance is None:
        raise ValueError("Model must be fitted before scoring corpus")
    if not hasattr(model, "knn") or model.knn is None:
        raise ValueError("Model has no fitted KNN object")

    md = model.mahal_distance
    n = x_corpus.shape[0]
    n_neighbors = max(2, min(n, int(k) + 1))
    sig_dim = x_corpus.shape[1]

    modified = (
        (x_corpus - md.mu)
        @ md.Vt.T
        @ np.diag(md.S ** (-1))
    )

    candidate_distances, train_indices = model.knn.kneighbors(
        modified,
        n_neighbors=n_neighbors,
        return_distance=True,
    )

    test_indices = np.arange(n).reshape(-1, 1)
    differences = model.signatures_train[train_indices] - x_corpus[test_indices]
    denominator = np.linalg.norm(differences, axis=-1)
    projector = np.identity(sig_dim) - md.Vt.T @ md.Vt
    numerator = np.linalg.norm(differences @ projector, axis=-1)
    rho = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    candidate_distances = np.asarray(candidate_distances, dtype=float)
    candidate_distances[denominator < md.zero_thres] = 0.0
    candidate_distances[rho > md.subspace_thres] = np.inf
    candidate_distances[train_indices == test_indices] = np.inf

    scores = np.min(candidate_distances, axis=1)
    scores[~np.isfinite(scores)] = np.nan
    return scores


def separation_stats(scores: np.ndarray, y: np.ndarray) -> dict:
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)

    normal = s[y == 0]
    bubble = s[y == 1]
    normal = normal[np.isfinite(normal)]
    bubble = bubble[np.isfinite(bubble)]
    if normal.size == 0 or bubble.size == 0:
        raise ValueError("Need both classes with finite scores for separation stats")

    nq = np.percentile(normal, [5, 25, 50, 75, 95])
    bq = np.percentile(bubble, [5, 25, 50, 75, 95])

    overlap_low = max(nq[0], bq[0])
    overlap_high = min(nq[-1], bq[-1])
    union_low = min(nq[0], bq[0])
    union_high = max(nq[-1], bq[-1])
    overlap = max(0.0, overlap_high - overlap_low)
    union = max(union_high - union_low, 1e-12)
    overlap_ratio = overlap / union

    return {
        "mean_score_normal": float(np.mean(normal)),
        "mean_score_bubble": float(np.mean(bubble)),
        "median_score_normal": float(np.median(normal)),
        "median_score_bubble": float(np.median(bubble)),
        "normal_q5_q95": (float(nq[0]), float(nq[-1])),
        "bubble_q5_q95": (float(bq[0]), float(bq[-1])),
        "percentile_overlap_ratio_q5_q95": float(overlap_ratio),
    }


def plot_score_hist(scores: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    normal = s[(y == 0) & np.isfinite(s)]
    bubble = s[(y == 1) & np.isfinite(s)]

    fig = plt.figure(figsize=(9, 5))
    bins = 50
    plt.hist(normal, bins=bins, alpha=0.6, density=True, label="y=0 normal")
    plt.hist(bubble, bins=bins, alpha=0.6, density=True, label="y=1 bubble")
    plt.xlabel("Unsupervised score (higher = more anomalous)")
    plt.ylabel("Density")
    plt.title("Test score distribution by class")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate System B unsupervised SigMahaKNN on synthetic signatures."
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--p0", type=float, default=0.9)
    parser.add_argument(
        "--sig_dir",
        type=str,
        default="data/processed/systemB_signatures",
        help="Directory with train/val/test_signatures_depth{d}.npz",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/systemB_unsupervised_synth",
    )
    parser.add_argument("--save_csv", action="store_true")
    args = parser.parse_args()

    sig_dir = Path(args.sig_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split(sig_dir=sig_dir, split="train", depth=args.depth)
    _, _ = load_split(sig_dir=sig_dir, split="val", depth=args.depth)
    x_test, y_test = load_split(sig_dir=sig_dir, split="test", depth=args.depth)

    x_corpus = x_train[y_train == 0]
    if x_corpus.shape[0] < max(10, args.k + 1):
        raise ValueError(
            f"Not enough normal samples in training corpus: {x_corpus.shape[0]}"
        )

    model = fit_sig_maha_knn(x_corpus=x_corpus, k=args.k)

    test_scores = score_windows(model=model, x_windows=x_test, n_neighbors=args.k)
    test_scores = np.asarray(test_scores, dtype=float).reshape(-1)
    finite_mask = np.isfinite(test_scores)
    if int(np.sum(finite_mask)) < 2:
        raise ValueError("Too few finite test scores for evaluation")

    y_eval = y_test[finite_mask]
    s_eval = test_scores[finite_mask]
    auc = float(roc_auc_score(y_eval, s_eval))
    auc_inv = float(roc_auc_score(y_eval, -s_eval))

    corpus_scores = score_corpus_excluding_self(model=model, x_corpus=x_corpus, k=args.k)
    corpus_scores = corpus_scores[np.isfinite(corpus_scores)]
    if corpus_scores.size == 0:
        raise ValueError("No finite corpus scores for calibration")

    p_rank = percentile_rank(scores=test_scores, ref_scores=corpus_scores)
    bubble_prob = sharpen_tail(p_rank, p0=args.p0)

    sep = separation_stats(scores=s_eval, y=y_eval)

    print("\n=== Unsupervised Synthetic Evaluation ===")
    print(f"depth={args.depth} | k={args.k}")
    print(f"train size={x_train.shape[0]} | test size={x_test.shape[0]}")
    print(f"normal corpus size={x_corpus.shape[0]}")
    print(f"ROC-AUC(test, score): {auc:.6f}")
    print(f"ROC-AUC(test, -score): {auc_inv:.6f}")
    if auc < 0.5:
        print("WARNING: score direction may be flipped (higher should mean more anomalous).")
    print(f"mean(score|normal): {sep['mean_score_normal']:.6f}")
    print(f"mean(score|bubble): {sep['mean_score_bubble']:.6f}")
    print(
        "percentile overlap (q5-q95 interval overlap ratio): "
        f"{sep['percentile_overlap_ratio_q5_q95']:.6f}"
    )
    print(f"normal q5/q95: {sep['normal_q5_q95']}")
    print(f"bubble q5/q95: {sep['bubble_q5_q95']}")

    hist_path = out_dir / f"test_score_hist_depth{args.depth}_k{args.k}.png"
    plot_score_hist(scores=test_scores, y=y_test, out_path=hist_path)
    print(f"Saved histogram: {hist_path.resolve()}")

    if args.save_csv:
        out = pd.DataFrame(
            {
                "label": y_test,
                "score": test_scores,
                "prob_percentile_vs_corpus": p_rank,
                "bubble_prob": bubble_prob,
            }
        )
        out_csv = out_dir / f"test_scores_depth{args.depth}_k{args.k}.csv"
        out.to_csv(out_csv, index=False)
        print(f"Saved CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
