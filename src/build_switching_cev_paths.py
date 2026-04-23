"""
build_switching_cev_paths.py
============================
Builds System B base and lead-lag paths for the `switching_cev` dataset.

Key difference from other datasets
-----------------------------------
labels in switching_cev are per-timestep (shape N × L), not per-path.
When we slice a rolling window ending at step t, we use labels[:, t] as
the label for that window — i.e. the regime at the *end* of the window.

The saved .npz files match the format expected by
compute_systemB_features_all.py, with one extra key:
  labels_per_step : (N, n_steps+1) int8  — kept for downstream use

Usage
-----
    python build_switching_cev_paths.py
    python build_switching_cev_paths.py --in_dir data/raw/switching_cev \\
                                         --out_dir data/processed/switching_cev_systemB_paths
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_raw_split(raw_dir: Path, split: str):
    path = raw_dir / f"{split}_paths.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    data = np.load(path, allow_pickle=True)
    prices = np.asarray(data["prices"], dtype=np.float64)
    times  = np.asarray(data["times"],  dtype=np.float64).reshape(-1)
    # per-timestep labels: (N, L)
    labels = np.asarray(data["labels"], dtype=np.int8)
    return prices, times, labels


def normalize_times(times: np.ndarray) -> np.ndarray:
    span = float(times[-1] - times[0])
    if span <= 0.0:
        raise ValueError("Time grid must be strictly increasing")
    return (times - times[0]) / span


def build_base_paths(prices: np.ndarray, times: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(N, L) → (N, L, 2):  [t_norm, log(S/S0)]"""
    N, L = prices.shape
    t_norm = normalize_times(times)
    out = np.empty((N, L, 2), dtype=np.float64)
    out[:, :, 0] = t_norm[None, :]
    batch = np.maximum(prices, eps)
    s0    = np.maximum(batch[:, [0]], eps)
    out[:, :, 1] = np.log(batch / s0)
    return out


def lead_lag_1d(x: np.ndarray) -> np.ndarray:
    """1-D sequence of length L → (2L-1, 2) lead-lag embedding."""
    L = x.shape[0]
    out = np.empty((2 * L - 1, 2), dtype=np.float64)
    out[0] = [x[0], x[0]]
    idx = 1
    for k in range(1, L):
        out[idx]     = [x[k], x[k - 1]]
        out[idx + 1] = [x[k], x[k]]
        idx += 2
    return out


def build_lead_lag_paths(prices: np.ndarray, times: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(N, L) → (N, 2L-1, 3):  [t_norm, lead(log S/S0), lag(log S/S0)]"""
    N, L = prices.shape
    t_ll = np.linspace(0.0, 1.0, 2 * L - 1, dtype=np.float64)
    out  = np.empty((N, 2 * L - 1, 3), dtype=np.float64)
    out[:, :, 0] = t_ll[None, :]

    batch = np.maximum(prices, eps)
    s0    = np.maximum(batch[:, [0]], eps)
    log_prices = np.log(batch / s0)

    for i in range(N):
        ll = lead_lag_1d(log_prices[i])
        out[i, :, 1] = ll[:, 0]
        out[i, :, 2] = ll[:, 1]

    return out


def path_output_name(split: str, variant: str) -> str:
    if variant == "base":
        return f"{split}_paths_for_signature.npz"
    if variant == "ll":
        return f"{split}_ll_paths_for_signature.npz"
    raise ValueError(f"Unknown variant: {variant}")


def save_paths_file(
    path: Path,
    paths: np.ndarray,
    labels: np.ndarray,
    labels_per_step: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        paths=paths,
        labels=labels,                   # per-path end-step label (scalar per path)
        labels_per_step=labels_per_step, # full (N, L) label matrix
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build System B paths for switching_cev.")
    ap.add_argument("--in_dir",  type=str, default="data/raw/switching_cev")
    ap.add_argument("--out_dir", type=str, default="data/processed/switching_cev_systemB_paths")
    ap.add_argument("--eps",     type=float, default=1e-6)
    args = ap.parse_args()

    root    = project_root()
    raw_dir = root / args.in_dir
    out_dir = root / args.out_dir
    ensure_dir(out_dir)

    for split in ("train", "val", "test"):
        print(f"\nProcessing split: {split}")
        prices, times, labels_2d = load_raw_split(raw_dir, split)
        # labels_2d : (N, L)  — per-timestep

        # Per-path scalar label = label at the *last* step (end of path)
        labels_end = labels_2d[:, -1].astype(np.int8)  # (N,)

        # ---------- base paths ----------
        base_paths = build_base_paths(prices, times, eps=args.eps)
        base_out   = out_dir / path_output_name(split, "base")
        save_paths_file(base_out, base_paths, labels_end, labels_2d)
        print(f"  Saved base paths  -> {base_out} | {base_paths.shape}")

        # ---------- lead-lag paths ----------
        ll_paths = build_lead_lag_paths(prices, times, eps=args.eps)
        ll_out   = out_dir / path_output_name(split, "ll")
        save_paths_file(ll_out, ll_paths, labels_end, labels_2d)
        print(f"  Saved ll paths    -> {ll_out}   | {ll_paths.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
