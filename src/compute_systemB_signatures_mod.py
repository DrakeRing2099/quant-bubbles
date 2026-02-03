import argparse
import json
from pathlib import Path

import numpy as np
import torch
import signatory


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_paths(paths_dir: Path, split_name: str):
    path = paths_dir / f"{split_name}_paths_for_signature.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing System B signature-paths file: {path}")

    data = np.load(path)
    paths = data["paths"]        # (N, L, 2) : [t_norm, log_price_over_S0]
    labels = data["labels"]      # (N,)
    model_id = data["model_id"]  # (N,)
    return paths, labels, model_id


def lead_lag_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    L = x.shape[0]
    if L < 2:
        raise ValueError("Lead-lag requires L >= 2")

    out = np.empty((2 * L - 1, 2), dtype=float)
    out[0, 0] = x[0]
    out[0, 1] = x[0]

    idx = 1
    for k in range(1, L):
        out[idx, 0] = x[k]
        out[idx, 1] = x[k - 1]
        idx += 1

        out[idx, 0] = x[k]
        out[idx, 1] = x[k]
        idx += 1

    return out


def apply_leadlag(paths: np.ndarray) -> np.ndarray:
    if paths.ndim != 3 or paths.shape[2] != 2:
        raise ValueError(f"Expected paths shape (N,L,2), got {paths.shape}")

    N, L, _ = paths.shape
    if L < 2:
        raise ValueError("Lead-lag requires L >= 2")

    t_ll = np.linspace(0.0, 1.0, 2 * L - 1, dtype=float)

    ll_paths = np.empty((N, 2 * L - 1, 3), dtype=float)
    ll_paths[:, :, 0] = t_ll[None, :]

    for i in range(N):
        x = paths[i, :, 1]
        ll = lead_lag_1d(x)
        ll_paths[i, :, 1] = ll[:, 0]
        ll_paths[i, :, 2] = ll[:, 1]

    return ll_paths


def compute_batch_features(paths_np: np.ndarray, depth: int, device: str, transform: str) -> np.ndarray:
    x = torch.from_numpy(paths_np).to(torch.float32).to(device)
    with torch.no_grad():
        if transform == "signature":
            feat = signatory.signature(x, depth=depth)
        elif transform == "logsignature":
            try:
                feat = signatory.logsignature(x, depth=depth)
            except TypeError:
                feat = signatory.logsignature(x, depth=depth, mode="words")
        else:
            raise ValueError(f"Unknown transform: {transform}")
    return feat.cpu().numpy()


def compute_features(paths: np.ndarray, depth: int, device: str, batch_size: int, transform: str) -> np.ndarray:
    N = paths.shape[0]
    feats = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = np.asarray(paths[start:end], dtype=np.float32)
        feats.append(compute_batch_features(batch, depth=depth, device=device, transform=transform))
    return np.vstack(feats)


def save_features(out_dir: Path, split_name: str, depth: int, tag: str, X, labels, model_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_name}_{tag}_depth{depth}.npz"
    # Save as X_sig to make it drop-in compatible with your existing training script style
    np.savez_compressed(out_path, X_sig=X, labels=labels, model_id=model_id)
    print(f"Saved {split_name} features to {out_path} with shape {X.shape}")


def save_metadata(out_dir: Path, depth: int, device: str, batch_size: int, variants_done: list[str]):
    meta_path = out_dir / f"metadata_depth{depth}.json"
    meta = {
        "system": "B",
        "input": "systemB_paths_for_signature.npz",
        "signature_depth": int(depth),
        "signatory_includes_level0": False,
        "compute": {"device": device, "batch_size": int(batch_size)},
        "variants": variants_done,
        "variant_definitions": {
            "base_sig": {"leadlag": False, "transform": "signature", "channels": ["t_norm", "log_price"]},
            "base_log": {"leadlag": False, "transform": "logsignature", "channels": ["t_norm", "log_price"]},
            "ll_sig":   {"leadlag": True,  "transform": "signature", "channels": ["t_norm", "lead(log_price)", "lag(log_price)"]},
            "ll_log":   {"leadlag": True,  "transform": "logsignature", "channels": ["t_norm", "lead(log_price)", "lag(log_price)"]},
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_path}")


def expand_variants(vlist: list[str]) -> list[str]:
    if len(vlist) == 1 and vlist[0] == "all":
        return ["base_sig", "base_log", "ll_sig", "ll_log"]
    return vlist


def main():
    parser = argparse.ArgumentParser(
        description="System B: compute multiple feature variants (signature/logsignature x base/leadlag) for comparison."
    )
    parser.add_argument("--depth", type=int, default=3, help="Signature/logsignature depth.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for signatory computation.")
    parser.add_argument(
        "--in_dir",
        type=str,
        default="data/processed/systemB_paths",
        help="Input directory containing {split}_paths_for_signature.npz",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/systemB_signatures_mod",
        help="Output directory for computed features (kept separate from frozen pipeline).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["all"],
        choices=["all", "base_sig", "base_log", "ll_sig", "ll_log"],
        help="Which variants to compute.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available. Use --device cpu.")

    variants = expand_variants(args.variants)

    root = project_root()
    in_dir = root / args.in_dir
    out_dir = root / args.out_dir

    variants_done = []

    for split_name in ["train", "val", "test"]:
        base_paths, labels, model_id = load_paths(in_dir, split_name)

        # Precompute lead-lag once per split if needed (expensive but only once)
        ll_paths = None
        if any(v.startswith("ll_") for v in variants):
            ll_paths = apply_leadlag(base_paths)

        for v in variants:
            if v == "base_sig":
                X = compute_features(base_paths, depth=args.depth, device=args.device, batch_size=args.batch_size, transform="signature")
                save_features(out_dir, split_name, args.depth, v, X, labels, model_id)

            elif v == "base_log":
                X = compute_features(base_paths, depth=args.depth, device=args.device, batch_size=args.batch_size, transform="logsignature")
                save_features(out_dir, split_name, args.depth, v, X, labels, model_id)

            elif v == "ll_sig":
                X = compute_features(ll_paths, depth=args.depth, device=args.device, batch_size=args.batch_size, transform="signature")
                save_features(out_dir, split_name, args.depth, v, X, labels, model_id)

            elif v == "ll_log":
                X = compute_features(ll_paths, depth=args.depth, device=args.device, batch_size=args.batch_size, transform="logsignature")
                save_features(out_dir, split_name, args.depth, v, X, labels, model_id)

            else:
                raise ValueError(f"Unknown variant: {v}")

        if split_name == "train":
            variants_done = variants[:]  # record once

    save_metadata(out_dir, depth=args.depth, device=args.device, batch_size=args.batch_size, variants_done=variants_done)


if __name__ == "__main__":
    main()
