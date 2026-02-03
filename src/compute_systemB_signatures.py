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
    paths = data["paths"]        # (N, L, d)
    labels = data["labels"]      # (N,)
    model_id = data["model_id"]  # (N,)
    return paths, labels, model_id


def compute_signature_batch(paths_np: np.ndarray, depth: int, device: str) -> np.ndarray:
    """
    paths_np: (B, L, d) numpy float
    returns: (B, Dsig) numpy float
    """
    x = torch.from_numpy(paths_np).to(torch.float32).to(device)
    with torch.no_grad():
        sig = signatory.signature(x, depth=depth)
    return sig.cpu().numpy()


def compute_signatures(paths: np.ndarray, depth: int, device: str, batch_size: int) -> np.ndarray:
    N = paths.shape[0]
    sigs = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sig_batch = compute_signature_batch(paths[start:end], depth=depth, device=device)
        sigs.append(sig_batch)
    return np.vstack(sigs)


def save_signatures(sig_dir: Path, split_name: str, depth: int, X_sig, labels, model_id):
    sig_dir.mkdir(parents=True, exist_ok=True)
    out_path = sig_dir / f"{split_name}_signatures_depth{depth}.npz"
    np.savez_compressed(out_path, X_sig=X_sig, labels=labels, model_id=model_id)
    print(f"Saved {split_name} signatures (depth {depth}) to {out_path} with shape {X_sig.shape}")


def save_metadata(sig_dir: Path, depth: int, device: str, batch_size: int):
    meta_path = sig_dir / f"metadata_depth{depth}.json"
    meta = {
        "system": "B",
        "features": "signature",
        "signature_depth": depth,
        "signatory_includes_level0": False,  # signatory returns levels 1..depth
        "expected_dim_for_d2": (2**1 + 2**2 + 2**3) if depth == 3 else None,
        "compute": {"device": device, "batch_size": batch_size},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute System B signature features from signature-ready paths.")
    parser.add_argument("--depth", type=int, default=3, help="Signature depth.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for signature computation.")
    args = parser.parse_args()

    root = project_root()
    data_dir = root / "data"
    paths_dir = data_dir / "processed" / "systemB_paths"
    sig_dir = data_dir / "processed" / "systemB_signatures"

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device=cuda but CUDA is not available. Use --device cpu.")

    depth = args.depth

    for split_name in ["train", "val", "test"]:
        paths, labels, model_id = load_paths(paths_dir, split_name)
        X_sig = compute_signatures(paths, depth=depth, device=device, batch_size=args.batch_size)
        save_signatures(sig_dir, split_name, depth, X_sig, labels, model_id)

    save_metadata(sig_dir, depth=depth, device=device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
