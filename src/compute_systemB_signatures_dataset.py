import argparse
import json
from pathlib import Path

import numpy as np
import torch
import signatory


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def split_filename(split: str, variant: str) -> str:
    if variant == "base":
        return f"{split}_paths_for_signature.npz"
    if variant == "ll":
        return f"{split}_ll_paths_for_signature.npz"
    raise ValueError("variant must be 'base' or 'll'")


def load_split(paths_dir: Path, split: str, variant: str):
    p = paths_dir / split_filename(split, variant)
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    d = np.load(p, allow_pickle=True)
    return d["paths"], d["labels"], d["params"], d["param_names"]


def compute_signatures(paths: np.ndarray, depth: int, device: str, batch_size: int):
    N = paths.shape[0]
    sigs = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = torch.from_numpy(paths[start:end].astype(np.float32)).to(device)
        with torch.no_grad():
            s = signatory.signature(x, depth=depth)
        sigs.append(s.cpu().numpy())
    return np.vstack(sigs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_dir", type=str, default="data/processed/shifted_cev_systemB_paths")
    ap.add_argument("--out_dir", type=str, default="data/processed/shifted_cev_systemB_signatures")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--variant", type=str, default="base", choices=["base", "ll"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")

    root = project_root()
    paths_dir = root / args.paths_dir
    out_dir = root / args.out_dir
    ensure_dir(out_dir)

    meta = {
        "system": "B",
        "dataset": "shifted_cev_random_displacement",
        "variant": args.variant,
        "signature_depth": int(args.depth),
        "device": args.device,
        "batch_size": int(args.batch_size),
        "paths_dir": str(paths_dir.relative_to(root)),
        "signatory_includes_level0": False,
    }

    for split in ["train", "val", "test"]:
        paths, labels, params, param_names = load_split(paths_dir, split, args.variant)
        X_sig = compute_signatures(paths, depth=args.depth, device=args.device, batch_size=args.batch_size)

        outp = out_dir / f"{split}_{args.variant}_signatures_depth{args.depth}.npz"
        np.savez_compressed(
            outp,
            X_sig=X_sig,
            labels=labels,
            params=params,
            param_names=param_names,
        )
        print(f"Saved {split} signatures => {outp} | X_sig={X_sig.shape}")

    with open(out_dir / f"metadata_{args.variant}_depth{args.depth}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
