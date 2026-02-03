import argparse
from pathlib import Path
import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_paths_base(prices: np.ndarray, times: np.ndarray, eps: float):
    """
    Output paths (N, L, 2): [t_norm, log(S/S0)]
    """
    N, L = prices.shape
    if times.shape[0] != L:
        raise ValueError(f"times length {times.shape[0]} != prices length {L}")

    t_norm = times / times[-1]
    t_chan = np.broadcast_to(t_norm[None, :, None], (N, L, 1))

    S0 = np.maximum(prices[:, [0]], eps)
    P = np.maximum(prices, eps)
    x = np.log(P / S0)  # (N,L)

    return np.concatenate([t_chan, x[:, :, None]], axis=2)  # (N,L,2)


def lead_lag_1d(x: np.ndarray) -> np.ndarray:
    """
    For x length L:
      (x1,x1), (x2,x1), (x2,x2), ..., (xL, x_{L-1}), (xL,xL)
    Returns (2L-1, 2)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    L = x.shape[0]
    if L < 2:
        raise ValueError("Lead-lag requires L>=2")

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


def build_paths_leadlag(prices: np.ndarray, times: np.ndarray, eps: float):
    """
    Output lead-lag paths (N, 2L-1, 3): [t_norm, lead(x), lag(x)]
    where x = log(S/S0)
    """
    N, L = prices.shape
    if times.shape[0] != L:
        raise ValueError(f"times length {times.shape[0]} != prices length {L}")

    S0 = np.maximum(prices[:, [0]], eps)
    P = np.maximum(prices, eps)
    x = np.log(P / S0)  # (N,L)

    t_ll = np.linspace(0.0, 1.0, 2 * L - 1, dtype=float)
    out = np.empty((N, 2 * L - 1, 3), dtype=float)
    out[:, :, 0] = t_ll[None, :]

    for i in range(N):
        ll = lead_lag_1d(x[i])
        out[i, :, 1] = ll[:, 0]  # lead
        out[i, :, 2] = ll[:, 1]  # lag

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw/shifted_cev")
    ap.add_argument("--out_dir", type=str, default="data/processed/shifted_cev_systemB_paths")
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    root = project_root()
    raw_dir = root / args.raw_dir
    out_dir = root / args.out_dir
    ensure_dir(out_dir)

    for split in ["train", "val", "test"]:
        inp = raw_dir / f"{split}_paths.npz"
        if not inp.exists():
            raise FileNotFoundError(f"Missing {inp}")

        d = np.load(inp, allow_pickle=True)
        prices = d["prices"]
        times = d["times"]
        labels = d["labels"]
        params = d["params"]
        param_names = d["param_names"]

        # base
        paths_base = build_paths_base(prices, times, eps=args.eps)
        outp_base = out_dir / f"{split}_paths_for_signature.npz"
        np.savez_compressed(
            outp_base,
            paths=paths_base,
            labels=labels,
            params=params,
            param_names=param_names,
        )
        print(f"Saved {split} BASE paths => {outp_base} | {paths_base.shape}")

        # lead-lag
        paths_ll = build_paths_leadlag(prices, times, eps=args.eps)
        outp_ll = out_dir / f"{split}_ll_paths_for_signature.npz"
        np.savez_compressed(
            outp_ll,
            paths=paths_ll,
            labels=labels,
            params=params,
            param_names=param_names,
        )
        print(f"Saved {split} LEADLAG paths => {outp_ll} | {paths_ll.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
