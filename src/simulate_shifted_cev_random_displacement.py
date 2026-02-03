import argparse
import json
from pathlib import Path

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def simulate_path_shifted_cev(
    S0: float,
    sigma: float,
    beta: float,
    d: float,
    dt: float,
    n_steps: int,
    eps: float,
    rng: np.random.Generator,
):
    """
    Displaced/shifted CEV (paper-style):
      dS_t = sigma * (S_t + d)^beta dW_t,  S0>0

    Euler-Maruyama:
      S_{n+1} = S_n + sigma*(S_n + d)^beta * sqrt(dt) * Z
    """
    S = np.empty(n_steps + 1, dtype=float)
    S[0] = S0

    for n in range(n_steps):
        z = rng.normal()
        vol = sigma * ((S[n] + d) ** beta)
        S[n + 1] = S[n] + vol * np.sqrt(dt) * z

        # positivity floor (numerical, consistent with your current pipeline)
        if S[n + 1] <= 0.0 or not np.isfinite(S[n + 1]):
            S[n + 1] = eps

    return S


def sample_params_shifted_cev(
    n_paths: int,
    S0: float,
    sigma: float,
    a: float,
    A: float,
    D: float,
    rng: np.random.Generator,
):
    """
    Paper-style random displacement experiment:
      d ~ Unif[0, D]
      beta ~ Unif(a, 1] for half (true martingale => label 0)
      beta ~ Unif(1, A) for half (strict local martingale => label 1)

    Labels are EXACTLY by beta threshold:
      label = 1 iff beta > 1
    """
    # Balanced split
    n0 = n_paths // 2
    n1 = n_paths - n0

    beta0 = rng.uniform(a, 1.0, size=n0)      # <= 1
    beta1 = rng.uniform(np.nextafter(1.0, 2.0), A, size=n1)  # > 1

    betas = np.concatenate([beta0, beta1])
    labels = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    # displacement per path
    ds = rng.uniform(0.0, D, size=n_paths)

    # shuffle together
    idx = rng.permutation(n_paths)
    betas = betas[idx]
    ds = ds[idx]
    labels = labels[idx]

    # sigma fixed for now (like paper often does), but still save it
    sigmas = np.full(n_paths, float(sigma), dtype=float)
    S0s = np.full(n_paths, float(S0), dtype=float)

    params = np.stack([S0s, sigmas, betas, ds], axis=1)  # (N,4)
    param_names = np.array(["S0", "sigma", "beta", "d"], dtype=object)

    return params, param_names, labels


def generate_split(cfg: dict, split_name: str, seed: int):
    S0 = float(cfg["S0"])
    T = float(cfg["T"])
    n_steps = int(cfg["n_steps"])
    dt = T / n_steps
    eps = float(cfg["eps_floor"])

    # paper-style ranges
    a = float(cfg["beta_a"])
    A = float(cfg["beta_A"])
    D = float(cfg["disp_D"])
    sigma = float(cfg["sigma"])

    n_paths = int(cfg["n_paths"][split_name])

    rng = np.random.default_rng(seed)

    params, param_names, labels = sample_params_shifted_cev(
        n_paths=n_paths, S0=S0, sigma=sigma, a=a, A=A, D=D, rng=rng
    )

    prices = np.empty((n_paths, n_steps + 1), dtype=float)
    times = np.linspace(0.0, T, n_steps + 1, dtype=float)

    # simulate
    clipped_paths = 0
    for i in range(n_paths):
        S0_i, sigma_i, beta_i, d_i = params[i]
        path = simulate_path_shifted_cev(
            S0=S0_i,
            sigma=sigma_i,
            beta=beta_i,
            d=d_i,
            dt=dt,
            n_steps=n_steps,
            eps=eps,
            rng=rng,
        )
        prices[i, :] = path
        if np.any(path <= eps + 1e-15):
            clipped_paths += 1

    clip_frac = clipped_paths / n_paths

    return prices, times, labels, params, param_names, clip_frac


def save_split(out_dir: Path, split_name: str, prices, times, labels, params, param_names):
    ensure_dir(out_dir)
    out_path = out_dir / f"{split_name}_paths.npz"
    np.savez_compressed(
        out_path,
        prices=prices,
        times=times,
        labels=labels,
        params=params,
        param_names=param_names,
    )
    print(f"Saved {split_name} => {out_path} | prices={prices.shape}, labels={labels.shape}, params={params.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="data/raw/shifted_cev/sim_config.json")
    ap.add_argument("--out_dir", type=str, default="data/raw/shifted_cev")
    args = ap.parse_args()

    root = project_root()
    config_path = root / args.config
    out_dir = root / args.out_dir

    with open(config_path, "r") as f:
        cfg = json.load(f)

    base_seed = int(cfg.get("random_seed", 12345))

    for j, split in enumerate(["train", "val", "test"]):
        seed = base_seed + 1000 * j
        prices, times, labels, params, param_names, clip_frac = generate_split(cfg, split, seed)
        save_split(out_dir, split, prices, times, labels, params, param_names)
        print(f"[{split}] clipped_paths_frac ~= {clip_frac:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
