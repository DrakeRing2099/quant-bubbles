import argparse
import json
from pathlib import Path

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sample_sigma1_balanced(cfg, n_paths, rng):
    """
    Paper-style balanced sampling:
      half sigma1 ~ U[0, 1/3]   -> true martingale label 0
      half sigma1 ~ U(1/3, 1]   -> strict local martingale label 1
    """
    lo_tm = float(cfg["sigma1_lo_tm"])
    hi_tm = float(cfg["sigma1_hi_tm"])
    lo_slm = float(cfg["sigma1_lo_slm"])
    hi_slm = float(cfg["sigma1_hi_slm"])

    n0 = n_paths // 2
    n1 = n_paths - n0

    s1_tm = rng.uniform(lo_tm, hi_tm, size=n0)

    # ensure > 1/3 for SLM half (avoid exactly 1/3)
    lo = np.nextafter(lo_slm, hi_slm)
    s1_slm = rng.uniform(lo, hi_slm, size=n1)

    sigma1 = np.concatenate([s1_tm, s1_slm])
    labels = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    idx = rng.permutation(n_paths)
    return sigma1[idx], labels[idx]


def simulate_sin_paths(cfg, split_name: str, seed: int):
    Y0 = float(cfg["Y0"])
    v0 = float(cfg["v0"])
    T = float(cfg["T"])
    n_steps = int(cfg["n_steps"])
    dt = T / n_steps

    alpha = float(cfg["alpha"])
    kappa = float(cfg["kappa"])
    L = float(cfg["L"])

    sigma2 = float(cfg["sigma2"])
    a1 = float(cfg["a1"])
    a2 = float(cfg["a2"])

    n_paths = int(cfg["n_paths"][split_name])

    rng = np.random.default_rng(seed)

    # sample sigma1 + labels (paper-balanced)
    sigma1, labels = sample_sigma1_balanced(cfg, n_paths, rng)

    times = np.linspace(0.0, T, n_steps + 1, dtype=float)
    prices = np.empty((n_paths, n_steps + 1), dtype=float)

    # store params per path for debugging/repro
    # [Y0, v0, alpha, kappa, L, a1, a2, sigma1, sigma2]
    params = np.empty((n_paths, 9), dtype=float)
    param_names = np.array(
        ["Y0", "v0", "alpha", "kappa", "L", "a1", "a2", "sigma1", "sigma2"],
        dtype=object,
    )

    # We simulate with a positivity-stable scheme:
    # - When kappa == 0, v follows geometric BM exactly:
    #     v_{t+dt} = v_t * exp( a1*sqrt(dt) z1 + a2*sqrt(dt) z2 - 0.5*(a1^2+a2^2) dt )
    # - Then Y is simulated in log form:
    #     logY_{t+dt} = logY_t + v_t^alpha*(sigma1*sqrt(dt) z1 + sigma2*sqrt(dt) z2)
    #                 - 0.5 * v_t^{2alpha}*(sigma1^2+sigma2^2) dt
    #
    # This keeps Y positive and avoids eps flooring.

    for i in range(n_paths):
        s1 = float(sigma1[i])

        v = v0
        logY = np.log(Y0)

        prices[i, 0] = Y0

        for n in range(n_steps):
            z1 = rng.normal()
            z2 = rng.normal()

            if kappa == 0.0:
                # exact geometric update
                v = v * np.exp(
                    (a1 * np.sqrt(dt) * z1) + (a2 * np.sqrt(dt) * z2)
                    - 0.5 * (a1 * a1 + a2 * a2) * dt
                )
            else:
                # fallback Euler (not used in paper config, but kept for completeness)
                v = v + (a1 * v * np.sqrt(dt) * z1) + (a2 * v * np.sqrt(dt) * z2) + kappa * (L - v) * dt
                if v <= 1e-12 or not np.isfinite(v):
                    v = 1e-12

            v_a = v ** alpha
            incr = v_a * (s1 * np.sqrt(dt) * z1 + sigma2 * np.sqrt(dt) * z2)
            drift = -0.5 * (v_a * v_a) * (s1 * s1 + sigma2 * sigma2) * dt
            logY = logY + incr + drift

            prices[i, n + 1] = float(np.exp(logY))

        params[i, :] = np.array([Y0, v0, alpha, kappa, L, a1, a2, s1, sigma2], dtype=float)

    return prices, times, labels, params, param_names


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
    print(f"Saved {split_name} => {out_path} | prices={prices.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="data/raw/sin/sim_config.json")
    ap.add_argument("--out_dir", type=str, default="data/raw/sin")
    args = ap.parse_args()

    root = project_root()
    config_path = root / args.config
    out_dir = root / args.out_dir

    with open(config_path, "r") as f:
        cfg = json.load(f)

    base_seed = int(cfg.get("random_seed", 12345))

    for j, split in enumerate(["train", "val", "test"]):
        seed = base_seed + 1000 * j
        prices, times, labels, params, param_names = simulate_sin_paths(cfg, split, seed)
        save_split(out_dir, split, prices, times, labels, params, param_names)

    print("Done.")


if __name__ == "__main__":
    main()
