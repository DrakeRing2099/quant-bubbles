"""
simulate_switching_cev.py
=========================
Generates the `switching_cev` dataset: regime-switching CEV paths where
normal (γ₁ ≤ 1) and bubble (γ₁ > 1) regimes alternate according to a
Poisson process.

Saved format (per split):  data/raw/switching_cev/{split}_paths.npz
Keys
----
prices      : (N, n_steps+1)  float64   — simulated price paths
times       : (n_steps+1,)    float64   — time grid [0, T]
labels      : (N, n_steps+1)  int8      — per-timestep regime label
                                          0 = normal (true martingale, γ₁ ≤ 1)
                                          1 = bubble  (strict local martingale, γ₁ > 1)
gamma1_path : (N, n_steps+1)  float32   — actual γ₁ used at each step (for inspection)
params      : (N, 4)          float32   — [gamma0, lambda_switch, initial_gamma1, initial_label]
param_names : (4,)            object    — column names for params

Windowed-label convention (used by downstream training)
-------------------------------------------------------
When building rolling windows of length W ending at step t, the label
for that window is labels[t] — i.e. the regime at the *end* of the window.
This matches the existing pipeline's approach of stamping the window with
its end-date.

Usage
-----
    python simulate_switching_cev.py                          # uses default config
    python simulate_switching_cev.py --config data/raw/switching_cev/sim_config.json
    python simulate_switching_cev.py --out_dir data/raw/switching_cev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Poisson regime schedule
# ---------------------------------------------------------------------------

def build_regime_schedule(
    n_steps: int,
    dt: float,
    lambda_switch: float,
    initial_regime: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns an integer array of shape (n_steps + 1,) with values 0 or 1
    indicating the regime at each time step.

    Switch times are drawn from a Poisson process with rate `lambda_switch`
    (switches per unit time). Concretely, inter-switch intervals are sampled
    as Exponential(1 / lambda_switch) and accumulated until they exceed T.

    Parameters
    ----------
    n_steps       : number of simulation steps
    dt            : step size
    lambda_switch : expected number of regime switches per unit time T
    initial_regime: 0 (normal) or 1 (bubble) at t=0
    rng           : numpy random Generator

    Returns
    -------
    regime : (n_steps+1,) int8 array, values in {0, 1}
    """
    T = n_steps * dt
    regime = np.empty(n_steps + 1, dtype=np.int8)

    # Collect switch times in [0, T]
    switch_times: list[float] = []
    t = 0.0
    while True:
        # inter-arrival time ~ Exp(lambda_switch)
        gap = rng.exponential(scale=1.0 / lambda_switch)
        t += gap
        if t >= T:
            break
        switch_times.append(t)

    # Fill regime array
    current = initial_regime
    sw_idx = 0
    for step in range(n_steps + 1):
        t_now = step * dt
        while sw_idx < len(switch_times) and switch_times[sw_idx] <= t_now:
            current = 1 - current  # flip
            sw_idx += 1
        regime[step] = current

    return regime


# ---------------------------------------------------------------------------
# Per-step γ₁ sampling
# ---------------------------------------------------------------------------

def sample_gamma1_for_regimes(
    regime: np.ndarray,
    gamma1_lo_normal: float,
    gamma1_hi_normal: float,
    gamma1_lo_bubble: float,
    gamma1_hi_bubble: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Assign a γ₁ value to each step.

    Within each contiguous regime segment, γ₁ is sampled ONCE (not per step),
    so the path has a consistent local volatility exponent within each regime
    spell — matching how real bubble episodes behave.

    Returns
    -------
    gamma1_arr : (n_steps+1,) float32
    """
    n = len(regime)
    gamma1_arr = np.empty(n, dtype=np.float32)

    i = 0
    while i < n:
        r = int(regime[i])
        # find end of this contiguous segment
        j = i + 1
        while j < n and regime[j] == r:
            j += 1

        # sample one γ₁ for this segment
        if r == 0:
            g = rng.uniform(gamma1_lo_normal, gamma1_hi_normal)
        else:
            g = rng.uniform(gamma1_lo_bubble, gamma1_hi_bubble)

        gamma1_arr[i:j] = float(g)
        i = j

    return gamma1_arr


# ---------------------------------------------------------------------------
# CEV Euler-Maruyama with time-varying γ₁
# ---------------------------------------------------------------------------

def simulate_switching_cev_path(
    S0: float,
    gamma0: float,
    gamma1_arr: np.ndarray,
    dt: float,
    eps: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulates dS_t = gamma0 * S_t^{gamma1_t} dW_t  (Euler-Maruyama)

    gamma1_arr : (n_steps+1,) — γ₁ at each step (step n uses gamma1_arr[n])
    Returns S : (n_steps+1,) float64
    """
    n_steps = len(gamma1_arr) - 1
    S = np.empty(n_steps + 1, dtype=np.float64)
    S[0] = S0

    for n in range(n_steps):
        z = rng.standard_normal()
        vol = gamma0 * (S[n] ** gamma1_arr[n])
        S_next = S[n] + vol * np.sqrt(dt) * z
        S[n + 1] = S_next if (np.isfinite(S_next) and S_next > 0.0) else eps

    return S


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def generate_split(cfg: dict, split_name: str, seed: int):
    """
    Generate all paths for one split.

    Returns
    -------
    prices      : (N, n_steps+1) float64
    times       : (n_steps+1,)   float64
    labels      : (N, n_steps+1) int8      per-timestep regime label
    gamma1_path : (N, n_steps+1) float32   γ₁ at each step
    params      : (N, 4)         float32   [gamma0, lambda_switch, initial_gamma1, initial_label]
    param_names : (4,)           object
    """
    S0              = float(cfg["S0"])
    T               = float(cfg["T"])
    n_steps         = int(cfg["n_steps"])
    dt              = T / n_steps
    eps             = float(cfg["eps_floor"])
    gamma0          = float(cfg["gamma0"])
    lambda_switch   = float(cfg["lambda_switch"])
    gamma1_lo_norm  = float(cfg["gamma1_lo_normal"])
    gamma1_hi_norm  = float(cfg["gamma1_hi_normal"])
    gamma1_lo_bub   = float(cfg["gamma1_lo_bubble"])
    gamma1_hi_bub   = float(cfg["gamma1_hi_bubble"])
    n_paths         = int(cfg["n_paths"][split_name])

    rng = np.random.default_rng(seed)

    times       = np.linspace(0.0, T, n_steps + 1, dtype=np.float64)
    prices      = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    labels      = np.empty((n_paths, n_steps + 1), dtype=np.int8)
    gamma1_path = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    params      = np.empty((n_paths, 4), dtype=np.float32)

    for i in range(n_paths):
        # randomise initial regime: balanced across paths
        initial_regime = int(i % 2)  # alternates 0,1,0,1,...

        regime = build_regime_schedule(
            n_steps=n_steps,
            dt=dt,
            lambda_switch=lambda_switch,
            initial_regime=initial_regime,
            rng=rng,
        )

        g1_arr = sample_gamma1_for_regimes(
            regime,
            gamma1_lo_normal=gamma1_lo_norm,
            gamma1_hi_normal=gamma1_hi_norm,
            gamma1_lo_bubble=gamma1_lo_bub,
            gamma1_hi_bubble=gamma1_hi_bub,
            rng=rng,
        )

        path = simulate_switching_cev_path(
            S0=S0,
            gamma0=gamma0,
            gamma1_arr=g1_arr,
            dt=dt,
            eps=eps,
            rng=rng,
        )

        prices[i]      = path
        labels[i]      = regime
        gamma1_path[i] = g1_arr
        params[i]      = [gamma0, lambda_switch, float(g1_arr[0]), float(initial_regime)]

        if (i + 1) % 10_000 == 0:
            print(f"  [{split_name}] simulated {i+1}/{n_paths} paths …")

    param_names = np.array(["gamma0", "lambda_switch", "initial_gamma1", "initial_label"], dtype=object)
    return prices, times, labels, gamma1_path, params, param_names


def save_split(
    out_dir: Path,
    split_name: str,
    prices: np.ndarray,
    times: np.ndarray,
    labels: np.ndarray,
    gamma1_path: np.ndarray,
    params: np.ndarray,
    param_names: np.ndarray,
) -> None:
    ensure_dir(out_dir)
    out_path = out_dir / f"{split_name}_paths.npz"
    np.savez_compressed(
        out_path,
        prices=prices,
        times=times,
        labels=labels,
        gamma1_path=gamma1_path,
        params=params,
        param_names=param_names,
    )
    n_bubble_steps = int((labels == 1).sum())
    n_total_steps  = int(labels.size)
    frac_bubble    = n_bubble_steps / n_total_steps
    print(
        f"Saved {split_name} => {out_path} | "
        f"paths={prices.shape[0]}, steps={prices.shape[1]}, "
        f"bubble_frac={frac_bubble:.3f}"
    )


# ---------------------------------------------------------------------------
# Default config (written to disk if not present)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "random_seed": 42,

    # path / SDE
    "S0": 100.0,
    "T": 1.0,
    "n_steps": 252,
    "eps_floor": 1e-6,

    # CEV diffusion coefficient (same for both regimes, only exponent switches)
    "gamma0": 0.3,

    # Poisson switch rate: expected switches per unit time T
    # 4.0 => ~4 regime switches per path on average
    "lambda_switch": 4.0,

    # γ₁ ranges per regime
    "gamma1_lo_normal": 0.2,
    "gamma1_hi_normal": 1.0,
    "gamma1_lo_bubble": 1.0,   # exclusive lower bound handled by sampler
    "gamma1_hi_bubble": 2.0,

    # dataset sizes
    "n_paths": {
        "train": 80000,
        "val":   10000,
        "test":  10000
    }
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Simulate regime-switching CEV paths (Poisson switches, per-step labels)."
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to sim_config.json. If omitted, uses built-in defaults.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/raw/switching_cev",
        help="Output directory for split .npz files and config.",
    )
    args = ap.parse_args()

    root = project_root()
    out_dir = root / args.out_dir
    ensure_dir(out_dir)

    # Load or write config
    if args.config is not None:
        config_path = root / args.config
        with open(config_path, "r") as f:
            cfg = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        cfg = DEFAULT_CONFIG
        config_path = out_dir / "sim_config.json"
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"Wrote default config to {config_path}")
        else:
            # config already exists on disk — load it
            with open(config_path, "r") as f:
                cfg = json.load(f)
            print(f"Loaded existing config from {config_path}")

    base_seed = int(cfg.get("random_seed", 42))

    for j, split in enumerate(["train", "val", "test"]):
        seed = base_seed + 1000 * j
        print(f"\nGenerating split: {split}  (seed={seed})")
        prices, times, labels, gamma1_path, params, param_names = generate_split(cfg, split, seed)
        save_split(out_dir, split, prices, times, labels, gamma1_path, params, param_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
