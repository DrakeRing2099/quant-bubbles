import json 
import numpy as np
import pandas as pd
from pathlib import Path

# Load config and models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATHS_DIR = DATA_DIR / "raw" / "paths"


def load_sim_config(config_path: Path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg

def load_models(models_path: Path):
    df = pd.read_csv(models_path)
    return df 


# SDE Simulators

def simulate_path_gbm(S0, sigma_gbm, dt, n_steps, eps=1e-6, rng=None):
    """
    dS_t = sigma_bgm * S_t dW_t (risk-neutral, r=0)
    Euler-Maruyama discretization
    """
    if rng is None:
        rng = np.random.default_rng()

    S = np.empty(n_steps + 1)
    S[0] = S0

    for n in range(n_steps):
        z = rng.normal()
        sigma = sigma_gbm * S[n]
        S[n + 1] = S[n] + sigma * np.sqrt(dt) * z 
        if S[n + 1] <= 0:
            S[n + 1] = eps

    return S


def simulate_path_cev(S0, gamma0, gamma1, dt, n_steps, eps=1e-6, rng=None):
    """
    dS_t = gamma0 * S_t^{gamma1} dW_t
    Bubble vs No Bubble comes from gamma1:
        - gamma1 <= 1 -> true martingale (no bubble)
        - gamma1 > 1 -> strict local martingale (bubble)
    """
    if rng is None:
        rng = np.random.default_rng()

    S = np.empty(n_steps + 1)
    S[0] = S0 

    for n in range(n_steps):
        z = rng.normal()
        sigma = gamma0 * (S[n] ** gamma1)
        S[n + 1] = S[n] + sigma * np.sqrt(dt) * z 
        if S[n + 1] <= 0:
            S[n + 1] = eps 

    return S 


def simulate_path_for_model(model_row, S0, dt, n_steps, eps, rng):
    """
    Wrapper that reads a row of models.csv and calls the right simulator
    """

    sde_type = model_row["sde_type"]

    if sde_type == "gbm":
        sigma_gbm = float(model_row["sigma_gbm"])
        return simulate_path_gbm(S0, sigma_gbm, dt, n_steps, eps=eps, rng=rng)
    
    elif sde_type == "cev":
        gamma0 = float(model_row["gamma0"])
        gamma1 = float(model_row["gamma1"])
        return simulate_path_cev(S0, gamma0, gamma1, dt, n_steps, eps=eps, rng=rng)

    else:
        raise ValueError(f"Unknown sde_type: {sde_type}")
    


# Generate split

def generate_split(split_name, models_df, sim_cfg, base_seed):
    """
     Generate all paths for one split: train / val / test.
    Returns:
      prices:   (N_paths, n_steps+1)
      times:    (n_steps+1,)
      model_id: (N_paths,)
      labels:   (N_paths,)
    """
    S0 = float(sim_cfg["S0"])
    T = float(sim_cfg["T"])
    n_steps = int(sim_cfg["n_steps"])
    dt = T / n_steps
    eps = float(sim_cfg["eps_floor"])

    n_paths_per_model = sim_cfg["n_paths_per_model"][split_name]
    
    n_models = len(models_df)
    total_paths = n_models * n_paths_per_model

    prices = np.empty((total_paths, n_steps + 1), dtype=float)
    model_ids = np.empty(total_paths, dtype=int)
    labels = np.empty(total_paths, dtype=int)

    times = np.linspace(0.0, T, n_steps + 1)

    # one RNG per split for reproducibility
    rng = np.random.default_rng(base_seed)

    idx = 0

    for _, row in models_df.iterrows():
        model_id = int(row["model_id"])
        label = int(row["label"])

        for _ in range(n_paths_per_model):
            path = simulate_path_for_model(row, S0, dt, n_steps, eps, rng)
            prices[idx, :] = path
            model_ids[idx] = model_id
            labels[idx] = label
            idx += 1

    return prices, times, model_ids, labels


def save_split(split_name, prices, times, model_ids, labels):
    RAW_PATHS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_PATHS_DIR / f"{split_name}_paths.npz"

    np.savez_compressed(
        out_path,
        prices=prices,
        times=times,
        model_id=model_ids,
        labels=labels,
    )
    print(f"Saved {split_name} split to {out_path} with {prices.shape[0]} paths.")


# Main
def main():
    config_path = DATA_DIR / "raw" / "sim_config.json"
    models_path = DATA_DIR / "raw" / "models.csv"

    sim_cfg = load_sim_config(config_path)
    models_df = load_models(models_path)

    base_seed = int(sim_cfg.get("random_seed", 12345))

    for i, split_name in enumerate(["train", "val", "test"]):
        # change seed slightly per split to avoid identical paths
        split_seed = base_seed + i * 1000

        prices, times, model_ids, labels = generate_split(
            split_name, models_df, sim_cfg, split_seed
        )
        save_split(split_name, prices, times, model_ids, labels)


if __name__ == "__main__":
    main()