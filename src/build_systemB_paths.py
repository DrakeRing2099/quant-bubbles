import argparse
import json
from pathlib import Path

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_sim_config(data_dir: Path):
    config_path = data_dir / "raw" / "sim_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing sim config: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def load_raw_split(raw_paths_dir: Path, split_name: str):
    in_path = raw_paths_dir / f"{split_name}_paths.npz"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing raw paths file: {in_path}")

    data = np.load(in_path)
    prices = data["prices"]      # (N, L)
    times = data["times"]        # (L,)
    labels = data["labels"]      # (N,)
    model_id = data["model_id"]  # (N,)
    return prices, times, labels, model_id


def build_paths_time_logprice(prices: np.ndarray, times: np.ndarray, S0: float):
    """
    Convert raw prices into a 2-channel path:
      channel 0: t in [0,1]
      channel 1: log(S_t / S0)
    Output shape: (N, L, 2)
    """
    N, L = prices.shape

    t_norm = times / times[-1]                 # (L,)
    log_prices = np.log(prices / S0)           # (N, L)

    T_chan = np.broadcast_to(t_norm, (N, L))[:, :, None]  # (N, L, 1)
    P_chan = log_prices[:, :, None]                       # (N, L, 1)

    paths = np.concatenate([T_chan, P_chan], axis=2)      # (N, L, 2)
    return paths


def save_split(out_dir: Path, split_name: str, paths, labels, model_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_name}_paths_for_signature.npz"
    np.savez_compressed(out_path, paths=paths, labels=labels, model_id=model_id)
    print(f"Saved {split_name} signature paths to {out_path} with shape {paths.shape}")


def save_metadata(out_dir: Path, S0: float):
    meta_path = out_dir / "metadata.json"
    meta = {
        "system": "B",
        "path_representation": "channels",
        "channels": ["time_normalized_0_1", "log_price_over_S0"],
        "S0_used_for_normalization": float(S0),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Build System B signature-ready paths from raw simulated prices.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/systemB_paths",
        help="Output directory (relative to project root).",
    )
    args = parser.parse_args()

    root = project_root()
    data_dir = root / "data"
    raw_paths_dir = data_dir / "raw" / "paths"
    out_dir = root / args.out_dir

    sim_cfg = load_sim_config(data_dir)
    S0 = float(sim_cfg["S0"])

    for split_name in ["train", "val", "test"]:
        prices, times, labels, model_id = load_raw_split(raw_paths_dir, split_name)
        paths = build_paths_time_logprice(prices, times, S0)
        save_split(out_dir, split_name, paths, labels, model_id)

    save_metadata(out_dir, S0)


if __name__ == "__main__":
    main()
