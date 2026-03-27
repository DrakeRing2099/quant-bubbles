from __future__ import annotations

import argparse

from systemB_pipeline_common import (
    SPLITS,
    build_base_paths,
    build_lead_lag_paths,
    ensure_dir,
    load_json,
    load_raw_split,
    path_output_name,
    project_root,
    relative_or_absolute,
    resolve_dataset_spec,
    resolve_dir,
    save_paths_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build System B base and lead-lag paths for cev, shifted_cev, or sin datasets."
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["cev", "shifted_cev", "sin"])
    parser.add_argument("--in_dir", type=str, default=None, help="Raw dataset directory.")
    parser.add_argument("--out_dir", type=str, default=None, help="Processed System B paths directory.")
    parser.add_argument("--depth", type=int, default=3, help="Accepted for CLI symmetry; not used here.")
    parser.add_argument("--device", type=str, default="cpu", help="Accepted for CLI symmetry; not used here.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for path construction.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Floor applied before log normalization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = project_root()
    spec = resolve_dataset_spec(args.dataset)
    raw_dir = resolve_dir(root, args.in_dir, spec.raw_dir)
    out_dir = resolve_dir(root, args.out_dir, spec.paths_dir)
    ensure_dir(out_dir)

    sim_config = load_json(raw_dir / "sim_config.json")
    payload_keys_by_split: dict[str, list[str]] = {}

    for split in SPLITS:
        prices, times, payload = load_raw_split(raw_dir, split)
        payload_keys_by_split[split] = sorted(payload.keys())

        base_paths = build_base_paths(prices, times, eps=args.eps, batch_size=args.batch_size)
        base_path = out_dir / path_output_name(split, "base")
        save_paths_file(base_path, base_paths, payload)
        print(f"Saved {split} base paths -> {base_path} | {base_paths.shape}")

        ll_paths = build_lead_lag_paths(prices, times, eps=args.eps, batch_size=args.batch_size)
        ll_path = out_dir / path_output_name(split, "ll")
        save_paths_file(ll_path, ll_paths, payload)
        print(f"Saved {split} lead-lag paths -> {ll_path} | {ll_paths.shape}")

    metadata = {
        "system": "B",
        "dataset": spec.name,
        "dataset_label": spec.dataset_label,
        "raw_dir": relative_or_absolute(raw_dir, root),
        "out_dir": relative_or_absolute(out_dir, root),
        "splits": list(SPLITS),
        "path_variants": {
            "base": {
                "filename": "{split}_paths_for_signature.npz",
                "channels": ["t_norm", "log(S/S0)"],
            },
            "lead_lag": {
                "filename": "{split}_ll_paths_for_signature.npz",
                "channels": ["t_norm", "lead(log(S/S0))", "lag(log(S/S0))"],
            },
        },
        "raw_payload_keys": payload_keys_by_split,
        "compute": {
            "eps": args.eps,
            "batch_size": args.batch_size,
            "depth_arg": args.depth,
            "device_arg": args.device,
        },
        "sim_config": sim_config,
    }
    metadata_path = out_dir / "metadata.json"
    write_json(metadata_path, metadata)
    print(f"Saved metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
