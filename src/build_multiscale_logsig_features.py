from __future__ import annotations

import argparse

try:
    from src.multiscale_xgb_common import (
        default_feature_paths,
        feature_split_filename,
        load_raw_split,
        multiscale_logsignature_features,
        normalize_scales,
        project_root,
        resolve_dataset_spec,
        save_feature_split,
        write_feature_metadata,
    )
except ImportError:
    from multiscale_xgb_common import (
        default_feature_paths,
        feature_split_filename,
        load_raw_split,
        multiscale_logsignature_features,
        normalize_scales,
        project_root,
        resolve_dataset_spec,
        save_feature_split,
        write_feature_metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multiscale logsignature features for multiscale_xgb.")
    parser.add_argument("--dataset", required=True, choices=["cev", "shifted_cev", "sin"])
    parser.add_argument("--in_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--local_lookback", type=int, default=10)
    parser.add_argument("--scales", type=str, default="21,50,108,252")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    spec = resolve_dataset_spec(args.dataset)
    raw_dir, out_dir = default_feature_paths(args.dataset, in_dir=args.in_dir, out_dir=args.out_dir)
    scales = normalize_scales(args.scales)

    payload_keys: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        prices, _, payload = load_raw_split(raw_dir, split)
        X_feat = multiscale_logsignature_features(
            prices,
            scales=scales,
            depth=args.depth,
            device=args.device,
            batch_size=args.batch_size,
            local_lookback=args.local_lookback,
        )
        payload_keys[split] = sorted(payload.keys())
        out_path = out_dir / feature_split_filename(split, args.depth)
        save_feature_split(out_path, X_feat, payload)
        print(f"Saved {split} features -> {out_path} | {X_feat.shape}")

    meta_path = out_dir / f"metadata_depth{args.depth}.json"
    write_feature_metadata(
        meta_path,
        dataset=args.dataset,
        dataset_label=spec.dataset_label,
        raw_dir=raw_dir,
        out_dir=out_dir,
        depth=args.depth,
        device=args.device,
        batch_size=args.batch_size,
        scales=scales,
        local_lookback=args.local_lookback,
        payload_keys=payload_keys,
    )
    print(f"Saved metadata -> {meta_path}")


if __name__ == "__main__":
    main()
