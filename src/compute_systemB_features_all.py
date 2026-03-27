from __future__ import annotations

import argparse

from systemB_pipeline_common import (
    FEATURE_VARIANTS,
    SPLITS,
    compatibility_feature_aliases,
    compute_features,
    ensure_dir,
    feature_output_name,
    load_processed_paths,
    project_root,
    relative_or_absolute,
    resolve_dataset_spec,
    resolve_dir,
    save_feature_file,
    validate_device,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute System B signature and logsignature features for cev, shifted_cev, or sin datasets."
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["cev", "shifted_cev", "sin"])
    parser.add_argument("--in_dir", type=str, default=None, help="Processed System B paths directory.")
    parser.add_argument("--out_dir", type=str, default=None, help="Feature output directory.")
    parser.add_argument("--depth", type=int, default=3, help="Signature/logsignature depth.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for signatory computation.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base_sig", "base_log", "ll_sig", "ll_log"],
        choices=["base_sig", "base_log", "ll_sig", "ll_log"],
        help="Which feature variants to compute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_device(args.device)

    root = project_root()
    spec = resolve_dataset_spec(args.dataset)
    in_dir = resolve_dir(root, args.in_dir, spec.paths_dir)
    out_dir = resolve_dir(root, args.out_dir, spec.features_dir)
    ensure_dir(out_dir)

    variants_done: list[str] = []
    alias_map: dict[str, list[str]] = {}

    for split in SPLITS:
        cache: dict[str, tuple[object, dict[str, object]]] = {}
        for variant in args.variants:
            cfg = FEATURE_VARIANTS[variant]
            path_variant = str(cfg["path_variant"])

            if path_variant not in cache:
                cache[path_variant] = load_processed_paths(in_dir, split, path_variant)

            paths, payload = cache[path_variant]
            features = compute_features(
                paths=paths,
                depth=args.depth,
                device=args.device,
                batch_size=args.batch_size,
                transform=str(cfg["transform"]),
            )

            primary_name = feature_output_name(split, variant, args.depth)
            primary_path = out_dir / primary_name
            save_feature_file(primary_path, features, payload)
            print(f"Saved {split} {variant} -> {primary_path} | {features.shape}")

            aliases = compatibility_feature_aliases(args.dataset, split, variant, args.depth)
            alias_map[f"{split}:{variant}"] = aliases
            for alias_name in aliases:
                alias_path = out_dir / alias_name
                save_feature_file(alias_path, features, payload)
                print(f"Saved compatibility alias -> {alias_path}")

        variants_done = list(args.variants)

    metadata = {
        "system": "B",
        "dataset": spec.name,
        "dataset_label": spec.dataset_label,
        "paths_dir": relative_or_absolute(in_dir, root),
        "out_dir": relative_or_absolute(out_dir, root),
        "signature_depth": args.depth,
        "signatory_includes_level0": False,
        "compute": {
            "device": args.device,
            "batch_size": args.batch_size,
        },
        "variants": variants_done,
        "variant_definitions": {
            name: FEATURE_VARIANTS[name] for name in variants_done
        },
        "compatibility_aliases": alias_map,
    }
    metadata_path = out_dir / f"metadata_depth{args.depth}.json"
    write_json(metadata_path, metadata)
    print(f"Saved metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
