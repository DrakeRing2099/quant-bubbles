from __future__ import annotations

import argparse

try:
    from src.run_systemB_experiment import run_all, run_experiment
except ImportError:
    from run_systemB_experiment import run_all, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive wizard for System B experiments.")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch_size", type=int, default=512)
    return parser.parse_args()


def choose(prompt: str, options: list[tuple[str, str]]) -> str:
    print(prompt)
    for idx, (_, label) in enumerate(options, start=1):
        print(f"{idx}. {label}")
    while True:
        raw = input("> ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        print("Enter one of the listed numbers.")


def choose_depth(default: int = 3) -> int:
    print(f"Select depth (default = {default})")
    raw = input("> ").strip()
    return default if raw == "" else int(raw)


def main() -> None:
    args = parse_args()
    if args.run_all:
        run_all(depth=3, device=args.device, batch_size=args.batch_size)
        return

    dataset = choose("Select dataset:", [("cev", "cev"), ("shifted_cev", "shifted_cev"), ("sin", "sin")])
    path_kind = choose("Select path type:", [("base", "base"), ("ll", "lead-lag")])
    feature_kind = choose("Select feature type:", [("sig", "signature"), ("log", "logsignature")])
    depth = choose_depth()

    run_experiment(
        dataset=dataset,
        path_kind=path_kind,
        feature_kind=feature_kind,
        depth=depth,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
