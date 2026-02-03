# realdata_systemB_vol_overlay.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse your existing, verified pipeline pieces
from realdata_systemB_batch_scan import project_root, load_model, scan_ticker


def realized_var(close: pd.Series, window: int) -> pd.Series:
    """Rolling realized variance proxy = mean of squared log returns."""
    logret = np.log(close).diff()
    rv = (logret ** 2).rolling(window).mean()
    return rv


def minmax01(x: pd.Series) -> pd.Series:
    a = x.values.astype(float)
    lo = np.nanmin(a)
    hi = np.nanmax(a)
    return (x - lo) / (hi - lo + 1e-12)


def main():
    p = argparse.ArgumentParser(description="Overlay System B P(bubble) with realized variance (sanity check).")
    p.add_argument("--ticker", type=str, default="YESBANK.NS")
    p.add_argument("--start", type=str, default="2014-01-01")
    p.add_argument("--end", type=str, default="2020-12-31")
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--save_plot", action="store_true")
    p.add_argument("--out_dir", type=str, default="outputs/systemB_verify")
    args = p.parse_args()

    root = project_root()
    scaler, clf = load_model(root, depth=args.depth)

    # Run your supervised system B (reused code)
    s, probs_df = scan_ticker(
        ticker=args.ticker,
        scaler=scaler,
        clf=clf,
        start=args.start,
        end=args.end,
        window=args.window,
        step=args.step,
        depth=args.depth,
    )

    # Compute realized variance on the same close series
    rv = realized_var(s, window=args.window)

# ensure Series
    if isinstance(rv, pd.DataFrame):
        if rv.shape[1] == 1:
            rv = rv.iloc[:, 0]
        else:
            # pick the first column if multiple
            rv = rv.iloc[:, 0]

    rv = rv.rename("rv")

    df = pd.concat(
        [
            probs_df["bubble_prob"].rename("bubble_prob"),
            rv,
        ],
        axis=1,
    ).dropna()


    df["rv_norm"] = minmax01(df["rv"])

    # Plot
    plt.figure(figsize=(11, 4))
    plt.plot(df.index, df["bubble_prob"], label="P(bubble)")
    plt.plot(df.index, df["rv_norm"], label="Realized variance (norm)")
    plt.title(f"{args.ticker} — System B vs realized variance (window={args.window})")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()

    if args.save_plot:
        out_dir = root / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.ticker}_depth{args.depth}_window{args.window}_step{args.step}_vol_overlay.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {out_path}")

    plt.show()

    # Print a quick correlation number (optional but useful)
    corr = float(np.corrcoef(df["bubble_prob"].values, df["rv_norm"].values)[0, 1])
    print(f"Correlation(P(bubble), RV_norm) = {corr:.4f}")


if __name__ == "__main__":
    main()
