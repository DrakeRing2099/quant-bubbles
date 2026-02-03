import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
import torch
import signatory
from joblib import load


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_model(root: Path, depth: int):
    model_dir = root / "models" / "systemB"
    scaler_path = model_dir / f"scaler_depth{depth}.joblib"
    model_path = model_dir / f"logreg_depth{depth}.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    scaler = load(scaler_path)
    clf = load(model_path)
    return scaler, clf


def fetch_adj_close(ticker: str, start: str, end: str | None) -> pd.Series:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=False,  # more stable in batch runs
    )
    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for {ticker}")

    if "Close" not in df.columns:
        raise ValueError(f"Unexpected columns for {ticker}: {list(df.columns)}")

    s = df["Close"].dropna()

    # Remove duplicates (can happen with some feeds)
    s = s[~s.index.duplicated(keep="last")]

    # Must be enough for many windows
    if len(s) < 400:
        raise ValueError(f"Too few points for {ticker}: {len(s)}")

    return s


def make_paths_from_series(s: pd.Series, window: int, step: int) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Build signature paths for sliding windows.

    Path per window: (window, 2) with channels:
      [t in [0,1], log(price / price_0)]

    Skips windows with:
      - wrong length (shouldn't happen, but safe)
      - nonpositive prices
      - non-finite log values
    """
    s = s.dropna()
    x = np.asarray(s.values, dtype=float).reshape(-1)  # force 1D float
    dates = s.index

    ends = list(range(window, len(x) + 1, step))
    if len(ends) < 5:
        raise ValueError(f"Not enough windows: {len(ends)} (window={window}, step={step}, n={len(x)})")

    t_fixed = np.linspace(0.0, 1.0, window, dtype=float).reshape(-1)

    paths: List[np.ndarray] = []
    end_dates: List[pd.Timestamp] = []

    for e in ends:
        w = np.asarray(x[e - window : e], dtype=float).reshape(-1)

        if w.shape[0] != window:
            continue

        # Reject nonpositive or nonfinite prices
        if not np.all(np.isfinite(w)):
            continue
        if w[0] <= 0.0 or np.any(w <= 0.0):
            continue

        lp = np.log(w / w[0]).reshape(-1)

        # Reject inf/nan from log
        if lp.shape[0] != window:
            continue
        if not np.all(np.isfinite(lp)):
            continue

        # Final safety: lengths must match
        if t_fixed.shape[0] != lp.shape[0]:
            continue

        path = np.column_stack((t_fixed, lp))  # (window, 2), safer than np.stack([..], axis=1)
        paths.append(path)
        end_dates.append(dates[e - 1])

    if not paths:
        raise ValueError("All windows skipped (nonpositive/nonfinite data or not enough valid history).")

    return np.stack(paths, axis=0), end_dates


def signatures_from_paths(paths: np.ndarray, depth: int) -> np.ndarray:
    X = torch.from_numpy(paths).to(torch.float32)
    with torch.no_grad():
        sig = signatory.signature(X, depth=depth)
    return sig.cpu().numpy()


def scan_ticker(
    ticker: str,
    scaler,
    clf,
    start: str,
    end: str | None,
    window: int,
    step: int,
    depth: int,
) -> Tuple[pd.Series, pd.DataFrame]:
    s = fetch_adj_close(ticker, start, end)
    paths, end_dates = make_paths_from_series(s, window=window, step=step)

    Xsig = signatures_from_paths(paths, depth=depth)
    Xsig_s = scaler.transform(Xsig)
    probs = clf.predict_proba(Xsig_s)[:, 1]

    out = pd.DataFrame({"Date": end_dates, "bubble_prob": probs}).set_index("Date")
    return s, out


def summarize_probs(df: pd.DataFrame, lookback_days: int = 252) -> dict:
    recent = df.tail(lookback_days) if len(df) > lookback_days else df
    latest = float(df["bubble_prob"].iloc[-1])
    max1y = float(recent["bubble_prob"].max())
    mean1y = float(recent["bubble_prob"].mean())
    frac80 = float((recent["bubble_prob"] >= 0.8).mean())
    frac60 = float((recent["bubble_prob"] >= 0.6).mean())
    return {
        "latest_prob": latest,
        "max_prob_1y": max1y,
        "mean_prob_1y": mean1y,
        "frac_ge_0p8_1y": frac80,
        "frac_ge_0p6_1y": frac60,
        "n_windows": int(len(df)),
    }


def maybe_plot(ticker: str, s: pd.Series, probs_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    plt.plot(s.index, s.values)
    plt.title(f"{ticker} adjusted close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    fig.savefig(out_dir / f"{ticker}_price.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(probs_df.index, probs_df["bubble_prob"].values)
    plt.title(f"{ticker} — System B bubble probability")
    plt.xlabel("Date")
    plt.ylabel("P(bubble)")
    plt.ylim(-0.02, 1.02)
    fig.savefig(out_dir / f"{ticker}_bubbleprob.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Batch scan System B bubble probabilities over many tickers.")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--tickers", type=str, default="")
    parser.add_argument("--tickers_file", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="outputs/systemB_scan.csv")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--plots_dir", type=str, default="outputs/systemB_plots")
    args = parser.parse_args()

    root = project_root()
    scaler, clf = load_model(root, depth=args.depth)

    tickers: List[str] = []
    if args.tickers.strip():
        tickers += [t.strip() for t in args.tickers.split(",") if t.strip()]

    if args.tickers_file.strip():
        p = Path(args.tickers_file)
        if not p.exists():
            raise FileNotFoundError(f"tickers_file not found: {p}")
        tickers += [
            line.strip()
            for line in p.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    tickers = list(dict.fromkeys(tickers))  # dedupe preserve order
    if not tickers:
        raise ValueError("Provide --tickers or --tickers_file")

    out_rows = []
    out_dir_plots = root / args.plots_dir

    for i, ticker in enumerate(tickers, 1):
        try:
            s, probs_df = scan_ticker(
                ticker=ticker,
                scaler=scaler,
                clf=clf,
                start=args.start,
                end=args.end,
                window=args.window,
                step=args.step,
                depth=args.depth,
            )

            summ = summarize_probs(probs_df, lookback_days=252)
            summ["ticker"] = ticker
            out_rows.append(summ)

            print(
                f"[{i}/{len(tickers)}] {ticker}: "
                f"latest={summ['latest_prob']:.3f} max1y={summ['max_prob_1y']:.3f} "
                f"(windows={summ['n_windows']})"
            )

            if args.save_plots:
                maybe_plot(ticker, s, probs_df, out_dir_plots)

        except Exception as e:
            print(f"[{i}/{len(tickers)}] {ticker}: SKIP ({e})")
            out_rows.append({"ticker": ticker, "error": str(e)})

    out = pd.DataFrame(out_rows)
    if "latest_prob" in out.columns:
        out = out.sort_values(by=["latest_prob", "max_prob_1y"], ascending=False, na_position="last")

    out_path = root / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved scan summary to: {out_path}")


if __name__ == "__main__":
    main()
