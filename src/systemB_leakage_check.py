# src/systemB_leakage_check.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import signatory
from joblib import load
import yfinance as yf


def project_root() -> Path:
    # this file lives in: <root>/src/systemB_leakage_check.py
    return Path(__file__).resolve().parents[1]


def fetch_adj_close(ticker: str, start: str, end: str | None) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/date range.")

    # yfinance sometimes returns multi-index columns; normalize hard.
    # We want a 1D Series of closes.
    if isinstance(df.columns, pd.MultiIndex):
        # typical structure: ('Close', 'YESBANK.NS')
        if ("Close" in df.columns.get_level_values(0)):
            close = df["Close"]
        else:
            raise ValueError(f"Couldn't find Close in multiindex columns: {df.columns}")
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        if "Close" not in df.columns:
            raise ValueError(f"Expected 'Close' column, got columns={list(df.columns)}")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

    close = close.dropna()
    close.name = "Close"
    return close



def make_paths_from_series(s: pd.Series, window: int, step: int):
    s = s.dropna()
    x = np.asarray(s.values, dtype=float).reshape(-1)
    dates = s.index

    ends = list(range(window, len(x) + 1, step))
    if len(ends) < 5:
        raise ValueError(f"Not enough windows: {len(ends)} (window={window}, step={step}, n={len(x)})")

    t_fixed = np.linspace(0.0, 1.0, window, dtype=float).reshape(-1)

    paths = []
    end_dates = []
    for e in ends:
        w = np.asarray(x[e - window : e], dtype=float).reshape(-1)
        if w.shape[0] != window:
            continue
        if not np.all(np.isfinite(w)):
            continue
        if w[0] <= 0.0 or np.any(w <= 0.0):
            continue

        lp = np.log(w / w[0]).reshape(-1)
        if not np.all(np.isfinite(lp)):
            continue

        path = np.column_stack((t_fixed, lp))  # (window, 2): [t, log(price/price0)]
        paths.append(path)
        end_dates.append(dates[e - 1])

    if not paths:
        raise ValueError("All windows skipped (nonpositive/nonfinite data or not enough valid history).")

    return np.stack(paths, axis=0), pd.to_datetime(pd.Index(end_dates))


def signatures_from_paths(paths: np.ndarray, depth: int) -> np.ndarray:
    X = torch.from_numpy(paths).to(torch.float32)
    with torch.no_grad():
        sig = signatory.signature(X, depth=depth)
    return sig.cpu().numpy()


def worst_drawdown_trough(close: pd.Series):
    """
    Returns (trough_date, trough_drawdown, peak_date_before_trough)
    drawdown is negative, e.g. -0.75 for -75%
    """
    s = close.dropna().copy()
    running_max = s.cummax()
    dd = s / running_max - 1.0
    trough_date = dd.idxmin()
    trough_dd = float(dd.loc[trough_date])

    peak_date = s.loc[:trough_date].idxmax()
    return trough_date, trough_dd, peak_date


def main():
    ap = argparse.ArgumentParser(description="Check whether System B triggers only AFTER a crash (leakage check).")
    ap.add_argument("--ticker", type=str, default="YESBANK.NS")
    ap.add_argument("--start", type=str, default="2017-01-01")
    ap.add_argument("--end", type=str, default="2020-06-01")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--window", type=int, default=252)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.8, help="High-prob threshold for 'signal'.")
    ap.add_argument("--lookback_days", type=int, default=500, help="Only analyze last N calendar days for signal timing.")
    args = ap.parse_args()

    root = project_root()

    # model paths (must already exist)
    model_dir = root / "models" / "systemB"
    scaler_path = model_dir / f"scaler_depth{args.depth}.joblib"
    clf_path = model_dir / f"logreg_depth{args.depth}.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    if not clf_path.exists():
        raise FileNotFoundError(f"Missing classifier: {clf_path}")

    scaler = load(scaler_path)
    clf = load(clf_path)

    close = fetch_adj_close(args.ticker, args.start, args.end)

    # Crash proxy: worst peak-to-trough drawdown in the fetched period
    trough_date, trough_dd, peak_date = worst_drawdown_trough(close)

    paths, end_dates = make_paths_from_series(close, window=args.window, step=args.step)
    Xsig = signatures_from_paths(paths, depth=args.depth)
    Xsig_s = scaler.transform(Xsig)
    probs = clf.predict_proba(Xsig_s)[:, 1]
    probs_df = pd.DataFrame({"bubble_prob": probs}, index=end_dates).sort_index()

    # focus on recent window (optional)
    if args.lookback_days > 0:
        cutoff = probs_df.index.max() - pd.Timedelta(days=int(args.lookback_days))
        probs_df = probs_df.loc[probs_df.index >= cutoff]
        close_cut = close.loc[close.index >= cutoff]
    else:
        close_cut = close

    # first high-prob date
    hits = probs_df[probs_df["bubble_prob"] >= args.threshold]
    first_hit = hits.index.min() if len(hits) else None

    # counts pre/post trough
    pre_trough_hits = hits[hits.index < trough_date]
    post_trough_hits = hits[hits.index >= trough_date]

    print("\n=== System B leakage / timing check ===")
    print(f"Ticker: {args.ticker}")
    print(f"Range:  {args.start} to {args.end}")
    print(f"Model:  depth={args.depth}, window={args.window}, step={args.step}")
    print(f"Signal threshold: {args.threshold}")

    print("\n--- Crash proxy (worst drawdown in price series) ---")
    print(f"Peak date:   {peak_date.date()}")
    print(f"Trough date: {trough_date.date()}")
    print(f"Drawdown:    {trough_dd*100:.2f}%")

    print("\n--- Signal timing ---")
    if first_hit is None:
        print("No high-prob signal found at this threshold.")
    else:
        print(f"First high-prob date: {first_hit.date()}")
        print("First hit is BEFORE trough?" , bool(first_hit < trough_date))
        print("First hit is AFTER trough? " , bool(first_hit >= trough_date))

    print("\n--- Hit counts ---")
    print(f"Total windows analyzed: {len(probs_df)}")
    print(f"High-prob windows:      {len(hits)}")
    print(f"High-prob BEFORE trough:{len(pre_trough_hits)}")
    print(f"High-prob AFTER trough: {len(post_trough_hits)}")

    print("\n--- Quick sanity (tail) ---")
    print(probs_df.tail(10))

    # Optional: save to outputs for inspection
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"systemB_leakagecheck_{args.ticker.replace('.','_')}_d{args.depth}.csv"
    probs_df.to_csv(out_csv)
    print(f"\nSaved probs to: {out_csv}")


if __name__ == "__main__":
    main()
