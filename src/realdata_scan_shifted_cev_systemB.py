import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import torch
import signatory
from joblib import load

import matplotlib.pyplot as plt

try:
    from src.sklearn_compat import load_joblib_with_sklearn_compat
except ImportError:
    from sklearn_compat import load_joblib_with_sklearn_compat


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fetch_adj_close(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if df is None or len(df) == 0:
        raise ValueError(f"No data for {ticker}")
    if "Close" not in df.columns:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    s = df["Close"].dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s


def lead_lag_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    L = x.shape[0]
    if L < 2:
        raise ValueError("Lead-lag requires L >= 2")

    out = np.empty((2 * L - 1, 2), dtype=float)
    out[0, 0] = x[0]
    out[0, 1] = x[0]

    idx = 1
    for k in range(1, L):
        out[idx, 0] = x[k]
        out[idx, 1] = x[k - 1]
        idx += 1
        out[idx, 0] = x[k]
        out[idx, 1] = x[k]
        idx += 1
    return out


def window_to_path_base(close_window: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(close_window, dtype=float).reshape(-1)
    w = np.maximum(w, eps)
    t = np.linspace(0.0, 1.0, len(w), dtype=float)
    x = np.log(w / w[0])
    return np.stack([t, x], axis=1)  # (L,2)


def window_to_path_leadlag(close_window: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(close_window, dtype=float).reshape(-1)
    w = np.maximum(w, eps)
    x = np.log(w / w[0])  # (L,)
    ll = lead_lag_1d(x)   # (2L-1, 2)
    t_ll = np.linspace(0.0, 1.0, ll.shape[0], dtype=float)
    out = np.empty((ll.shape[0], 3), dtype=float)
    out[:, 0] = t_ll
    out[:, 1] = ll[:, 0]  # lead
    out[:, 2] = ll[:, 1]  # lag
    return out  # (2L-1,3)


def make_paths(s: pd.Series, window: int, step: int, variant: str) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    x = s.values.astype(float)
    dates = s.index
    ends = list(range(window, len(x) + 1, step))

    paths = []
    end_dates = []
    for e in ends:
        w = x[e - window : e]
        if len(w) != window:
            continue
        try:
            if variant == "base":
                p = window_to_path_base(w)
            elif variant == "ll":
                p = window_to_path_leadlag(w)
            else:
                raise ValueError("variant must be base or ll")
        except Exception:
            continue
        paths.append(p)
        end_dates.append(dates[e - 1])

    if not paths:
        raise ValueError("No valid windows")
    return np.asarray(paths, dtype=np.float32), end_dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="YESBANK.NS")
    ap.add_argument("--start", type=str, default="2014-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--window", type=int, default=252)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--variant", type=str, default="ll", choices=["base", "ll"])
    ap.add_argument("--model_dir", type=str, default="models/shifted_cev_systemB")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    root = project_root()
    model_dir = root / args.model_dir
    scaler = load_joblib_with_sklearn_compat(model_dir / f"scaler_{args.variant}_depth{args.depth}.joblib")
    clf = load_joblib_with_sklearn_compat(model_dir / f"logreg_{args.variant}_depth{args.depth}.joblib")

    s = fetch_adj_close(args.ticker, args.start, args.end)
    paths, end_dates = make_paths(s, args.window, args.step, args.variant)

    x = torch.from_numpy(paths).to(args.device)
    with torch.no_grad():
        X_sig = signatory.signature(x, depth=args.depth).cpu().numpy()

    probs = clf.predict_proba(scaler.transform(X_sig))[:, 1]
    out = pd.Series(probs, index=pd.to_datetime(end_dates), name="bubble_prob")

    print(out.tail(10))
    print(f"\nWindows: {len(out)} | Latest: {out.iloc[-1]:.4f} | Max: {out.max():.4f}")

    if args.plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # --- Price ---
        axes[0].plot(s.index, s.values)
        axes[0].set_title(f"{args.ticker} | Price")
        axes[0].set_ylabel("Adj Close")

        # --- Bubble probability ---
        axes[1].plot(out.index, out.values)
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_title(
            f"Bubble probability ({args.variant}_sig, depth={args.depth})"
        )
        axes[1].set_ylabel("P(bubble)")
        axes[1].set_xlabel("Date")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
