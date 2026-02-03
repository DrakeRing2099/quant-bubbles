import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import signatory
from joblib import load

import yfinance as yf


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/date range.")
    if "Close" not in df.columns:
        raise ValueError("Expected 'Close' column in downloaded data.")
    close = df["Close"].dropna()
    return close


def window_to_path(close_window: np.ndarray) -> np.ndarray:
    close_window = np.asarray(close_window).reshape(-1)  # force (L,)
    L = close_window.shape[0]
    t = np.linspace(0.0, 1.0, L)
    x = np.log(close_window / close_window[0])
    return np.stack([t, x], axis=1)  # (L, 2)



def signatures_for_windows(close: np.ndarray, window: int, step: int, depth: int, device: str, batch_size: int):
    """
    Build rolling windows from `close`, compute depth-`depth` signatures for each window.
    Returns:
      end_dates_idx: indices of window end points
      X_sig: (N_windows, D_sig)
    """
    n = len(close)
    starts = list(range(0, n - window + 1, step))
    if not starts:
        raise ValueError(f"Not enough data points ({n}) for window={window}.")

    end_idx = [s + window - 1 for s in starts]

    # Build paths in batches
    sigs = []
    batch_paths = []

    for i, s in enumerate(starts):
        w = close[s : s + window]
        path = window_to_path(w)
        batch_paths.append(path)

        if len(batch_paths) == batch_size:
            paths_np = np.asarray(batch_paths, dtype=np.float32)  # (B, L, 2)
            x = torch.from_numpy(paths_np).to(device)
            with torch.no_grad():
                sig = signatory.signature(x, depth=depth)
            sigs.append(sig.detach().cpu().numpy())
            batch_paths = []

    # flush remainder
    if batch_paths:
        paths_np = np.asarray(batch_paths, dtype=np.float32)
        x = torch.from_numpy(paths_np).to(device)
        with torch.no_grad():
            sig = signatory.signature(x, depth=depth)
        sigs.append(sig.detach().cpu().numpy())

    X_sig = np.vstack(sigs)
    return np.array(end_idx, dtype=int), X_sig


def main():
    parser = argparse.ArgumentParser(description="System B: historical case study on YES Bank using trained signature classifier.")
    parser.add_argument("--ticker", type=str, default="YESBANK.NS")
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default="2020-12-31")
    parser.add_argument("--window", type=int, default=252, help="Rolling window length in trading days.")
    parser.add_argument("--step", type=int, default=5, help="Step size between windows.")
    parser.add_argument("--depth", type=int, default=3, help="Signature depth (must match trained model).")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for signature computation.")
    parser.add_argument("--plot_price", action="store_true", help="Also plot the price series.")
    args = parser.parse_args()

    root = project_root()
    model_dir = root / "models" / "systemB"
    scaler_path = model_dir / f"scaler_depth{args.depth}.joblib"
    clf_path = model_dir / f"logreg_depth{args.depth}.joblib"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    if not clf_path.exists():
        raise FileNotFoundError(f"Missing classifier: {clf_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available. Use --device cpu.")

    device = args.device

    scaler = load(scaler_path)
    clf = load(clf_path)

    close_series = fetch_prices(args.ticker, args.start, args.end)
    close = close_series.values.astype(np.float64)
    dates = close_series.index

    end_idx, X_sig = signatures_for_windows(
        close=close,
        window=args.window,
        step=args.step,
        depth=args.depth,
        device=device,
        batch_size=args.batch_size,
    )

    X_sig_scaled = scaler.transform(X_sig)
    bubble_prob = clf.predict_proba(X_sig_scaled)[:, 1]
    prob_dates = dates[end_idx]

    out = pd.DataFrame({"bubble_prob": bubble_prob}, index=prob_dates)

    if args.plot_price:
        plt.figure()
        plt.plot(close_series.index, close_series.values)
        plt.title(f"{args.ticker} adjusted close ({args.start} to {args.end})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()

    plt.figure()
    plt.plot(out.index, out["bubble_prob"].values)
    plt.title(f"{args.ticker} — System B bubble probability (window={args.window}, step={args.step}, depth={args.depth})")
    plt.xlabel("Date")
    plt.ylabel("P(bubble)")
    plt.ylim(-0.05, 1.05)
    plt.show()

    print(out.tail(10))


if __name__ == "__main__":
    main()
