import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
import torch
import signatory
from joblib import load


# -------------------------
# Paths / loading
# -------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def variant_to_config(variant: str) -> tuple[bool, str]:
    """
    Maps your variant naming to (leadlag?, transform).
    Must match how you computed features + trained models.
    """
    if variant == "base_sig":
        return False, "signature"
    if variant == "base_log":
        return False, "logsignature"
    if variant == "ll_sig":
        return True, "signature"
    if variant == "ll_log":
        return True, "logsignature"
    raise ValueError(f"Unknown variant: {variant}. Use one of base_sig, base_log, ll_sig, ll_log.")


def load_model(root: Path, variant: str, depth: int):
    """
    Loads:
      models/systemB_variants/{variant}_depth{depth}/scaler.joblib
      models/systemB_variants/{variant}_depth{depth}/model.joblib
    """
    model_dir = root / "models" / "systemB_variants" / f"{variant}_depth{depth}"
    scaler_path = model_dir / "scaler.joblib"
    model_path = model_dir / "model.joblib"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    scaler = load(scaler_path)
    clf = load(model_path)
    return scaler, clf


def fetch_adj_close(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for {ticker}")

    if "Close" not in df.columns:
        raise ValueError(f"Unexpected columns for {ticker}: {list(df.columns)}")

    s = df["Close"].dropna()
    s = s[~s.index.duplicated(keep="last")]

    if len(s) < 400:
        raise ValueError(f"Too few points for {ticker}: {len(s)}")

    return s


# -------------------------
# Lead-lag for 1D sequence
# -------------------------

def lead_lag_1d(x: np.ndarray) -> np.ndarray:
    """
    Lead-lag transform for 1D sequence length L:
      (x1,x1), (x2,x1), (x2,x2), (x3,x2), ..., (xL, x_{L-1}), (xL, xL)
    returns (2L-1, 2)
    """
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


# -------------------------
# Window -> path (base or lead-lag)
# -------------------------

def window_to_path_base(close_window: np.ndarray) -> np.ndarray:
    """
    Base System B path per window: (L, 2) channels:
      [t in [0,1], log(price / price_0)]
    """
    w = np.asarray(close_window, dtype=float).reshape(-1)
    L = w.shape[0]

    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Invalid window prices (nonfinite or nonpositive)")

    t = np.linspace(0.0, 1.0, L, dtype=float)
    x = np.log(w / w[0])
    if not np.all(np.isfinite(x)):
        raise ValueError("Invalid log transform (nan/inf)")

    return np.stack([t, x], axis=1)  # (L,2)


def window_to_path_leadlag(close_window: np.ndarray) -> np.ndarray:
    """
    Lead-lag variant:
      - compute x = log(price/price0), length L
      - build lead-lag (2L-1, 2): (lead, lag)
      - rebuild time grid length (2L-1)
      => final path (2L-1, 3): [t, lead, lag]
    """
    w = np.asarray(close_window, dtype=float).reshape(-1)
    L = w.shape[0]

    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Invalid window prices (nonfinite or nonpositive)")

    x = np.log(w / w[0])
    if not np.all(np.isfinite(x)):
        raise ValueError("Invalid log transform (nan/inf)")

    ll = lead_lag_1d(x)  # (2L-1, 2)
    t_ll = np.linspace(0.0, 1.0, 2 * L - 1, dtype=float)

    out = np.empty((2 * L - 1, 3), dtype=float)
    out[:, 0] = t_ll
    out[:, 1] = ll[:, 0]  # lead
    out[:, 2] = ll[:, 1]  # lag
    return out


def make_paths_from_series(
    s: pd.Series, window: int, step: int, leadlag: bool
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Build paths for rolling windows.

    Returns:
      paths: (N_windows, L, d)
      end_dates: list of window end timestamps
    """
    s = s.dropna()
    x = np.asarray(s.values, dtype=float).reshape(-1)
    dates = s.index

    ends = list(range(window, len(x) + 1, step))
    if len(ends) < 5:
        raise ValueError(f"Not enough windows: {len(ends)} (window={window}, step={step}, n={len(x)})")

    paths: List[np.ndarray] = []
    end_dates: List[pd.Timestamp] = []

    for e in ends:
        w = x[e - window: e]
        try:
            if leadlag:
                path = window_to_path_leadlag(w)
            else:
                path = window_to_path_base(w)
        except ValueError:
            # skip bad windows
            continue

        paths.append(path)
        end_dates.append(dates[e - 1])

    if not paths:
        raise ValueError("All windows were skipped (bad data or too strict conditions).")

    # pad-check consistency:
    L0 = paths[0].shape[0]
    d0 = paths[0].shape[1]
    for p in paths:
        if p.shape != (L0, d0):
            raise ValueError("Inconsistent path shapes across windows (should not happen).")

    return np.stack(paths, axis=0), end_dates


# -------------------------
# Feature computation
# -------------------------

def features_from_paths(paths: np.ndarray, depth: int, transform: str, device: str) -> np.ndarray:
    X = torch.from_numpy(np.asarray(paths, dtype=np.float32)).to(device)
    with torch.no_grad():
        if transform == "signature":
            feat = signatory.signature(X, depth=depth)
        elif transform == "logsignature":
            try:
                feat = signatory.logsignature(X, depth=depth)
            except TypeError:
                feat = signatory.logsignature(X, depth=depth, mode="words")
        else:
            raise ValueError(f"Unknown transform: {transform}")
    return feat.detach().cpu().numpy()


# -------------------------
# Main scan
# -------------------------

def scan_ticker(
    ticker: str,
    start: str,
    end: Optional[str],
    window: int,
    step: int,
    variant: str,
    depth: int,
    device: str,
) -> Tuple[pd.Series, pd.DataFrame]:
    root = project_root()
    leadlag, transform = variant_to_config(variant)
    scaler, clf = load_model(root, variant=variant, depth=depth)

    s = fetch_adj_close(ticker, start=start, end=end)
    paths, end_dates = make_paths_from_series(s, window=window, step=step, leadlag=leadlag)

    Xfeat = features_from_paths(paths, depth=depth, transform=transform, device=device)
    Xfeat_s = scaler.transform(Xfeat)
    probs = clf.predict_proba(Xfeat_s)[:, 1]

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


def maybe_plot_and_save(
    ticker: str,
    s: pd.Series,
    probs_df: pd.DataFrame,
    out_dir: Path,
    variant: str,
    depth: int,
    threshold: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # price plot
    fig = plt.figure()
    plt.plot(s.index, s.values)
    plt.title(f"{ticker} adjusted close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    fig.savefig(out_dir / f"{ticker}_{variant}_d{depth}_price.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # prob plot
    fig = plt.figure()
    plt.plot(probs_df.index, probs_df["bubble_prob"].values)
    if threshold is not None:
        plt.axhline(threshold)
    plt.title(f"{ticker} — bubble probability ({variant}, depth={depth})")
    plt.xlabel("Date")
    plt.ylabel("P(bubble)")
    plt.ylim(-0.02, 1.02)
    fig.savefig(out_dir / f"{ticker}_{variant}_d{depth}_prob.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Zoomed-in plot: mid-2019 to mid-2020
    zoom_start = pd.Timestamp("2019-06-01")
    zoom_end = pd.Timestamp("2020-06-30")

    zoom_probs = probs_df.loc[(probs_df.index >= zoom_start) & (probs_df.index <= zoom_end)]
    zoom_price = s.loc[(s.index >= zoom_start) & (s.index <= zoom_end)]

    if len(zoom_probs) > 0 and len(zoom_price) > 0:
        fig = plt.figure()
        plt.plot(zoom_probs.index, zoom_probs["bubble_prob"].values)
        if threshold is not None:
            plt.axhline(threshold)
        plt.title(f"{ticker} — bubble probability (zoomed, {variant}, depth={depth})")
        plt.xlabel("Date")
        plt.ylabel("P(bubble)")
        plt.ylim(-0.02, 1.02)
        fig.savefig(out_dir / f"{ticker}_{variant}_d{depth}_prob_zoom_2019_2020.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        plt.plot(zoom_price.index, zoom_price.values)
        plt.title(f"{ticker} adjusted close (zoomed)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        fig.savefig(out_dir / f"{ticker}_{variant}_d{depth}_price_zoom_2019_2020.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # CSV
    probs_df.to_csv(out_dir / f"{ticker}_{variant}_d{depth}_probs.csv")



def run_scan_return_series(
    ticker: str,
    variant: str,
    depth: int,
    window: int,
    step: int,
    start: str | None = "2014-01-01",
    end: str | None = None,
    threshold: float = 0.8,
    device: str = "cpu",
):
    """
    GUI entrypoint.
    Returns dict with:
      - price: pd.Series (DatetimeIndex)
      - probs: pd.DataFrame with column 'bubble_prob' (DatetimeIndex)
      - meta: dict
    """
    s, probs_df = scan_ticker(
        ticker=ticker,
        start=start,
        end=end,
        window=int(window),
        step=int(step),
        variant=variant,
        depth=int(depth),
        device=device,
    )

    # Optional stats
    stats = summarize_probs(probs_df, lookback_days=252)

    return {
        "price": s,
        "probs": probs_df,
        "meta": {
            "ticker": ticker,
            "variant": variant,
            "depth": int(depth),
            "window": int(window),
            "step": int(step),
            "start": start,
            "end": end,
            "threshold": float(threshold),
            "device": device,
            "stats_1y": stats,
        },
    }



def main():
    parser = argparse.ArgumentParser(description="System B REALDATA scan for feature variants (lead-lag/logsignature).")
    parser.add_argument("--ticker", type=str, required=True, help="e.g. YESBANK.NS")
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--variant", type=str, default="base_sig",
                        choices=["base_sig", "base_log", "ll_sig", "ll_log"])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--out_dir", type=str, default="outputs/systemB_real_variants")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available. Use --device cpu.")

    s, probs = scan_ticker(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        window=args.window,
        step=args.step,
        variant=args.variant,
        depth=args.depth,
        device=args.device,
    )

    stats = summarize_probs(probs, lookback_days=252)
    print("\n=== Summary ===")
    print(f"Ticker: {args.ticker}")
    print(f"Variant: {args.variant} | depth={args.depth} | window={args.window} | step={args.step}")
    for k, v in stats.items():
        print(f"{k}: {v}")

    root = project_root()
    out_dir = root / args.out_dir
    if not args.no_plots:
        maybe_plot_and_save(
            ticker=args.ticker,
            s=s,
            probs_df=probs,
            out_dir=out_dir,
            variant=args.variant,
            depth=args.depth,
            threshold=args.threshold,
        )
        print(f"\nSaved outputs to: {out_dir.resolve()}")

    # Print tail for quick sanity
    print("\n=== Tail (last 10 windows) ===")
    print(probs.tail(10))


if __name__ == "__main__":
    main()
