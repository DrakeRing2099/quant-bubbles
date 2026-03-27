# realdata_systemB_pca_anomaly.py
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

try:
    from src.sklearn_compat import load_joblib_with_sklearn_compat
except ImportError:
    from sklearn_compat import load_joblib_with_sklearn_compat


def project_root() -> Path:
    # Matches your other scripts (expects this file is inside a subfolder, e.g., src/)
    return Path(__file__).resolve().parents[1]


def fetch_adj_close(ticker: str, start: str, end: str | None) -> pd.Series:
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


def make_paths_from_series(
    s: pd.Series,
    window: int,
    step: int,
    add_rv: bool,
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Rolling-window path builder.

    Base (same as your System B): channels [t in [0,1], log(price / price_0)]

    If add_rv is True, add a 3rd channel:
      cumulative realized variance proxy inside the window:
        rv_i = sum_{j<=i} (Δ log price_j)^2
      normalized to [0,1] inside the window for stability.
    """
    s = s.dropna()
    x = np.asarray(s.values, dtype=float).reshape(-1)
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
        if not np.all(np.isfinite(w)):
            continue
        if w[0] <= 0.0 or np.any(w <= 0.0):
            continue

        lp = np.log(w / w[0]).reshape(-1)
        if not np.all(np.isfinite(lp)):
            continue

        if not add_rv:
            path = np.column_stack((t_fixed, lp))  # (window, 2)
        else:
            # log-price increments inside window
            logw = np.log(w)
            dlog = np.diff(logw)  # length window-1
            sq = dlog * dlog
            rv = np.concatenate(([0.0], np.cumsum(sq)))  # length window

            # normalize rv to [0,1] (avoid blow-ups across regimes)
            rv_max = float(np.max(rv))
            if rv_max > 0:
                rv = rv / (rv_max + 1e-12)
            else:
                rv = rv * 0.0

            path = np.column_stack((t_fixed, lp, rv))  # (window, 3)

        paths.append(path)
        end_dates.append(dates[e - 1])

    if not paths:
        raise ValueError("All windows skipped (nonpositive/nonfinite data or not enough valid history).")

    return np.stack(paths, axis=0), end_dates


def signatures_from_paths(paths: np.ndarray, depth: int, device: str) -> np.ndarray:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available. Use --device cpu.")

    X = torch.from_numpy(paths).to(torch.float32).to(device)
    with torch.no_grad():
        sig = signatory.signature(X, depth=depth)
    return sig.detach().cpu().numpy()


def load_supervised_model(root: Path, depth: int):
    model_dir = root / "models" / "systemB"
    scaler_path = model_dir / f"scaler_depth{depth}.joblib"
    model_path = model_dir / f"logreg_depth{depth}.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    scaler = load_joblib_with_sklearn_compat(scaler_path)
    clf = load_joblib_with_sklearn_compat(model_path)
    return scaler, clf


def anomaly_scores(
    X: np.ndarray,
    fit_n: int,
    method: str,
    n_components: int,
    random_state: int = 0,
) -> Tuple[np.ndarray, dict]:
    """
    Unsupervised anomaly score using signature features.

    method:
      - "pca_recon": PCA reconstruction MSE in scaled feature space
      - "pca_maha" : PCA + Mahalanobis distance in PCA space (recommended)
      - "iforest"  : IsolationForest on baseline
      - "ocsvm"    : One-Class SVM on baseline (can be sensitive)

    Baseline is first fit_n windows. StandardScaling is fit on baseline only.
    """
    n = X.shape[0]
    fit_n = int(max(30, min(fit_n, n)))  # baseline needs enough samples

    scaler_u = StandardScaler()
    X_fit_s = scaler_u.fit_transform(X[:fit_n])
    X_all_s = scaler_u.transform(X)

    meta = {"fit_n": fit_n, "method": method}

    if method == "pca_recon":
        k = int(min(n_components, X_all_s.shape[1], fit_n - 1))
        pca = PCA(n_components=k, random_state=random_state)
        pca.fit(X_fit_s)

        Z_all = pca.transform(X_all_s)
        X_rec = pca.inverse_transform(Z_all)
        err = np.mean((X_all_s - X_rec) ** 2, axis=1)

        meta.update(
            {
                "n_components": int(pca.n_components_),
                "explained_var_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            }
        )
        return err, meta

    if method == "pca_maha":
        k = int(min(n_components, X_all_s.shape[1], fit_n - 1))
        pca = PCA(n_components=k, random_state=random_state)
        Z_fit = pca.fit_transform(X_fit_s)
        Z_all = pca.transform(X_all_s)

        cov = LedoitWolf().fit(Z_fit)
        md2 = cov.mahalanobis(Z_all)  # squared Mahalanobis

        meta.update(
            {
                "n_components": int(pca.n_components_),
                "explained_var_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
                "cov": "LedoitWolf",
            }
        )
        return md2, meta

    if method == "iforest":
        iso = IsolationForest(
            n_estimators=600,
            max_samples="auto",
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        iso.fit(X_fit_s)
        score = -iso.decision_function(X_all_s)  # higher = more anomalous
        meta.update({"n_estimators": 600, "contamination": "auto"})
        return score, meta

    if method == "ocsvm":
        oc = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        oc.fit(X_fit_s)
        score = -oc.decision_function(X_all_s)
        meta.update({"nu": 0.05, "kernel": "rbf"})
        return score, meta

    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(
        description="System B: Unsupervised anomaly detection on signature features (+ optional supervised overlay)."
    )
    parser.add_argument("--ticker", type=str, default="YESBANK.NS")
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default="2020-12-31")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")

    # THIS is the representation upgrade
    parser.add_argument(
        "--add_rv",
        action="store_true",
        help="Add cumulative squared log-returns (realized variance proxy) as a 3rd path channel.",
    )

    # Unsupervised settings
    parser.add_argument(
        "--method",
        type=str,
        default="pca_maha",
        choices=["pca_recon", "pca_maha", "iforest", "ocsvm"],
        help="Unsupervised anomaly scoring method.",
    )
    parser.add_argument(
        "--fit_frac",
        type=float,
        default=0.25,
        help="Fraction of earliest windows used to fit the baseline.",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=5,
        help="Used for PCA methods (recon/maha). Keep small for earlier signal.",
    )
    parser.add_argument("--seed", type=int, default=0)

    # Output
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/systemB_unsupervised")

    # Supervised optional overlay
    parser.add_argument("--no_supervised", action="store_true", help="Skip loading logreg model and bubble probability.")

    args = parser.parse_args()

    root = project_root()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) fetch data + windows → paths → signatures
    s = fetch_adj_close(args.ticker, args.start, args.end)
    paths, end_dates = make_paths_from_series(s, window=args.window, step=args.step, add_rv=args.add_rv)
    Xsig = signatures_from_paths(paths, depth=args.depth, device=args.device)

    # 2) unsupervised anomaly curve
    fit_n = int(max(30, round(args.fit_frac * Xsig.shape[0])))
    anomaly, meta = anomaly_scores(
        Xsig,
        fit_n=fit_n,
        method=args.method,
        n_components=args.pca_components,
        random_state=args.seed,
    )

    out = pd.DataFrame(
        {"anomaly_score": anomaly},
        index=pd.DatetimeIndex(end_dates, name="Date"),
    )

    # normalize for plotting convenience (0..1)
    a = out["anomaly_score"].values
    out["anomaly_score_norm"] = (a - np.min(a)) / (np.max(a) - np.min(a) + 1e-12)

    # 3) optional supervised overlay (NOTE: your saved supervised model expects 2-channel paths.
    # If you run with --add_rv, supervised overlay will likely break / mismatch feature dims.
    # So default behavior: automatically disable supervised when add_rv is on.
    if args.add_rv:
        args.no_supervised = True

    if not args.no_supervised:
        scaler_s, clf = load_supervised_model(root, depth=args.depth)
        Xsig_s = scaler_s.transform(Xsig)
        out["bubble_prob"] = clf.predict_proba(Xsig_s)[:, 1]

    # 4) save csv
    rep = "withRV" if args.add_rv else "base"
    tag = f"{args.ticker}_window{args.window}_step{args.step}_depth{args.depth}_{rep}_{args.method}"
    csv_path = out_dir / f"{tag}.csv"
    out.to_csv(csv_path)
    print(f"Saved: {csv_path}")
    print(f"Unsupervised meta: {meta}")
    print(out.tail(10))

    # 5) plots
    if args.save_plots:
        # price
        fig = plt.figure()
        plt.plot(s.index, s.values)
        plt.title(f"{args.ticker} adjusted close")
        plt.xlabel("Date")
        plt.ylabel("Price")
        fig.savefig(out_dir / f"{args.ticker}_price.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # anomaly score
        fig = plt.figure()
        plt.plot(out.index, out["anomaly_score_norm"].values)
        ev = meta.get("explained_var_ratio_sum", float("nan"))
        k = meta.get("n_components", "-")
        plt.title(f"{args.ticker} — {args.method} anomaly ({rep}) | fit_frac={args.fit_frac}, k={k}, EV={ev:.2f}")
        plt.xlabel("Date")
        plt.ylabel("Anomaly score (normalized)")
        fig.savefig(out_dir / f"{args.ticker}_{args.method}_anomaly_{rep}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plots to: {out_dir}")

if __name__ == "__main__":
    main()
