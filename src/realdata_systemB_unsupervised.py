import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from systemB_unsupervised import (
    compute_signatures_for_ticker,
    fit_corpus_signatures,
    fit_sig_maha_knn,
    leave_one_out_sig_maha_knn_scores,
    rolling_scores_to_prob,
    rolling_sig_maha_knn_scores,
    score_windows,
    scores_to_prob,
    smooth_prob,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_corpus_tickers(s: str | None) -> list[str]:
    if s is None:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def build_default_out_csv(
    root: Path,
    ticker: str,
    mode: str,
    variant: str,
    depth: int,
    window: int,
    step: int,
) -> Path:
    out_dir = root / "outputs" / "systemB_unsupervised"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace(".", "_")
    name = f"{safe_ticker}_{mode}_{variant}_d{depth}_w{window}_s{step}.csv"
    return out_dir / name


def maybe_plot(df: pd.DataFrame, ticker: str, mode: str, p0: float):
    prob_col = "bubble_prob_smooth" if "bubble_prob_smooth" in df.columns else "bubble_prob"
    fig = plt.figure()
    plt.plot(df.index, df["bubble_prob"].values, label="bubble_prob")
    if prob_col != "bubble_prob":
        plt.plot(df.index, df[prob_col].values, label=prob_col)
    plt.title(f"{ticker} - System B unsupervised bubble probability ({mode}, p0={p0})")
    plt.xlabel("Date")
    plt.ylabel("bubble_prob")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="System B unsupervised scan using signature_mahalanobis_knn (variance-norm Mahalanobis + kNN)."
    )
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--variant",
        type=str,
        default="base_sig",
        choices=["base_sig", "base_log", "ll_sig", "ll_log"],
    )
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--mode", type=str, default="self", choices=["self", "cross", "rolling"])
    parser.add_argument("--corpus_tickers", type=str, default=None)
    parser.add_argument("--rolling_ref_windows", type=int, default=150)
    parser.add_argument("--rolling_min_fit_windows", type=int, default=30)
    parser.add_argument("--rolling_min_ref_scores", type=int, default=10)

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--variance_selection", type=str, default="1sd")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--pca_type", type=str, default="svd")

    parser.add_argument("--p0", type=float, default=0.9)
    parser.add_argument("--smooth", type=int, default=5)

    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    corpus_tickers = parse_corpus_tickers(args.corpus_tickers)
    if args.mode == "cross" and not corpus_tickers:
        raise ValueError("--mode cross requires --corpus_tickers (comma-separated list)")

    root = project_root()
    end_dates, x_target = compute_signatures_for_ticker(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        window=args.window,
        step=args.step,
        depth=args.depth,
        variant=args.variant,
        device=args.device,
    )

    if args.mode in {"self", "cross"}:
        if args.mode == "self":
            scores = leave_one_out_sig_maha_knn_scores(
                x_windows=x_target,
                k=args.k,
                variance_selection=args.variance_selection,
                normalize=args.normalize,
                pca_type=args.pca_type,
                min_fit_windows=max(10, args.rolling_min_fit_windows),
            )
            ref_scores = scores[np.isfinite(scores)]
            bubble_prob = scores_to_prob(scores=scores, ref_scores=ref_scores, p0=args.p0)
        else:
            corpus = fit_corpus_signatures(
                mode=args.mode,
                target_ticker=args.ticker,
                start=args.start,
                end=args.end,
                window=args.window,
                step=args.step,
                depth=args.depth,
                variant=args.variant,
                device=args.device,
                corpus_tickers=corpus_tickers,
            )
            model = fit_sig_maha_knn(
                x_corpus=corpus,
                k=args.k,
                variance_selection=args.variance_selection,
                normalize=args.normalize,
                pca_type=args.pca_type,
            )
            scores = score_windows(model=model, x_windows=x_target, n_neighbors=args.k)
            ref_scores = score_windows(model=model, x_windows=corpus, n_neighbors=args.k)
            bubble_prob = scores_to_prob(scores=scores, ref_scores=ref_scores, p0=args.p0)
    else:
        scores = rolling_sig_maha_knn_scores(
            x_windows=x_target,
            k=args.k,
            rolling_ref_windows=args.rolling_ref_windows,
            min_fit_windows=args.rolling_min_fit_windows,
            variance_selection=args.variance_selection,
            normalize=args.normalize,
            pca_type=args.pca_type,
        )
        bubble_prob = rolling_scores_to_prob(
            scores=scores,
            p0=args.p0,
            min_ref_scores=max(1, args.rolling_min_ref_scores),
        )

    idx = pd.DatetimeIndex(end_dates, name="Date")
    out = pd.DataFrame(
        {
            "score_raw": np.asarray(scores, dtype=float),
            "bubble_prob": np.asarray(bubble_prob, dtype=float),
        },
        index=idx,
    )
    if args.smooth > 1:
        out["bubble_prob_smooth"] = smooth_prob(out["bubble_prob"].values, smooth_window=args.smooth)

    out_csv = Path(args.out_csv) if args.out_csv else build_default_out_csv(
        root=root,
        ticker=args.ticker,
        mode=args.mode,
        variant=args.variant,
        depth=args.depth,
        window=args.window,
        step=args.step,
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv)

    print(f"Saved CSV: {out_csv.resolve()}")
    print(out.tail(10))

    if args.plot:
        maybe_plot(df=out, ticker=args.ticker, mode=args.mode, p0=args.p0)


if __name__ == "__main__":
    main()
