from __future__ import annotations

import inspect
from typing import Iterable

import numpy as np
import pandas as pd

from realdata_systemB_variants import (
    fetch_adj_close,
    features_from_paths,
    make_paths_from_series,
    variant_to_config,
)


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"Empty array not allowed, got shape={arr.shape}")
    return arr


def compute_signatures_for_ticker(
    ticker: str,
    start: str,
    end: str | None,
    window: int,
    step: int,
    depth: int,
    variant: str,
    device: str = "cpu",
) -> tuple[list[pd.Timestamp], np.ndarray]:
    leadlag, transform = variant_to_config(variant)
    close = fetch_adj_close(ticker=ticker, start=start, end=end)
    paths, end_dates = make_paths_from_series(
        close,
        window=int(window),
        step=int(step),
        leadlag=leadlag,
    )
    x_sig = features_from_paths(
        paths=paths,
        depth=int(depth),
        transform=transform,
        device=device,
    )
    return end_dates, _as_2d_float(x_sig)


def fit_corpus_signatures(
    mode: str,
    target_ticker: str,
    start: str,
    end: str | None,
    window: int,
    step: int,
    depth: int,
    variant: str,
    device: str = "cpu",
    corpus_tickers: Iterable[str] | None = None,
) -> np.ndarray:
    mode = mode.lower().strip()
    if mode == "self":
        _, x_self = compute_signatures_for_ticker(
            ticker=target_ticker,
            start=start,
            end=end,
            window=window,
            step=step,
            depth=depth,
            variant=variant,
            device=device,
        )
        return x_self

    if mode == "cross":
        if corpus_tickers is None:
            raise ValueError("mode='cross' requires corpus_tickers")
        xs: list[np.ndarray] = []
        for t in corpus_tickers:
            t = t.strip()
            if not t:
                continue
            _, x_t = compute_signatures_for_ticker(
                ticker=t,
                start=start,
                end=end,
                window=window,
                step=step,
                depth=depth,
                variant=variant,
                device=device,
            )
            xs.append(x_t)
        if not xs:
            raise ValueError("No valid corpus_tickers windows were produced")
        return _as_2d_float(np.vstack(xs))

    raise ValueError("mode must be one of: self, cross")


def _load_sig_maha_knn_classes():
    try:
        from signature_mahalanobis_knn.sig_mahal_knn import SignatureMahalanobisKNN

        return SignatureMahalanobisKNN
    except ImportError as exc:
        try:
            from signature_mahalanobis_knn.sig_mahal_knn import SigMahaKNN

            return SigMahaKNN
        except ImportError:
            raise ImportError(
                "Missing dependency 'signature_mahalanobis_knn'. "
                "Install with: pip install signature_mahalanobis_knn"
            ) from exc


def _call_fit(model, x_corpus: np.ndarray) -> None:
    sig = inspect.signature(model.fit)
    params = set(sig.parameters.keys())
    if "signatures_train" in params:
        model.fit(signatures_train=x_corpus)
        return
    model.fit(x_corpus)


def _call_predict(model, x_windows: np.ndarray, n_neighbors: int | None = None) -> np.ndarray:
    if hasattr(model, "conformance"):
        sig = inspect.signature(model.conformance)
        params = set(sig.parameters.keys())
        kwargs = {}
        if n_neighbors is None:
            n_neighbors = int(getattr(model, "_sigmk_n_neighbors", 5))
        if "n_neighbors" in params:
            kwargs["n_neighbors"] = int(max(1, n_neighbors))
        if "return_indices" in params:
            kwargs["return_indices"] = False
        if "signatures_test" in params:
            return np.asarray(model.conformance(signatures_test=x_windows, **kwargs), dtype=float).reshape(-1)
        return np.asarray(model.conformance(x_windows, **kwargs), dtype=float).reshape(-1)

    sig = inspect.signature(model.predict)
    params = set(sig.parameters.keys())
    if "signatures_test" in params:
        return np.asarray(model.predict(signatures_test=x_windows), dtype=float).reshape(-1)
    return np.asarray(model.predict(x_windows), dtype=float).reshape(-1)


def fit_sig_maha_knn(
    x_corpus: np.ndarray,
    k: int = 5,
    variance_selection: str = "1sd",
    normalize: bool = False,
    pca_type: str = "svd",
):
    x_corpus = _as_2d_float(x_corpus)
    cls = _load_sig_maha_knn_classes()

    ctor_sig = inspect.signature(cls)
    ctor_params = set(ctor_sig.parameters.keys())
    kwargs = {}
    if "k" in ctor_params:
        kwargs["k"] = int(k)
    if "variance_selection" in ctor_params:
        kwargs["variance_selection"] = variance_selection
    if "normalize" in ctor_params:
        kwargs["normalize"] = bool(normalize)
    if "pca_type" in ctor_params:
        kwargs["pca_type"] = pca_type

    model = cls(**kwargs)
    setattr(model, "_sigmk_n_neighbors", int(max(1, k)))
    _call_fit(model, x_corpus)
    return model


def score_windows(model, x_windows: np.ndarray, n_neighbors: int | None = None) -> np.ndarray:
    x_windows = _as_2d_float(x_windows)
    scores = _call_predict(model, x_windows=x_windows, n_neighbors=n_neighbors)
    if scores.shape[0] != x_windows.shape[0]:
        raise ValueError(
            f"Unexpected score shape {scores.shape}; expected first dim {x_windows.shape[0]}"
        )
    return scores.astype(float, copy=False)


def percentile_rank(scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1)
    r = np.asarray(ref_scores, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.full(s.shape, np.nan, dtype=float)
    sorted_ref = np.sort(r)
    return np.searchsorted(sorted_ref, s, side="right") / float(sorted_ref.size)


def scores_to_prob(scores: np.ndarray, ref_scores: np.ndarray, p0: float = 0.9) -> np.ndarray:
    if not (0.0 <= p0 < 1.0):
        raise ValueError("p0 must satisfy 0 <= p0 < 1")
    p = percentile_rank(scores=scores, ref_scores=ref_scores)
    denom = max(1.0 - float(p0), 1e-12)
    return np.clip((p - float(p0)) / denom, 0.0, 1.0)


def smooth_prob(prob: np.ndarray, smooth_window: int) -> np.ndarray:
    p = np.asarray(prob, dtype=float).reshape(-1)
    if smooth_window <= 1:
        return p
    return pd.Series(p).rolling(window=int(smooth_window), min_periods=1).mean().to_numpy(dtype=float)


def rolling_sig_maha_knn_scores(
    x_windows: np.ndarray,
    k: int = 5,
    rolling_ref_windows: int = 150,
    min_fit_windows: int | None = None,
    variance_selection: str = "1sd",
    normalize: bool = False,
    pca_type: str = "svd",
) -> np.ndarray:
    x_windows = _as_2d_float(x_windows)
    n = x_windows.shape[0]
    out = np.full(n, np.nan, dtype=float)

    if min_fit_windows is None:
        min_fit_windows = max(30, int(k) + 1)

    for i in range(n):
        lo = max(0, i - int(rolling_ref_windows))
        x_ref = x_windows[lo:i]
        if x_ref.shape[0] < int(min_fit_windows):
            continue
        model = fit_sig_maha_knn(
            x_corpus=x_ref,
            k=k,
            variance_selection=variance_selection,
            normalize=normalize,
            pca_type=pca_type,
        )
        out[i] = float(score_windows(model, x_windows[i : i + 1], n_neighbors=k)[0])
    return out


def rolling_scores_to_prob(
    scores: np.ndarray,
    p0: float = 0.9,
    min_ref_scores: int = 30,
) -> np.ndarray:
    if not (0.0 <= p0 < 1.0):
        raise ValueError("p0 must satisfy 0 <= p0 < 1")
    s = np.asarray(scores, dtype=float).reshape(-1)
    out = np.full(s.shape, np.nan, dtype=float)
    hist: list[float] = []
    denom = max(1.0 - float(p0), 1e-12)

    for i, v in enumerate(s):
        if not np.isfinite(v):
            continue
        if len(hist) >= int(min_ref_scores):
            p = percentile_rank(np.asarray([v], dtype=float), np.asarray(hist, dtype=float))[0]
            out[i] = float(np.clip((p - float(p0)) / denom, 0.0, 1.0))
        hist.append(float(v))
    return out


def leave_one_out_sig_maha_knn_scores(
    x_windows: np.ndarray,
    k: int = 5,
    variance_selection: str = "1sd",
    normalize: bool = False,
    pca_type: str = "svd",
    min_fit_windows: int = 30,
) -> np.ndarray:
    x_windows = _as_2d_float(x_windows)
    n = x_windows.shape[0]
    out = np.full(n, np.nan, dtype=float)
    min_fit = int(max(1, min_fit_windows))

    for i in range(n):
        x_ref = np.delete(x_windows, i, axis=0)
        if x_ref.shape[0] < min_fit:
            continue
        model = fit_sig_maha_knn(
            x_corpus=x_ref,
            k=k,
            variance_selection=variance_selection,
            normalize=normalize,
            pca_type=pca_type,
        )
        out[i] = float(score_windows(model, x_windows[i : i + 1], n_neighbors=k)[0])
    return out
