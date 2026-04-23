from __future__ import annotations

import os
from functools import lru_cache

import numpy as np


_FORCED_BACKEND = os.environ.get("SIG_BACKEND", "").strip().lower()

_signatory = None
_torch = None
_iisignature = None

if _FORCED_BACKEND != "iisignature":
    try:
        import torch as _torch  # type: ignore[assignment]
        import signatory as _signatory  # type: ignore[assignment]
    except ImportError:
        _torch = None
        _signatory = None

if _signatory is None or _FORCED_BACKEND == "iisignature":
    try:
        import iisignature as _iisignature  # type: ignore[assignment]
    except ImportError:
        _iisignature = None

if _FORCED_BACKEND == "signatory" and _signatory is None:
    raise ImportError(
        "SIG_BACKEND=signatory was requested but signatory is not installed."
    )

if _signatory is None and _iisignature is None:
    raise ImportError(
        "No signature backend is available. Install signatory or iisignature."
    )

ACTIVE_BACKEND = "signatory" if _signatory is not None else "iisignature"


@lru_cache(maxsize=None)
def _prepared_logsignature(channels: int, depth: int):
    if _iisignature is None:
        raise RuntimeError("iisignature backend is not available")
    return _iisignature.prepare(channels, depth)


def _validate_device(device: str) -> None:
    if device != "cuda":
        return
    if _torch is None or not _torch.cuda.is_available():
        raise RuntimeError("Requested device 'cuda' but CUDA is not available.")


def compute_signature(
    paths: np.ndarray,
    depth: int,
    device: str = "cpu",
) -> np.ndarray:
    path_array = np.asarray(paths, dtype=np.float32)

    if ACTIVE_BACKEND == "signatory":
        _validate_device(device)
        tensor = _torch.from_numpy(path_array).to(device)
        with _torch.no_grad():
            feat = _signatory.signature(tensor, depth=depth)
        return feat.detach().cpu().numpy().astype(np.float32, copy=False)

    feats = [
        _iisignature.sig(path.astype(np.float64, copy=False), depth)
        for path in path_array
    ]
    return np.asarray(feats, dtype=np.float32)


def compute_logsignature(
    paths: np.ndarray,
    depth: int,
    device: str = "cpu",
) -> np.ndarray:
    path_array = np.asarray(paths, dtype=np.float32)

    if ACTIVE_BACKEND == "signatory":
        _validate_device(device)
        tensor = _torch.from_numpy(path_array).to(device)
        with _torch.no_grad():
            try:
                feat = _signatory.logsignature(tensor, depth=depth)
            except TypeError:
                feat = _signatory.logsignature(tensor, depth=depth, mode="words")
        return feat.detach().cpu().numpy().astype(np.float32, copy=False)

    prepared = _prepared_logsignature(path_array.shape[2], depth)
    feats = [
        _iisignature.logsig(path.astype(np.float64, copy=False), prepared)
        for path in path_array
    ]
    return np.asarray(feats, dtype=np.float32)


def backend_info() -> dict[str, object]:
    return {
        "active_backend": ACTIVE_BACKEND,
        "forced_backend": _FORCED_BACKEND or None,
        "signatory_available": _signatory is not None,
        "iisignature_available": _iisignature is not None,
    }
