from __future__ import annotations

import sys
import json
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


# ---------------------------
# Repo root detection
# ---------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# Add repo root to import path so `import src.*` works
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ScanConfig:
    ticker: str
    variant: str
    depth: int
    window: int
    step: int
    threshold: float
    start: str | None = None
    end: str | None = None

    # NEW:
    model_kind: str = "supervised"  # "supervised" | "iforest"
    fit_frac: float = 0.25          # iforest only
    seed: int = 0                   # iforest only
    device: str = "cpu"             # iforest only (and future)


@dataclass
class ScanResult:
    price: pd.Series                 # index datetime, values float
    probs: pd.DataFrame              # index datetime, column bubble_prob
    meta: Dict[str, Any]             # any extras (paths, timing, etc.)


# ============================================================
# Choose ONE adapter by setting USE_IMPORT_ADAPTER.
# ============================================================
USE_IMPORT_ADAPTER = True  # set False to use CLI/subprocess adapter


# ---------------------------
# Adapter A: import + call a function from your code
# ---------------------------
def _run_scan_via_import(cfg: ScanConfig) -> ScanResult:
    """
    Routes to the correct module based on cfg.model_kind.

    Expected callable name in both modules:
      run_scan_return_series(...)

    Supervised module (variants) should accept:
      ticker, variant, depth, window, step, start, end, threshold (and optionally others)

    IF module should accept:
      ticker, start, end, window, step, depth, device, fit_frac, seed
    """
    FUNC_NAME = "run_scan_return_series"

    if cfg.model_kind == "iforest":
        IMPORT_PATH = "src.realdata_systemB_iforest"
    else:
        IMPORT_PATH = "src.realdata_systemB_variants"

    module = __import__(IMPORT_PATH, fromlist=[FUNC_NAME])
    fn = getattr(module, FUNC_NAME)

    if cfg.model_kind == "iforest":
        out = fn(
            ticker=cfg.ticker,
            start=cfg.start or "2014-01-01",
            end=cfg.end,
            window=int(cfg.window),
            step=int(cfg.step),
            depth=int(cfg.depth),
            device=getattr(cfg, "device", "cpu"),
            fit_frac=float(getattr(cfg, "fit_frac", 0.25)),
            seed=int(getattr(cfg, "seed", 0)),
        )
    else:
        # Keep supervised call shape aligned with your variants script
        out = fn(
            ticker=cfg.ticker,
            variant=cfg.variant,
            depth=int(cfg.depth),
            window=int(cfg.window),
            step=int(cfg.step),
            start=cfg.start,
            end=cfg.end,
            threshold=float(cfg.threshold),
        )

    # Accept either dict or tuple
    if isinstance(out, dict):
        price = out["price"]
        probs = out["probs"]
        meta = out.get("meta", {})
    else:
        # (price, probs, meta?) style
        if len(out) == 2:
            price, probs = out
            meta = {}
        else:
            price, probs, meta = out

    # Basic validations
    if not isinstance(probs, pd.DataFrame):
        raise TypeError("Expected probs to be a pandas DataFrame")
    if "bubble_prob" not in probs.columns:
        raise ValueError("probs DataFrame must contain a 'bubble_prob' column")

    price = _ensure_datetime_index_series(price)
    probs = _ensure_datetime_index_df(probs)

    return ScanResult(price=price, probs=probs, meta=meta)


# ---------------------------
# Adapter B: call your existing CLI script and parse its outputs
# ---------------------------
def _run_scan_via_cli(cfg: ScanConfig) -> ScanResult:
    """
    Runs a CLI script and expects it to write a CSV with date + bubble_prob.
    Script path is selected based on cfg.model_kind.
    """
    gui_out = OUTPUTS_DIR / "gui_runs"
    gui_out.mkdir(parents=True, exist_ok=True)

    run_id = f"{cfg.ticker}_{cfg.model_kind}_{cfg.variant}_d{cfg.depth}_w{cfg.window}_s{cfg.step}_{int(time.time())}"
    out_csv = gui_out / f"{run_id}.csv"
    out_meta = gui_out / f"{run_id}.meta.json"

    # Choose script
    if cfg.model_kind == "iforest":
        SCRIPT_PATH = SRC_DIR / "realdata_systemB_iforest.py"
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--ticker", cfg.ticker,
            "--window", str(cfg.window),
            "--step", str(cfg.step),
            "--depth", str(cfg.depth),
            "--fit_frac", str(getattr(cfg, "fit_frac", 0.25)),
            "--seed", str(getattr(cfg, "seed", 0)),
            "--device", str(getattr(cfg, "device", "cpu")),
            "--out_dir", str(gui_out),
        ]
    else:
        SCRIPT_PATH = SRC_DIR / "realdata_systemB_variants.py"
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--ticker", cfg.ticker,
            "--variant", cfg.variant,
            "--depth", str(cfg.depth),
            "--window", str(cfg.window),
            "--step", str(cfg.step),
            "--threshold", str(cfg.threshold),
            "--out_csv", str(out_csv),
        ]

    if cfg.start:
        cmd += ["--start", cfg.start]
    if cfg.end:
        cmd += ["--end", cfg.end]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "CLI scan failed.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    # Supervised expects out_csv, IF script as written may not output CSV by default.
    # If you're using CLI adapter for IF, modify IF script to accept --out_csv.
    if not out_csv.exists():
        # Try to discover IF output CSV if script wrote one in out_dir
        # Otherwise fail with clear message.
        raise FileNotFoundError(
            f"Expected output CSV not found at: {out_csv}\n"
            "If using CLI adapter, ensure your script writes CSV via --out_csv "
            "(recommended), or keep USE_IMPORT_ADAPTER=True."
        )

    probs = pd.read_csv(out_csv)
    if "date" not in probs.columns or "bubble_prob" not in probs.columns:
        raise ValueError("CSV must have columns: 'date', 'bubble_prob'")

    probs["date"] = pd.to_datetime(probs["date"])
    probs = probs.set_index("date").sort_index()

    if "price" in probs.columns:
        price = probs["price"].copy()
        probs = probs[["bubble_prob"]].copy()
    else:
        price = pd.Series(index=probs.index, data=[float("nan")] * len(probs), name="price")

    meta = {
        "run_id": run_id,
        "out_csv": str(out_csv),
        "stdout_tail": proc.stdout[-1000:],
        "model_kind": cfg.model_kind,
    }

    out_meta.write_text(json.dumps(meta, indent=2))
    return ScanResult(price=_ensure_datetime_index_series(price), probs=_ensure_datetime_index_df(probs), meta=meta)


def run_scan(cfg: ScanConfig) -> ScanResult:
    if USE_IMPORT_ADAPTER:
        return _run_scan_via_import(cfg)
    return _run_scan_via_cli(cfg)


# ---------------------------
# Helpers
# ---------------------------
def _ensure_datetime_index_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _ensure_datetime_index_series(s: pd.Series) -> pd.Series:
    out = s.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()
