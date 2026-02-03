## 0) Frozen assumptions (do not change)

* **Path representation (per window):** channels are
  $$
  (t \in [0,1],\ \log(\text{price}/\text{price}_0))
  $$

* **Signature library:** `signatory`
* **Classifier:** Logistic Regression
* **Sliding-window logic:** already validated

## 1) Repository flow 

**Training time (synthetic):**

1. Generate / prepare **System B paths** (synthetic SDE price paths → rolling-window paths).
2. Compute **signature features** (depth = 3 / 4).
3. Train **Logistic Regression** classifier (save scaler + model).
4. Validate on held-out synthetic test split.

**Inference time (real data):**

1. Pull historical adjusted close for ticker(s).
2. Slide a rolling window (`window`, `step`).
3. Convert each window into a path with the frozen representation.
4. Compute signatures (same depth).
5. Apply scaler + classifier → `P(bubble)` per window.
6. Summarize / plot results.

Tested on real Indian stocks (examples): **YES Bank, Suzlon** (bubble-ish), **HDFC Bank, ITC** (non-bubble), **Vodafone Idea** (sideways adversarial).

---

## 2) Setup

### 2.1 Create environment + install deps

```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -U pip
pip install numpy pandas scikit-learn joblib matplotlib tqdm yfinance
pip install signatory torch
```

Notes:

* If you want GPU signature computation, you need a CUDA-enabled PyTorch + `signatory` that matches it.
* The “frozen pipeline” runs fine on CPU (just slower).

---

## 3) Key commands (the ones you already standardized)

These are explicitly listed as the canonical commands in `FLOW_STATE.md`:

```bash
python src/compute_systemB_signatures.py --depth X
python src/train_systemB_classifier.py --depth X
python src/realdata_systemB.py .
python src/realdata_systemB_batch_scan.py .
```

Where `X` is typically `3` or `4` (both depths are already trained/saved per your note).

---

## 4) File-by-file (System B only)

### `src/compute_systemB_signatures.py`

**Purpose:** Compute signature features for System B training splits (train/val/test) at a chosen depth.

**Input:** prebuilt System B paths for each split
**Output:** feature matrices + metadata in a processed directory

**Command:**

```bash
python src/compute_systemB_signatures.py --depth 3
python src/compute_systemB_signatures.py --depth 4
```

### `src/train_systemB_classifier.py`

**Purpose:** Train the frozen System B classifier: `StandardScaler + LogisticRegression`.

**Input:** computed signature features for train/val/test
**Output:** saved scaler + classifier artifacts (depth-specific)

**Command:**

```bash
python src/train_systemB_classifier.py --depth 3
python src/train_systemB_classifier.py --depth 4
```

### `src/realdata_systemB.py`

**Purpose:** Run System B on a single real ticker (rolling windows) to output a time series of bubble probabilities.

**Input:** ticker (real market series), trained model artifacts
**Output:** per-window probabilities + (optionally) plots/CSV

**Command (typical usage):**

```bash
python src/realdata_systemB.py --depth 3 --ticker "YESBANK.NS"
```

### `src/realdata_systemB_batch_scan.py`

**Purpose:** Batch scan many tickers and output a ranked CSV summary (and optional plots).

Key args (from the script): depth/window/step/start/end + tickers list + output paths.

**Command examples:**

```bash
# Scan explicit list
python src/realdata_systemB_batch_scan.py \
  --depth 3 \
  --tickers "YESBANK.NS,SUZLON.NS,HDFCBANK.NS,ITC.NS" \
  --window 252 --step 5 \
  --start 2015-01-01 \
  --out_csv outputs/systemB_scan.csv

# Scan from file (one ticker per line, # comments allowed)
python src/realdata_systemB_batch_scan.py \
  --depth 3 \
  --tickers_file data/tickers_nse.txt \
  --save_plots \
  --plots_dir outputs/systemB_plots
```

What it writes:

* `outputs/systemB_scan.csv` by default
* Optional plots per ticker:

  * `outputs/systemB_plots/<TICKER>_price.png`
  * `outputs/systemB_plots/<TICKER>_bubbleprob.png`

### `src/compute_systemB_signatures_mod.py` (comparison / ablation helper)

**Purpose:** Compute multiple feature variants for comparison:

* base signature
* base logsignature
* lead-lag signature
* lead-lag logsignature

Key args + defaults:

* `--depth` (default 3)
* `--device cpu|cuda`
* `--batch_size` (default 512)
* `--in_dir` (default `data/processed/systemB_paths`)
* `--out_dir` (default `data/processed/systemB_signatures_mod`)
* `--variants all|base_sig|base_log|ll_sig|ll_log`

**Command examples:**

```bash
# compute all variants (base/logsig + leadlag/logsig)
python src/compute_systemB_signatures_mod.py --depth 3 --variants all

# compute only lead-lag logsig on GPU
python src/compute_systemB_signatures_mod.py \
  --depth 4 --device cuda --batch_size 1024 \
  --variants ll_log
```

### `src/realdata_systemB_variants.py` (real-data path construction variants)

**Purpose:** Defines exactly how real rolling windows are turned into paths for:

* **Base path:** `(t, log(price/price0))`
* **Lead-lag path:** `(t, lead, lag)` where lead/lag are built from the same log-price series

This is the canonical place to look if you’re debugging “why does a window become shape (L,2) or (2L-1,3)?”

---

## 5) Outputs you should expect

### Trained artifacts

Per `FLOW_STATE.md`, **depth 3 and depth 4 models are trained and saved** (don’t retrain unless you intentionally want to change things).

### Batch scan summary

A CSV where each ticker includes summary stats like:

* latest probability
* max/mean over last year
* fraction of windows above thresholds
* number of windows scanned

---

## 6) Sanity checklist (quick “is this working?”)

1. `compute_systemB_signatures.py --depth 3` runs without errors.
2. `train_systemB_classifier.py --depth 3` produces saved artifacts.
3. `realdata_systemB_batch_scan.py --depth 3 --tickers ...` produces:

   * `outputs/systemB_scan.csv`
   * (optional) plots if `--save_plots` is set.

If any of these fail, **don’t refactor** — fix inputs/paths first. 
---
