# Quant Bubbles — Interactive Inspector

This is a lightweight interactive tool for inspecting bubble likelihood and anomaly signals in equity price series using path-signature–based methods.

The interface allows exploration across different stocks, window sizes, model variants, and detection paradigms, with all results shown as rolling time-series aligned to market data.


## What this tool does

- Computes **rolling bubble likelihood scores** from price paths using pre-trained signature-based classifiers
- Supports multiple **signature variants** (base, lead–lag, logsignature)
- Includes an **unsupervised baseline** (Isolation Forest) for anomaly detection
- Allows **window-size sensitivity analysis** and cross-stock comparisons
- Provides clean visualizations and CSV export of the displayed results

This tool is intended for **exploratory analysis and qualitative inspection**, not for trading or prediction.

---

## Models included

### Supervised (Signature-based)
- Path signature features computed on rolling price windows
- Logistic regression classifiers trained on synthetic data
- Variants:
  - Base signature
  - Lead–lag signature
  - Logsignature
  - Lead–lag + logsignature

### Unsupervised (Baseline)
- Isolation Forest applied to signature features
- Used as a reference to contrast supervised bubble likelihood signals

---

## Folder structure (relevant parts)

```

QuantBubblesInspector/
├── app.py
├── backend.py
├── src/
│   ├── realdata_systemB_variants.py
│   └── realdata_systemB_isolation_forest.py
├── models/
│   └── systemB/
├── universes/
│   └── universes.json

```

---

## Setup

This project uses a virtual environment created via a provided setup script.

### Requirements
- Python 3.9 or newer
- Windows (tested)

### Installation
From the project root directory:

```bat
setup.bat
```

This will:

* Create a virtual environment
* Install required dependencies

---

## Running the application

After setup completes:

```bat
streamlit run app.py
```

A browser window will open with the interactive interface.

---

## Notes

* All plots and CSV exports correspond to the **currently visible time range**
* No data or results are written to disk automatically
* The tool assumes an active internet connection for fetching market data

---

## Disclaimer

This software is for **research and educational purposes only**.
It does not constitute financial advice or a trading system.
