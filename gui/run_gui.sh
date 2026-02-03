#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv_gui" ]; then
  python3 -m venv .venv_gui
fi

source .venv_gui/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_gui.txt

streamlit run app.py
