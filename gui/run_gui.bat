@echo off
setlocal

REM Run from gui folder
cd /d %~dp0

REM Create venv for GUI
if not exist .venv_gui (
  python -m venv .venv_gui
)

call .venv_gui\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements_gui.txt

REM Launch Streamlit
streamlit run app.py

endlocal
