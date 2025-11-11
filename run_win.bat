@echo off
setlocal
set PY_CMD=python
where py >nul 2>nul && set PY_CMD=py -3.12
if not exist .venv (
  %PY_CMD% -m venv .venv
)
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
endlocal
