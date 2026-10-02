"""Enchaîne tout le pipeline : backtest, rapport, puis indique comment lancer le dashboard."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for script in ("run_backtest.py", "make_report.py"):
    print(f"\n>>> {script}")
    subprocess.run([sys.executable, str(HERE / script)], check=True)

print("\nTerminé. Lance le dashboard avec :\n    streamlit run app/streamlit_app.py")
