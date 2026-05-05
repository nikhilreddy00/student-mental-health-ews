"""
Student Mental Health Early Warning System
Starts the FastAPI backend server.

Usage:
  python code/dashboard.py
  # Then open http://localhost:5173 for the React frontend
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    api_path = Path(__file__).parent.parent / "app" / "backend"
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "api:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=str(api_path),
    )
