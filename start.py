#!/usr/bin/env python3
"""
start.py
========
Single-command launcher: starts both the FastAPI backend and Streamlit frontend.
Press Ctrl+C to stop both.

Usage:
    python start.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path


def check_env():
    """Verify the .env file exists and looks configured."""
    if not Path(".env").exists():
        print("\n❌  .env file not found!")
        print("    Copy the example:  cp .env.example .env")
        print("    (No API keys needed — just set OLLAMA_MODEL)\n")
        sys.exit(1)


def check_ollama():
    """Warn if Ollama doesn't appear to be running."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"   ✅ Ollama running  — models: {models or ['(none pulled yet)']}")
            return
    except Exception:
        pass

    print("   ⚠️  Ollama not detected at localhost:11434")
    print("       → Download: https://ollama.com/download")
    print("       → Start:    ollama serve")

    from dotenv import load_dotenv
    load_dotenv()
    model = os.getenv("OLLAMA_MODEL", "llama3")
    print(f"       → Pull model: ollama pull {model}\n")
    print("   Continuing anyway — the backend will show a clear error if Ollama is missing.\n")


def main():
    print("\n" + "=" * 60)
    print("  🎙️  Voice Interview Agent  —  Offline Edition")
    print("  LLM: Ollama  |  STT: Faster-Whisper  |  TTS: pyttsx3")
    print("=" * 60 + "\n")

    check_env()
    check_ollama()

    Path("uploads").mkdir(exist_ok=True)
    Path("transcripts").mkdir(exist_ok=True)

    procs = []
    try:
        print("▶  Starting FastAPI backend  (http://127.0.0.1:8000) ...")
        backend = subprocess.Popen(
            [sys.executable, "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append(backend)
        time.sleep(2)

        print("▶  Starting Streamlit frontend  (http://localhost:8501) ...")
        frontend = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run",
             "frontend/streamlit_app.py",
             "--server.headless=true",
             "--server.port=8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append(frontend)
        time.sleep(2)

        print("\n" + "=" * 60)
        print("  ✅  Both servers are running!")
        print("  →  Open:      http://localhost:8501")
        print("  →  API docs:  http://localhost:8000/docs")
        print("  →  Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        while all(p.poll() is None for p in procs):
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n⏹  Shutting down...")
    finally:
        for p in procs:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("✅  Stopped.\n")


if __name__ == "__main__":
    main()
