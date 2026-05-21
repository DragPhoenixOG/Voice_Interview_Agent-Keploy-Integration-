"""
utils/ollama_client.py
======================
Shared Ollama client — wraps the local Ollama HTTP API for clean usage
across all agents. Ollama runs 100% on your machine with no internet
required after the initial model pull.

Install Ollama:  https://ollama.com/download
Pull a model:    ollama pull llama3
"""

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:0.5b")
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://host.docker.internal:11434"
)


def _check_ollama_running():
    """Verify Ollama server is reachable before making requests."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def _check_model_available(model: str) -> bool:
    """Check if the requested model has been pulled locally."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            # Match both "llama3" and "llama3:latest" style names
            return any(m.startswith(model.split(":")[0]) for m in models)
    except Exception:
        pass
    return False


def chat(
    messages: list,
    system_prompt: str = None,
    temperature: float = 0.7,
    max_tokens: int = 200,
    model: str = None,
) -> str:
    """
    Send a chat request to the local Ollama server and return the response text.

    Args:
        messages:      List of {"role": "user"/"assistant", "content": "..."} dicts
        system_prompt: Optional system instruction prepended to the conversation
        temperature:   Creativity (0.0 = deterministic, 1.0 = creative)
        max_tokens:    Max tokens to generate
        model:         Override the model set in .env

    Returns:
        Response text string

    Raises:
        RuntimeError if Ollama is not running or model not found
    """
    selected_model = model or OLLAMA_MODEL

    # ── Preflight checks ──────────────────────────────────────────────────────
    if not _check_ollama_running():
        raise RuntimeError(
            "Ollama is not running.\n"
            "Start it with:  ollama serve\n"
            "Download at:    https://ollama.com/download"
        )

    if not _check_model_available(selected_model):
        raise RuntimeError(
            f"Model '{selected_model}' not found locally.\n"
            f"Pull it with:  ollama pull {selected_model}\n"
            f"Available models:  ollama list"
        )

    # ── Build message list ────────────────────────────────────────────────────
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    # ── Call Ollama /api/chat ─────────────────────────────────────────────────
    try:
        payload = {
            "model": selected_model,
            "messages": full_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096
            },
        }

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,  # Local inference can be slow on CPU — generous timeout
        )

        if response.status_code != 200:
            logger.error(f"Status Code: {response.status_code}")
            logger.error(f"Raw Response: {response.text}")
            response.raise_for_status()

        result = response.json()
        logger.info(f"Ollama raw response: {result}")
        # Support both /api/chat and /api/generate formats
        if "message" in result:
            text = result["message"]["content"].strip()  
        elif "response" in result:
            text = result["response"].strip()
        else:
            raise RuntimeError(f"Unexpected Ollama response: {result}")
        logger.info(f"Ollama ({selected_model}) responded: '{text[:60]}...'")
        return text

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama timed out. The model may be too large for your hardware.\n"
            "Try a smaller model:  ollama pull llama3:8b  or  ollama pull phi3"
        )
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        raise RuntimeError(f"Ollama request failed: {e}")


def list_local_models() -> list:
    """Return a list of locally available model names."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []
