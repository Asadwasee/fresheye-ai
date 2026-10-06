"""
core/api_keys.py
=================
Single source of truth for every secret / API key the app needs
(currently: the key for the open-source LLM provider used in the RAG
pipeline and the agent).

Design goals
------------
1. Secrets never live in source code - they are loaded from environment
   variables (populated from a local, git-ignored ".env" file).
2. One place to validate that required keys are present before the app
   starts serving requests, so failures happen at boot, not mid-request.
3. Easy to extend if you add more providers later (Together AI, Fireworks,
   HuggingFace Inference Endpoints, OpenRouter, etc.) - just add another
   `get_x_key()` function.

Usage
-----
    from core.api_keys import get_groq_api_key, LLM_PROVIDER

This module is intentionally dependency-free (only stdlib + python-dotenv)
so it can be imported very early in the app lifecycle.
"""
import os
from dotenv import load_dotenv

# Load variables from a local .env file (if present) into os.environ.
# In production you would instead inject real environment variables via
# your hosting platform's secret manager (Docker secrets, Kubernetes
# secrets, AWS Secrets Manager, etc.) and skip the .env file entirely.
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()


def get_groq_api_key() -> str | None:
    """Return the Groq API key used to call open-source Llama models
    hosted on Groq's low-latency inference infrastructure."""
    return os.getenv("GROQ_API_KEY", "").strip() or None


def get_ollama_base_url() -> str:
    """Return the base URL of a locally running Ollama server (fully
    open-source, fully local - no API key required)."""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()


def validate_llm_config() -> tuple[bool, str]:
    """Sanity-check that the selected LLM provider is actually usable.
    Returns (is_valid, message). The app logs a warning (not a crash) if
    this fails, and RAG / agent features degrade gracefully.
    """
    if LLM_PROVIDER == "groq":
        if not get_groq_api_key():
            return False, (
                "LLM_PROVIDER is 'groq' but GROQ_API_KEY is missing. "
                "Add it to your .env file, or switch LLM_PROVIDER=ollama "
                "to use a fully local model instead."
            )
        return True, "Groq API key detected."
    elif LLM_PROVIDER == "ollama":
        return True, f"Using local Ollama at {get_ollama_base_url()}."
    else:
        return False, f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'."
