"""Secure configuration loader for local environment variables and Streamlit secrets."""

from pathlib import Path
from typing import Mapping, Optional
import os

from dotenv import load_dotenv

SUPPORTED_PROVIDERS = ("gemini", "openrouter")
DEFAULT_PROVIDER = "gemini"
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openrouter": "google/gemini-2.5-flash",
}
PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
PROVIDER_MODEL_ENV = {
    "gemini": "GEMINI_MODEL",
    "openrouter": "OPENROUTER_MODEL",
}

_ENV_LOADED = False


def init_env() -> None:
    """Load local .env file into process environment without overriding existing vars."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
        _ENV_LOADED = True


def _get_secret(secrets: Optional[Mapping], key: str) -> Optional[str]:
    """Safely read a key from Streamlit secrets without raising on missing secrets file."""
    if secrets is None:
        return None
    try:
        if key in secrets and secrets[key]:
            val = str(secrets[key]).strip()
            return val or None
    except Exception:
        return None
    return None


def get_setting(key: str, default: str = "", secrets: Optional[Mapping] = None) -> str:
    """Resolve a non-secret configuration value from Streamlit secrets or environment."""
    init_env()
    secret_val = _get_secret(secrets, key)
    if secret_val is not None:
        return secret_val
    env_val = os.environ.get(key, "").strip()
    return env_val or default


def get_default_provider(secrets: Optional[Mapping] = None) -> str:
    """Return configured default LLM provider ('gemini' or 'openrouter')."""
    raw = get_setting("LLM_PROVIDER", DEFAULT_PROVIDER, secrets=secrets).lower()
    return raw if raw in SUPPORTED_PROVIDERS else DEFAULT_PROVIDER


def get_model_for_provider(provider: str, secrets: Optional[Mapping] = None) -> str:
    """Return configured model name for the selected provider."""
    norm = provider.lower()
    if norm not in SUPPORTED_PROVIDERS:
        norm = DEFAULT_PROVIDER
    env_key = PROVIDER_MODEL_ENV[norm]
    return get_setting(env_key, DEFAULT_MODELS[norm], secrets=secrets)


def resolve_provider_api_key(
    provider: str,
    user_override: str = "",
    secrets: Optional[Mapping] = None,
) -> Optional[str]:
    """
    Resolve the API key for the selected provider.

    Priority:
      1. Optional sidebar user override (in-memory only)
      2. Streamlit Community Cloud secrets
      3. Local environment variables (.env loaded via python-dotenv)
    """
    if user_override and user_override.strip():
        return user_override.strip()

    init_env()
    norm = provider.lower()
    env_key = PROVIDER_KEY_ENV.get(norm)
    if not env_key:
        return None

    secret_val = _get_secret(secrets, env_key)
    if secret_val:
        return secret_val

    env_val = os.environ.get(env_key, "").strip()
    return env_val or None


def get_missing_key_message(provider: str) -> str:
    """Return a safe, non-sensitive configuration error message for a missing key."""
    norm = provider.lower()
    env_key = PROVIDER_KEY_ENV.get(norm, "API_KEY")
    label = "Google Gemini API" if norm == "gemini" else "OpenRouter"
    return (
        f"Configuration required: `{env_key}` is not set for **{label}**. "
        f"Add `{env_key}` to your local `.env` file, configure it in Streamlit Cloud Secrets, "
        f"or enter a temporary key in the sidebar."
    )
