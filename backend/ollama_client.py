import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from ollama import Client

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)
load_dotenv(override=False)


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen3:8b")


def get_ollama_timeout() -> float:
    timeout_str = os.getenv("OLLAMA_TIMEOUT", "180.0")
    try:
        return float(timeout_str)
    except ValueError:
        return 180.0


def get_ollama_client() -> Client:
    return Client(host=get_ollama_base_url(), timeout=get_ollama_timeout())


def generate_chat_response(user_message: str, model_name: str | None = None) -> str:
    if not user_message or not user_message.strip():
        raise ValueError("User message cannot be empty.")

    client = get_ollama_client()
    response = client.chat(
        model=model_name or get_ollama_model(),
        messages=[{"role": "user", "content": user_message.strip()}],
    )

    if isinstance(response, dict):
        message = response.get("message", {}) or {}
        content = message.get("content", "") if isinstance(message, dict) else ""
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", "") if message is not None else ""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty response.")

    return content.strip()


def chat_with_ollama(
    messages: list[dict[str, str]],
    model_name: str | None = None,
    options: dict[str, Any] | None = None,
    think: bool | None = False,
) -> str:
    """Send structured messages to Ollama and return the text content."""
    client = get_ollama_client()
    kwargs: dict[str, Any] = {
        "model": model_name or get_ollama_model(),
        "messages": messages,
    }
    if options is not None:
        kwargs["options"] = options
    if think is not None:
        kwargs["think"] = think

    response = client.chat(**kwargs)

    if isinstance(response, dict):
        message = response.get("message", {}) or {}
        content = message.get("content", "") if isinstance(message, dict) else ""
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", "") if message is not None else ""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty response.")

    return content.strip()
