"""Shared OpenAI-compatible LLM configuration."""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

DEFAULT_LLM_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_KEY_PATHS = (
    _PROJECT_ROOT / ".secrets" / "llm.key",
    _PROJECT_ROOT / ".secrets" / "azure.key",
)


def get_llm_api_base_url() -> str:
    configured = os.environ.get("LLM_API_BASE_URL")
    if configured:
        return configured.rstrip("/")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        endpoint = azure_endpoint.rstrip("/")
        if not endpoint.endswith("/openai/v1"):
            endpoint += "/openai/v1"
        return endpoint
    return DEFAULT_LLM_API_BASE_URL


def get_llm_model() -> str:
    return os.environ.get("LLM_MODEL") or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or DEFAULT_LLM_MODEL


def _read_nonempty_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"LLM API key file is empty: {path}")
    return key


def load_llm_api_key(key_path: str | None = None) -> str:
    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
    )
    if api_key:
        return api_key

    configured_path = (
        key_path
        or os.environ.get("WORLDMODELSOC_LLM_KEY_PATH")
        or os.environ.get("WORLDMODELSOC_AZURE_KEY_PATH")
    )
    if configured_path:
        return _read_nonempty_key(Path(configured_path).expanduser())

    for path in _DEFAULT_KEY_PATHS:
        if path.exists():
            return _read_nonempty_key(path)

    raise RuntimeError(
        "Set LLM_API_KEY or OPENAI_API_KEY, configure "
        "WORLDMODELSOC_LLM_KEY_PATH, or place a key at .secrets/llm.key."
    )


LLM_API_BASE_URL = get_llm_api_base_url()
LLM_MODEL = get_llm_model()


def make_openai_client(key_path: str | None = None) -> OpenAI:
    return OpenAI(
        base_url=get_llm_api_base_url(),
        api_key=load_llm_api_key(key_path),
    )
