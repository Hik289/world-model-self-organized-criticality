"""Shared OpenAI-compatible LLM configuration."""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

DEFAULT_LLM_API_BASE_URL = "https://YOUR_LLM_API_BASE_URL/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_KEY_PATHS = (
    _PROJECT_ROOT / ".secrets" / "llm.key",
    _PROJECT_ROOT / ".secrets" / "azure.key",
)


def get_llm_api_base_url() -> str:
    return (
        os.environ.get("LLM_API_BASE_URL")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or DEFAULT_LLM_API_BASE_URL
    )


def get_llm_model() -> str:
    return os.environ.get("LLM_MODEL") or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or DEFAULT_LLM_MODEL


def load_llm_api_key(key_path: str | None = None) -> str:
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    if api_key:
        return api_key

    configured_path = (
        key_path
        or os.environ.get("WORLDMODELSOC_LLM_KEY_PATH")
        or os.environ.get("WORLDMODELSOC_AZURE_KEY_PATH")
    )
    if configured_path:
        return Path(configured_path).read_text(encoding="utf-8").strip()

    for path in _DEFAULT_KEY_PATHS:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

    raise RuntimeError(
        "Set LLM_API_KEY, WORLDMODELSOC_LLM_KEY_PATH, or place a key at .secrets/llm.key."
    )


LLM_API_BASE_URL = get_llm_api_base_url()
LLM_MODEL = get_llm_model()


def make_openai_client(key_path: str | None = None) -> OpenAI:
    return OpenAI(base_url=LLM_API_BASE_URL, api_key=load_llm_api_key(key_path))
