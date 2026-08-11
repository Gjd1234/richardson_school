"""LLM factory with free-model + multi-key rotation.

Uses the same free models and endpoint as the sibling `~/litellm` repo
(opencode.ai/zen/v1 + OPENCODE_API_KEY). A `RotatingChatModel` proxies
LangChain's ChatOpenAI and rotates through API keys (OPENCODE_API_KEYS as a
comma-separated list, else OPENCODE_API_KEY) and then through the model list
whenever a call fails or returns an empty result, so the daily job stays
resilient when a free model or key is rate-limited or goes down.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from functools import partial
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI


def _load_env_file(path: str | pathlib.Path | None) -> None:
    if not path:
        return
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k = k.strip()
            if v and not v.startswith("$") and k not in os.environ:
                os.environ[k] = v


def _env_dirs() -> list[pathlib.Path]:
    here = pathlib.Path(__file__).resolve().parent.parent
    return [here / ".env",
            pathlib.Path("~/litellm/.env").expanduser(),
            pathlib.Path("~/.litellm/.env").expanduser()]


def _load_env() -> None:
    for p in _env_dirs():
        _load_env_file(p)


def api_keys() -> list[str]:
    _load_env()
    raw = os.environ.get("OPENCODE_API_KEYS") or os.environ.get("OPENCODE_API_KEY", "")
    keys = [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]
    if not keys:
        raise RuntimeError(
            "OPENCODE_API_KEY not set. Add it to env or a .env file (same key ~/litellm uses). "
            "For multi-key load balancing, set OPENCODE_API_KEYS as a comma-separated list."
        )
    return keys


def api_key() -> str:
    return api_keys()[0]


def llm_config() -> dict:
    here = pathlib.Path(__file__).resolve().parent.parent
    cfg_path = here / "data" / "college-config.json"
    return json.loads(cfg_path.read_text()).get("llm", {})


def _rotation() -> list[str]:
    return (llm_config().get("rotation") or
            ["deepseek-v4-flash-free", "nemotron-3-ultra-free", "mimo-v2.5-free",
             "big-pickle", "ling-3.0-flash-free", "ling-3.0-tiny-free"])


def _client_for(model: str, key: str, temperature: float, **kw: Any) -> ChatOpenAI:
    base = llm_config().get("api_base", "https://opencode.ai/zen/v1")
    opts: dict[str, Any] = {
        "model": model,
        "base_url": base,
        "api_key": key,
        "temperature": temperature,
        "max_tokens": kw.pop("max_tokens", 3072),
        "timeout": kw.pop("timeout", 120),
        **kw,
    }
    return ChatOpenAI(**opts)


class RotatingChatModel(BaseChatModel):
    """LangChain chat model that rotates through API keys AND free models on failure.

    Ordering: keys first (load-balance / failover across accounts), then models
    within each key, mirroring the OpenCode Zen multi-key rotation feature request.
    """

    rotation: tuple[str, ...] = tuple(_rotation())
    temperature: float = 0.2
    role: str = "research"
    model_kwargs: dict = {}
    _cache: dict[str, ChatOpenAI] = {}

    def _client(self, model: str, key: str) -> ChatOpenAI:
        ckey = f"{model}|{key[:8]}|{self.temperature}|{sorted(self.model_kwargs.items())}"
        if ckey not in self._cache:
            self._cache[ckey] = _client_for(model, key, self.temperature, **self.model_kwargs)
        return self._cache[ckey]

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        last_err: Exception | None = None
        for key in api_keys():  # rotate across keys (failover on rate limits)
            for model in self.rotation:
                try:
                    result = self._client(model, key)._generate(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    )
                    gen = result.generations[0] if result.generations else None
                    if gen is None:
                        raise RuntimeError("empty generation")
                    text = getattr(gen, "text", None)
                    if not str(text or "").strip():
                        # ChatGeneration carries the message content
                        content = (gen.message.content or "") if getattr(gen, "message", None) else ""
                        if not str(content).strip():
                            raise RuntimeError("empty content")
                    return result
                except Exception as e:  # noqa: BLE001 - rotate on any transient failure
                    last_err = e
                    time.sleep(0.9)
        if last_err:
            raise last_err
        raise RuntimeError("All free models returned empty output.")

    @property
    def _llm_type(self) -> str:
        return "rotating-free"

    @property
    def _identifying_params(self) -> dict:
        return {"rotation": list(self.rotation), "role": self.role, "temperature": self.temperature}


def get_model(role: str = "research", temperature: float = 0.2, **kw: Any) -> RotatingChatModel:
    """Return a RotatingChatModel bound to the given role's preferred tier."""
    return RotatingChatModel(role=role, temperature=temperature, model_kwargs=kw)