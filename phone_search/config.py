"""Settings, overridable with environment variables (PS_*).

Defaults are tuned for the Nothing Phone (3) with Qwen3.5-4B on the Hexagon NPU,
based on the speed and quality measurements in the eval (~/src/llm-eval).
"""
import os
from dataclasses import dataclass, field


def _env(name, default, cast=str):
    value = os.environ.get(f"PS_{name}")
    return default if value is None else cast(value)


@dataclass
class Config:
    # llama-server (OpenAI-compatible) endpoint
    llm_url: str = field(default_factory=lambda: _env("LLM_URL", "http://127.0.0.1:8080"))

    # Search
    max_searches: int = field(default_factory=lambda: _env("MAX_SEARCHES", 3, int))   # per user question
    results_per_search: int = field(default_factory=lambda: _env("RESULTS_PER_SEARCH", 6, int))
    pages_to_fetch: int = field(default_factory=lambda: _env("PAGES_TO_FETCH", 5, int))
    fetch_timeout: float = field(default_factory=lambda: _env("FETCH_TIMEOUT", 8.0, float))
    browser_fetches: int = field(default_factory=lambda: _env("BROWSER_FETCHES", 2, int))  # slow-path retries

    # Sifting: what the model gets to read per search (~3k tokens at 4 chars/token)
    passage_chars: int = field(default_factory=lambda: _env("PASSAGE_CHARS", 600, int))
    max_passages: int = field(default_factory=lambda: _env("MAX_PASSAGES", 8, int))
    max_passages_per_source: int = field(default_factory=lambda: _env("MAX_PER_SOURCE", 2, int))

    # Generation
    think_for_tools: bool = field(default_factory=lambda: _env("THINK_FOR_TOOLS", "1") == "1")
    think_for_answer: bool = field(default_factory=lambda: _env("THINK_FOR_ANSWER", "0") == "1")
    max_tokens: int = field(default_factory=lambda: _env("MAX_TOKENS", 2048, int))
    history_turns: int = field(default_factory=lambda: _env("HISTORY_TURNS", 6, int))  # user+assistant pairs kept
