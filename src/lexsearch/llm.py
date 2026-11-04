"""Chat models through OpenRouter, with a hard spending cap read from each response's reported cost."""
from __future__ import annotations

import json
import os
import re
import threading
import time

OPENROUTER = "https://openrouter.ai/api/v1"
MODELS = {
    "qwen": "qwen/qwen3-30b-a3b-instruct-2507",  # open weights, 3B active parameters: runs on one GPU
    "haiku": "anthropic/claude-haiku-5.5",
    "sonnet": "anthropic/claude-sonnet-5.5",
}


class BudgetExceeded(RuntimeError):
    pass


def parse_json(text: str) -> dict:
    """The first JSON object in a reply, tolerating Markdown fences and text around it."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the answer")
    try:
        return json.JSONDecoder().raw_decode(text, start)[0]
    except json.JSONDecodeError:
        # The ids come first: a broken answer string after them (an unescaped quote) shouldn't lose them.
        ids = re.search(r'"article_ids"\s*:\s*(\[[^\]]*\])', text)
        if ids is None:
            raise
        return {"article_ids": json.loads(ids.group(1)), "answer": ""}


def _with_retries(call, tries: int = 6):
    """Rate limits and transient provider errors: back off and retry, up to about two minutes."""
    import openai

    for i in range(tries):
        try:
            return call()
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError, openai.APITimeoutError):
            if i == tries - 1:
                raise
            time.sleep(min(60, 2 * 2 ** i))


class Chat:
    def __init__(self, model: str, max_cost: float | None, client=None) -> None:
        from openai import OpenAI

        if max_cost is None and client is None:
            raise RuntimeError("Paid run without --max-cost: refused.")
        self.model = MODELS.get(model, model)
        self.client = client or OpenAI(base_url=OPENROUTER, api_key=os.getenv("OPENROUTER_API_KEY") or "missing",
                                       max_retries=2, timeout=180)
        self.max_cost, self.spent = max_cost, 0.0
        self._lock = threading.Lock()

    def __call__(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 4000):
        """One completion. Returns (message, cost, tokens in, tokens out)."""
        with self._lock:
            if self.max_cost is not None and self.spent >= self.max_cost:
                raise BudgetExceeded(f"${self.max_cost} reached")
        kw = {"tools": tools} if tools else {}
        resp = _with_retries(lambda: self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0, max_tokens=max_tokens, **kw))
        usage = resp.usage
        cost = float(((usage.model_extra or {}) if usage else {}).get("cost") or 0.0)
        with self._lock:
            self.spent += cost
        return (resp.choices[0].message, cost, usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0)
