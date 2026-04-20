"""Groq LLM client (OpenAI-compatible SDK) with rate-limit backoff."""

from __future__ import annotations

import time

import openai

from src.prompts import SYSTEM_PROMPT, build_messages

PRIMARY_MODEL = "llama-3.3-70b-versatile"
COMPARISON_MODEL = "llama-3.1-8b-instant"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_TOKENS = 512

_MAX_RETRIES = 4
_INITIAL_BACKOFF_S = 2.0  # doubles each retry; Groq free tier: 30 RPM / 6K TPM


def make_client(api_key: str) -> openai.OpenAI:
    """Return an OpenAI client pointed at Groq's API."""
    return openai.OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _call_with_backoff(
    client: openai.OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> openai.types.chat.ChatCompletion:
    """Call chat.completions.create with exponential backoff on 429."""
    delay = _INITIAL_BACKOFF_S
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
        except openai.RateLimitError:
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2


def get_interpretation(
    features: dict,
    api_key: str,
    model: str = PRIMARY_MODEL,
    client: openai.OpenAI | None = None,
) -> tuple[str, dict]:
    """Generate a plain-language PPG interpretation via Groq.

    Parameters
    ----------
    features : output of src.features.extract_features()
    api_key  : Groq API key (ignored when client is supplied)
    model    : Groq model ID; defaults to PRIMARY_MODEL
    client   : pre-built OpenAI client (pass a cached instance from app.py)

    Returns
    -------
    text  : interpretation string
    usage : dict with input_tokens, output_tokens
    """
    if client is None:
        client = make_client(api_key)

    user_messages = build_messages(features)
    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *user_messages,
    ]

    response = _call_with_backoff(client, model, full_messages, MAX_TOKENS)

    text = response.choices[0].message.content
    usage_obj = response.usage
    usage = {
        "input_tokens": usage_obj.prompt_tokens if usage_obj else 0,
        "output_tokens": usage_obj.completion_tokens if usage_obj else 0,
    }
    return text, usage
