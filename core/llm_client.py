"""
Intentra LLM Client - Multi-provider with automatic fallback priority.

Provider priority (first available key wins):
  1. Groq        (GROQ_API_KEY)           - fastest, free tier
  2. OpenRouter  (OPENROUTER_API_KEY)     - wide model selection
  3. Anthropic   (ANTHROPIC_API_KEY)      - claude models
  4. Local/Ngrok (LOCAL_API_KEY + LOCAL_BASE_URL) - offline / lm-studio

Set INTENTRA_PROVIDER=groq|openrouter|anthropic|local to force a specific one.
"""

import os
import json
from openai import OpenAI
from anthropic import Anthropic

# ── Provider default models ──────────────────────────────────────────────────
PROVIDER_MODELS = {
    "groq":       os.environ.get("GROQ_MODEL",       "llama-3.1-8b-instant"),
    "openrouter": os.environ.get("OPENROUTER_MODEL",  "mistralai/mistral-7b-instruct"),
    "anthropic":  os.environ.get("ANTHROPIC_MODEL",   "claude-3-haiku-20240307"),
    "local":      os.environ.get("LOCAL_MODEL_NAME",  "local-model"),
}

# ── Provider base URLs ────────────────────────────────────────────────────────
PROVIDER_BASES = {
    "groq":       "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _detect_provider() -> str | None:
    """
    Return the first available provider, respecting INTENTRA_PROVIDER override.
    Returns None if no provider is configured.
    """
    forced = os.environ.get("INTENTRA_PROVIDER", "").lower().strip()
    if forced in ("groq", "openrouter", "anthropic", "local"):
        return forced

    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("LOCAL_API_KEY") and os.environ.get("LOCAL_BASE_URL"):
        return "local"
    return None


def call_llm(prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """
    Call the best available LLM provider and return the text response.
    Raises RuntimeError if no provider is configured.
    """
    provider = _detect_provider()

    if provider is None:
        raise RuntimeError(
            "No LLM provider configured. Set one of: "
            "GROQ_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY, "
            "or LOCAL_API_KEY + LOCAL_BASE_URL in your .env file."
        )

    print(f"[Intentra] Using provider: {provider}")

    # ── Groq ────────────────────────────────────────────────────────────────
    if provider == "groq":
        client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=PROVIDER_BASES["groq"],
        )
        resp = client.chat.completions.create(
            model=PROVIDER_MODELS["groq"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    # ── OpenRouter ───────────────────────────────────────────────────────────
    if provider == "openrouter":
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=PROVIDER_BASES["openrouter"],
            default_headers={
                "HTTP-Referer": "https://intentra-jvd1.onrender.com",
                "X-Title": "Intentra",
            },
        )
        resp = client.chat.completions.create(
            model=PROVIDER_MODELS["openrouter"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    # ── Anthropic ─────────────────────────────────────────────────────────────
    if provider == "anthropic":
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=PROVIDER_MODELS["anthropic"],
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    # ── Local / Ngrok / LM-Studio ─────────────────────────────────────────────
    if provider == "local":
        client = OpenAI(
            api_key=os.environ.get("LOCAL_API_KEY", "lm-studio"),
            base_url=os.environ["LOCAL_BASE_URL"],
            timeout=7200.0,
        )
        resp = client.chat.completions.create(
            model=PROVIDER_MODELS["local"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    raise RuntimeError(f"Unknown provider: {provider}")


def extract_json(content: str):
    """Strip markdown fences and parse JSON from LLM output."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    return json.loads(content.strip())
