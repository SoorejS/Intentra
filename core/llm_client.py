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
import re
import traceback

# Known placeholder values that should be treated as "not set"
_PLACEHOLDERS = {
    "", "your_groq_api_key_here", "your_openrouter_api_key_here",
    "your_actual_api_key_here", "your_api_key_here", "sk-xxx",
    "your_anthropic_api_key_here",
}


def _is_real_key(value: str | None) -> bool:
    """Return True only if value is a non-empty, non-placeholder string."""
    if not value:
        return False
    return value.strip().lower() not in _PLACEHOLDERS


def _get_provider_models() -> dict:
    """Read model names fresh from env (not cached at import time)."""
    return {
        "groq":       os.environ.get("GROQ_MODEL",       "llama-3.1-8b-instant"),
        "openrouter": os.environ.get("OPENROUTER_MODEL",  "mistralai/mistral-7b-instruct"),
        "anthropic":  os.environ.get("ANTHROPIC_MODEL",   "claude-3-haiku-20240307"),
        "local":      os.environ.get("LOCAL_MODEL_NAME",  "local-model"),
    }


# Keep a module-level reference for the /api/provider endpoint
PROVIDER_MODELS = _get_provider_models()

# ── Provider base URLs ────────────────────────────────────────────────────────
PROVIDER_BASES = {
    "groq":       "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _detect_provider() -> str | None:
    """
    Return the first available provider, respecting INTENTRA_PROVIDER override.
    Returns None if no provider is configured with a real API key.
    """
    forced = os.environ.get("INTENTRA_PROVIDER", "").lower().strip()
    if forced in ("groq", "openrouter", "anthropic", "local"):
        return forced

    if _is_real_key(os.environ.get("GROQ_API_KEY")):
        return "groq"
    if _is_real_key(os.environ.get("OPENROUTER_API_KEY")):
        return "openrouter"
    if _is_real_key(os.environ.get("ANTHROPIC_API_KEY")):
        return "anthropic"
    if _is_real_key(os.environ.get("LOCAL_API_KEY")) and os.environ.get("LOCAL_BASE_URL"):
        return "local"
    return None


def call_llm(prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """
    Call the best available LLM provider and return the text response.
    Raises RuntimeError if no provider is configured or if the API call fails.
    """
    provider = _detect_provider()
    models = _get_provider_models()

    if provider is None:
        raise RuntimeError(
            "No LLM provider configured. Set one of: "
            "GROQ_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY, "
            "or LOCAL_API_KEY + LOCAL_BASE_URL in your .env file."
        )

    print(f"[Intentra LLM] Using provider: {provider}, model: {models.get(provider)}")

    try:
        # ── Groq ────────────────────────────────────────────────────────────
        if provider == "groq":
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ["GROQ_API_KEY"],
                base_url=PROVIDER_BASES["groq"],
            )
            groq_models_to_try = [
                models["groq"],
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768"
            ]
            last_err = None
            for m in groq_models_to_try:
                try:
                    resp = client.chat.completions.create(
                        model=m,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content
                except Exception as ex:
                    last_err = ex
                    if "model_not_found" in str(ex) or "404" in str(ex):
                        continue
                    raise ex
            raise last_err

        # ── OpenRouter ──────────────────────────────────────────────────────
        if provider == "openrouter":
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url=PROVIDER_BASES["openrouter"],
                default_headers={
                    "HTTP-Referer": "https://intentra-jvd1.onrender.com",
                    "X-Title": "Intentra",
                },
            )
            resp = client.chat.completions.create(
                model=models["openrouter"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

        # ── Anthropic ───────────────────────────────────────────────────────
        if provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            resp = client.messages.create(
                model=models["anthropic"],
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        # ── Local / Ngrok / LM-Studio ───────────────────────────────────────
        if provider == "local":
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("LOCAL_API_KEY", "lm-studio"),
                base_url=os.environ["LOCAL_BASE_URL"],
                timeout=120.0,
            )
            resp = client.chat.completions.create(
                model=models["local"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

        raise RuntimeError(f"Unknown provider: {provider}")

    except Exception as e:
        print(f"[Intentra LLM] ERROR calling {provider}: {e}")
        traceback.print_exc()
        raise RuntimeError(f"LLM call failed ({provider}): {e}") from e


def extract_json(content: str):
    """Strip markdown fences and parse JSON from LLM output with regex fallback."""
    if not content or not content.strip():
        raise ValueError("Empty LLM response — cannot extract JSON")

    # 1. Backtick codeblocks first
    if "```json" in content:
        block = content.split("```json")[1].split("```")[0].strip()
        try:
            return json.loads(block)
        except Exception:
            pass
    elif "```" in content:
        block = content.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(block)
        except Exception:
            pass

    # 2. Try direct JSON load
    try:
        return json.loads(content.strip())
    except Exception:
        pass

    # 3. Use regex / substring search to extract array [...] or object {...}
    arr_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", content)
    if arr_match:
        try:
            return json.loads(arr_match.group(0).strip())
        except Exception:
            pass

    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(content[start_idx:end_idx+1].strip())
        except Exception:
            pass

    return json.loads(content.strip())
