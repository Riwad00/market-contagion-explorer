from __future__ import annotations

import hashlib
import json
from typing import Any

import streamlit as st

from .llm_prompts import SYSTEM_PROMPT, build_user_prompt


class LLMError(RuntimeError):
    pass


def _extract_json_object(text: str) -> str:
    """
    Robust-ish extraction in case the model accidentally wraps output.
    Requirement: if invalid JSON, show friendly error + allow retry.
    """
    if not text:
        raise LLMError("Empty response from model.")
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise LLMError("Model did not return a JSON object.")
    return s[first : last + 1]


def _validate_brief_schema(obj: dict) -> None:
    required_top = {
        "pair",
        "time_context",
        "regime_summary",
        "what_changed",
        "lead_lag",
        "relationship_now",
        "reliability",
        "practical_implications",
        "caveats",
    }
    extra = set(obj.keys()) - required_top
    missing = required_top - set(obj.keys())
    if missing:
        raise LLMError(f"Missing keys in JSON: {sorted(missing)}")
    if extra:
        raise LLMError(f"Extra keys present (not allowed): {sorted(extra)}")

    wc = obj.get("what_changed", [])
    if not isinstance(wc, list) or not (2 <= len(wc) <= 4):
        raise LLMError('"what_changed" must be a list with 2 to 4 items.')

    pi = obj.get("practical_implications", [])
    if not isinstance(pi, list) or len(pi) != 3:
        raise LLMError('"practical_implications" must be a list with exactly 3 items.')

    cav = obj.get("caveats", [])
    if not isinstance(cav, list) or len(cav) != 3:
        raise LLMError('"caveats" must be a list with exactly 3 items.')


def parse_ai_brief(text: str) -> dict[str, Any]:
    try:
        raw = _extract_json_object(text)
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise LLMError("Top-level JSON must be an object.")
        _validate_brief_schema(obj)
        return obj
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON from model: {e.msg}") from e


def _get_session_cache() -> dict[str, Any]:
    if "llm_cache" not in st.session_state:
        st.session_state["llm_cache"] = {}
    return st.session_state["llm_cache"]


def generate_ai_brief(
    *,
    evidence_json: dict[str, Any],
    provider: str,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    if provider != "openai":
        raise LLMError("Only provider 'OpenAI' is implemented in this prototype.")
    if not model:
        raise LLMError("Model name is required.")
    if not api_key:
        raise LLMError("API key is required to use AI Brief.")

    user_prompt = build_user_prompt(evidence_json)
    system_prompt_runtime = SYSTEM_PROMPT.replace(
        "<EVIDENCE_JSON_HERE>",
        json.dumps(evidence_json, ensure_ascii=False, separators=(",", ":")),
    )

    # Stable cache key (do not use Python's built-in hash(), which is randomized per process).
    key_material = (provider + "|" + model + "|" + system_prompt_runtime + "|" + user_prompt).encode("utf-8")
    cache_key = hashlib.sha256(key_material).hexdigest()
    cache = _get_session_cache()
    if cache_key in cache:
        return cache[cache_key]

    try:
        from openai import OpenAI
    except Exception as e:
        raise LLMError("OpenAI SDK is not installed. Add 'openai' to requirements.txt.") from e

    try:
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt_runtime},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = getattr(resp, "output_text", None)
        if text is None:
            # Fallback: attempt to reconstruct from outputs
            text = ""
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        text += getattr(c, "text", "") or ""
        brief = parse_ai_brief(text)
        cache[cache_key] = brief
        return brief
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"LLM call failed: {type(e).__name__}: {e}") from e

