"""Thin LLM provider interface: Gemini, Mistral or the Claude Code CLI, with a hard timeout.

LLM_PROVIDER = auto | gemini | mistral | claude_cli | none
auto picks Gemini when GEMINI_API_KEY is set, then Mistral when MISTRAL_API_KEY is set,
otherwise none (the rules path handles everything). Model names always come from env vars.
Cost = (input tokens x PRICE_IN_PER_M + output tokens x PRICE_OUT_PER_M) / 1e6.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Protocol

_POOL = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")


@dataclass
class LLMResult:
    data: dict
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        pin = float(os.environ.get("PRICE_IN_PER_M", "0") or 0)
        pout = float(os.environ.get("PRICE_OUT_PER_M", "0") or 0)
        return round((self.input_tokens * pin + self.output_tokens * pout) / 1e6, 6)


class LLMError(RuntimeError):
    pass


class Provider(Protocol):
    name: str
    model: str

    def generate_json(self, system: str, prompt: str, schema: dict) -> LLMResult: ...


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str) -> None:
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate_json(self, system: str, prompt: str, schema: dict) -> LLMResult:
        from google.genai import types

        t0 = time.perf_counter()
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
        usage = getattr(resp, "usage_metadata", None)
        return LLMResult(
            data=json.loads(resp.text),
            model=f"gemini/{self.model}",
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class MistralProvider:
    name = "mistral"

    def __init__(self, model: str) -> None:
        self.model = model
        self.key = os.environ["MISTRAL_API_KEY"]

    def generate_json(self, system: str, prompt: str, schema: dict) -> LLMResult:
        import httpx

        t0 = time.perf_counter()
        r = httpx.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system + "\nReturn JSON matching: " + json.dumps(schema)},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=30,
        )
        if r.status_code >= 500:
            raise LLMError(f"mistral {r.status_code}")
        r.raise_for_status()
        body = r.json()
        usage = body.get("usage", {})
        return LLMResult(
            data=json.loads(body["choices"][0]["message"]["content"]),
            model=f"mistral/{self.model}",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class ClaudeCLIProvider:
    """Uses the local `claude -p` CLI with structured output. For offline compile and testing."""

    name = "claude_cli"

    def __init__(self, model: str) -> None:
        if not shutil.which("claude"):
            raise LLMError("claude CLI not found")
        self.model = model

    def generate_json(self, system: str, prompt: str, schema: dict) -> LLMResult:
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(schema),
             "--model", self.model, "--tools", "", "--no-session-persistence",
             "--system-prompt", system],
            capture_output=True, text=True, timeout=180,
        )
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMError(f"claude cli: {proc.stderr[:200] or proc.stdout[:200]}") from exc
        if out.get("is_error"):
            raise LLMError(f"claude cli: {out.get('result')}")
        data = out.get("structured_output")
        if data is None:
            data = json.loads(out.get("result", "{}"))
        usage = out.get("usage", {})
        return LLMResult(
            data=data,
            model=f"claude_cli/{self.model}",
            input_tokens=usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def make_provider(purpose: str = "runtime") -> Optional[Provider]:
    choice = os.environ.get("LLM_PROVIDER", "auto").lower()
    env_model = os.environ.get("GEMINI_MODEL_COMPILE" if purpose == "compile" else "GEMINI_MODEL_RUNTIME")
    try:
        if choice in ("gemini",) or (choice == "auto" and os.environ.get("GEMINI_API_KEY")):
            if not env_model:
                raise LLMError("set GEMINI_MODEL_RUNTIME / GEMINI_MODEL_COMPILE")
            return GeminiProvider(env_model)
        if choice == "mistral" or (choice == "auto" and os.environ.get("MISTRAL_API_KEY")):
            return MistralProvider(os.environ.get("MISTRAL_MODEL", "mistral-small-latest"))
        if choice == "claude_cli":
            return ClaudeCLIProvider(os.environ.get("CLAUDE_CLI_MODEL", "haiku"))
    except Exception:
        return None
    return None


def call_with_timeout(provider: Provider, system: str, prompt: str, schema: dict,
                      timeout_s: float, retries: int = 1) -> LLMResult:
    """Hard timeout; one retry on server-side errors only."""
    last: Exception | None = None
    deadline = time.monotonic() + timeout_s
    for _ in range(retries + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0.2:
            break
        fut = _POOL.submit(provider.generate_json, system, prompt, schema)
        try:
            return fut.result(timeout=remaining)
        except cf.TimeoutError as exc:
            raise LLMError(f"timeout after {timeout_s}s") from exc
        except LLMError as exc:
            last = exc
            if "5" not in str(exc)[:12]:
                break
        except Exception as exc:  # network, JSON, SDK errors
            last = exc
            break
    raise LLMError(str(last) if last else "LLM call failed")
