from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

MODEL = "gemini-3.8-flash"


class ProviderError(RuntimeError):
    pass


class OutputError(RuntimeError):
    """A finished call whose content is not a complete valid structured response."""


def safe_error(exc: Exception) -> str:
    message = str(getattr(exc, "message", ""))
    key = os.environ.get("GOOGLE_CLOUD_API_KEY", "")
    if key:
        message = message.replace(key, "[REDACTED]")
    message = re.sub(r"(?i)(key|api_key|access_token)=([^\s&]+)", r"\1=[REDACTED]", message)
    return message[:1200]


def generation_schema(schema: type[BaseModel]) -> dict:
    """Keep wire schema compact; enforce size/string bounds with Pydantic locally.

    Deeply nested bounded arrays expand the structured-output grammar and can
    produce an unhelpful Vertex INVALID_ARGUMENT. Do not weaken local validation.
    """
    def compact(value, names=False):
        if isinstance(value, list):
            return [compact(v) for v in value]
        if isinstance(value, dict):
            return {k: compact(v, names=k in {"properties", "$defs"}) for k, v in value.items() if names or k not in {"title", "minLength", "maxLength", "minItems", "maxItems"}}
        return value
    return compact(schema.model_json_schema())


class VertexProvider:
    model = MODEL
    reasoning = {"thinking_level": "HIGH"}
    author_provenance = {"model": MODEL, **reasoning}
    review_provenance = {"model": MODEL, **reasoning}

    def __init__(self, audit_dir: Path, max_calls: int = 100, timeout_seconds: int = 900):
        key = os.environ.get("GOOGLE_CLOUD_API_KEY")
        if not key:
            raise ProviderError("Set GOOGLE_CLOUD_API_KEY to a Vertex AI / Agent Platform API key")
        self.client = genai.Client(vertexai=True, api_key=key, http_options=types.HttpOptions(
            api_version="v1", timeout=timeout_seconds * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ))
        self.audit_dir = audit_dir
        audit_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        self.calls = 0
        self.usage: list[dict] = []

    def close(self):
        self.client.close()

    def generate(self, system: str, payload: dict, schema: type[BaseModel], label: str, max_tokens: int = 32768):
        prompt = json.dumps(payload, ensure_ascii=False)
        config = types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
            response_mime_type="application/json", response_json_schema=generation_schema(schema),
            max_output_tokens=max_tokens,
        )
        # Persist exact prompt/schema, but never client config, environment or credentials.
        (self.audit_dir / f"{label}.request.json").write_text(json.dumps({"model": MODEL, "thinking_level": "HIGH", "system": system, "payload": payload, "schema": generation_schema(schema)}, ensure_ascii=False, indent=2))
        for attempt in range(3):
            if self.calls >= self.max_calls:
                raise ProviderError(f"API call limit reached ({self.max_calls}); increase --max-api-calls to continue")
            self.calls += 1
            started = time.monotonic()
            try:
                result = self.client.models.generate_content(model=MODEL, contents=prompt, config=config)
            except (errors.APIError, httpx.TransportError) as exc:
                code = getattr(exc, "code", None)
                retryable = code in (408, 429, 500, 502, 503, 504) or isinstance(exc, httpx.TransportError)
                detail = safe_error(exc)
                entry = {"label": label, "attempt": attempt + 1, "error_type": type(exc).__name__, "status_code": code, "detail": detail, "seconds": round(time.monotonic()-started, 2)}
                self.usage.append(entry)
                self._save_usage()
                if retryable and attempt < 2:
                    time.sleep(min(20, 2 ** (attempt+1) + random.random()))
                    continue
                # Exception strings can include authenticated request URLs; never propagate them.
                raise ProviderError(f"Vertex request failed: {type(exc).__name__}, HTTP {code or 'network'}. {detail}") from None
            candidates = result.candidates or []
            reason = str(candidates[0].finish_reason.value) if candidates and candidates[0].finish_reason else "NO_CANDIDATE"
            metadata = {"label": label, "attempt": attempt + 1, "model_version": result.model_version, "finish_reason": reason,
                        "seconds": round(time.monotonic()-started, 2), "usage": result.usage_metadata.model_dump(mode="json") if result.usage_metadata else {}}
            self.usage.append(metadata)
            self._save_usage()
            # Extract visible text only; thought summaries are neither requested nor saved.
            raw = "".join(p.text or "" for c in candidates[:1] for p in (c.content.parts if c.content else []) if not p.thought)
            (self.audit_dir / f"{label}.response.txt").write_text(raw)
            if reason != "STOP" or not raw:
                raise OutputError(f"Incomplete or blocked response: finish_reason={reason}")
            try:
                parsed = schema.model_validate_json(raw)
            except ValidationError as exc:
                raise OutputError(str(exc)) from None
            metadata["content_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            self._save_usage()
            return parsed
        raise ProviderError("Request attempts exhausted")

    def _save_usage(self):
        (self.audit_dir / "usage.json").write_text(json.dumps(self.usage, ensure_ascii=False, indent=2))
