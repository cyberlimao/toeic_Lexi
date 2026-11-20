"""Anthropic Messages transport for authors and reviewers.

Each call is a fresh conversation. Only final text and usage are retained;
thinking blocks, authentication headers and credentials are never persisted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import httpx
from pydantic import BaseModel, ValidationError

from .provider import MODEL, OutputError, ProviderError, generation_schema

ANTHROPIC_MODEL = "claude-opus-5-5"
EFFORT = "xhigh"


def anthropic_schema(schema: type[BaseModel]) -> dict:
    def adapt(value, names=False):
        if isinstance(value, list):
            return [adapt(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {k: adapt(v, k in {"properties", "$defs"}) for k, v in value.items()
                  if names or k not in {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}}
        if not names and result.get("type") == "object":
            result["additionalProperties"] = False
        return result
    return adapt(generation_schema(schema))


def redact(message: str) -> str:
    for name in ("ANTHROPIC_API_KEY", "GOOGLE_CLOUD_API_KEY"):
        if os.environ.get(name):
            message = message.replace(os.environ[name], "[REDACTED]")
    return message[:1200]


class AnthropicProvider:
    model = ANTHROPIC_MODEL
    reasoning = {"effort": EFFORT, "thinking": "adaptive"}
    author_provenance = {"model": model, **reasoning}
    review_provenance = {"model": model, **reasoning}

    def __init__(self, audit_dir: Path, max_calls=240, timeout_seconds=900, *, client=None):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("Set ANTHROPIC_API_KEY for Claude generation or review")
        self.audit_dir = audit_dir
        audit_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls, self.calls, self.usage = max_calls, 0, []
        self.client = client or httpx.Client(
            base_url="https://api.anthropic.com", timeout=timeout_seconds,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
        )

    def close(self):
        self.client.close()

    def _save_usage(self):
        (self.audit_dir / "usage.json").write_text(json.dumps(self.usage, ensure_ascii=False, indent=2))

    def generate(self, system, payload, schema, label, max_tokens=65536):
        wire_schema = anthropic_schema(schema)
        # Thinking shares the output limit; use a fixed generous ceiling for
        # xhigh instead of silently inheriting small Gemini reviewer limits.
        body = {"model": ANTHROPIC_MODEL, "max_tokens": max(65536, max_tokens),
                "thinking": {"type": "adaptive"}, "stream": True,
                "system": system,
                "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": wire_schema}}}
        (self.audit_dir / f"{label}.request.json").write_text(json.dumps({
            "provider": "anthropic", "model": ANTHROPIC_MODEL, "effort": EFFORT,
            "system": system, "payload": payload, "schema": wire_schema, "wire_request": body,
        }, ensure_ascii=False, indent=2))
        for attempt in range(1, 4):
            if self.calls >= self.max_calls:
                raise ProviderError(f"API call limit reached ({self.max_calls})")
            self.calls += 1
            started = time.monotonic()
            entry = {"label": label, "attempt": attempt, "model": ANTHROPIC_MODEL, "effort": EFFORT}
            raw, stop, usage, model, completed = "", None, {}, None, False
            try:
                with self.client.stream("POST", "/v1/messages", json=body) as response:
                    entry["request_id"] = response.headers.get("request-id")
                    if response.status_code >= 400:
                        detail = redact(response.read().decode(errors="replace"))
                        entry.update(status_code=response.status_code, detail=detail)
                        response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        kind = event.get("type")
                        if kind == "message_start":
                            message = event["message"]
                            model = message.get("model")
                            entry["message_id"] = message.get("id")
                            usage.update(message.get("usage") or {})
                        elif kind == "content_block_start" and event["content_block"].get("type") == "text":
                            raw += event["content_block"].get("text", "")
                        elif kind == "content_block_delta" and event["delta"].get("type") == "text_delta":
                            raw += event["delta"]["text"]
                        elif kind == "message_delta":
                            stop = event["delta"].get("stop_reason", stop)
                            usage.update(event.get("usage") or {})
                        elif kind == "message_stop":
                            completed = True
                        elif kind == "error":
                            entry["detail"] = redact(json.dumps(event.get("error", {})))
                            raise httpx.ReadError("Anthropic stream error")
            except (httpx.HTTPError, ValueError) as exc:
                entry.update(error_type=type(exc).__name__, seconds=round(time.monotonic()-started, 2), usage=usage,
                             billing_complete=False)
                self.usage.append(entry); self._save_usage()
                if raw:
                    (self.audit_dir / f"{label}.attempt{attempt}.partial.txt").write_text(raw)
                status = entry.get("status_code")
                retryable = status in (408, 429, 500, 502, 503, 504, 529) or isinstance(exc, httpx.TransportError)
                if retryable and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise ProviderError(f"Anthropic request failed: {type(exc).__name__}, HTTP {status or 'stream/network'}. {entry.get('detail', '')}") from None
            entry.update(model_version=model, finish_reason=stop, usage=usage,
                         seconds=round(time.monotonic()-started, 2), stream_complete=completed,
                         content_sha256=hashlib.sha256(raw.encode()).hexdigest())
            self.usage.append(entry); self._save_usage()
            (self.audit_dir / f"{label}.response.txt").write_text(raw)
            if model != ANTHROPIC_MODEL:
                raise ProviderError(f"Unexpected returned model: {model}; no fallback accepted")
            if not completed or stop != "end_turn" or not raw:
                raise OutputError(f"Incomplete Anthropic response: stop_reason={stop}, stream_complete={completed}")
            try:
                return schema.model_validate_json(raw)
            except ValidationError as exc:
                raise OutputError(str(exc)) from None


class ReviewerRouter:
    """Route author/review roles to selected transports under one call cap."""
    def __init__(self, author, reviewer, author_system: str, max_calls=240):
        self.author, self.reviewer, self.author_system = author, reviewer, author_system
        self.max_calls = max_calls
        self.author_provenance = self._provenance(author)
        self.review_provenance = self._provenance(reviewer)
        self.model = self.author_provenance['model']
        self.reasoning = {k: v for k, v in self.author_provenance.items() if k != 'model'}

    @staticmethod
    def _provenance(provider):
        return {"model": getattr(provider, 'model', MODEL),
                **getattr(provider, 'reasoning', {"thinking_level": "HIGH"})}

    @property
    def calls(self):
        return self.author.calls + (self.reviewer.calls if self.reviewer is not self.author else 0)

    def generate(self, system, payload, schema, label, **kwargs):
        target = self.author if system == self.author_system else self.reviewer
        remaining = self.max_calls - self.calls
        if remaining <= 0:
            raise ProviderError(f"Combined API call limit reached ({self.max_calls})")
        target.max_calls = target.calls + remaining
        return target.generate(system, payload, schema, label, **kwargs)

    def close(self):
        try:
            self.author.close()
        finally:
            if self.reviewer is not self.author:
                self.reviewer.close()
