"""OpenAI Responses transport for independent TOEIC author/reviewer calls.

Only visible output text and response metadata are retained. Every call starts a
fresh conversation; authentication headers and reasoning items are never saved.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time

import httpx
from pydantic import BaseModel, ValidationError

from .provider import OutputError, ProviderError, generation_schema

OPENAI_MODEL = "gpt-6.1-sol"
EFFORT = "xhigh"
KEY_NAMES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_CLOUD_API_KEY")


def _redact_text(message: str) -> str:
    for name in KEY_NAMES:
        key = os.environ.get(name)
        if key:
            message = message.replace(key, "[REDACTED]")
    return message


def redact(message: str) -> str:
    return _redact_text(message)[:1200]


ERROR_FIELDS = ("type", "code", "message", "param", "status", "status_code")
TRANSIENT_API_ERRORS = {"server_error", "rate_limit_exceeded"}
NONRETRYABLE_API_ERRORS = {
    "insufficient_quota", "invalid_api_key", "authentication_error",
    "permission_error", "permission_denied", "unauthorized",
    "account_deactivated", "billing_hard_limit_reached", "billing_not_active",
    "invalid_request_error",
}


def _error_diagnostic(event: dict) -> dict:
    """Keep useful public error fields, including nested SSE error envelopes."""
    def fields(value, depth=0):
        result = {}
        for name in ERROR_FIELDS:
            if name not in value:
                continue
            item = value[name]
            if item is None or isinstance(item, (str, int, float, bool)):
                result[name] = redact(item) if isinstance(item, str) else item
        nested = value.get("error")
        if isinstance(nested, dict) and depth < 4:
            result["error"] = fields(nested, depth + 1)
        elif isinstance(nested, str):
            result["error"] = redact(nested)
        return result
    return fields(event)


def _error_sources(detail: dict) -> list[dict]:
    sources = [detail]
    while isinstance(sources[-1].get("error"), dict):
        sources.append(sources[-1]["error"])
    return sources


def _error_identifiers(detail: dict) -> set[str]:
    return {v for source in _error_sources(detail) for k in ("code", "type")
            if isinstance(v := source.get(k), str)}


def _transient_stream_error(detail: dict) -> bool:
    if _error_identifiers(detail) & NONRETRYABLE_API_ERRORS:
        return False
    # A specific inner code takes precedence over a broader outer error type.
    for source in reversed(_error_sources(detail)):
        for name in ("code", "type"):
            value = source.get(name)
            if isinstance(value, str) and value and value != "error":
                return value in TRANSIENT_API_ERRORS
    return False


def openai_schema(schema: type[BaseModel]) -> dict:
    """Adapt the wire grammar without weakening the original Pydantic model.

    Strict structured outputs require all object properties to be required,
    including fields with defaults. Nullable types retain their explicit null
    branch. Property and definition *names* must not be mistaken for keywords.
    """
    def adapt(value, names=False):
        if isinstance(value, list):
            return [adapt(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {k: adapt(v, k in {"properties", "$defs"})
                  for k, v in value.items()
                  if names or k not in {"default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}}
        if not names and result.get("type") == "object":
            result["additionalProperties"] = False
            result["required"] = list(result.get("properties", {}))
        return result
    return adapt(generation_schema(schema))


def _visible_output(response: dict) -> tuple[str, bool]:
    """Read text/refusal content, never serialize a response's reasoning items."""
    texts = []
    refused = False
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
            elif content.get("type") == "refusal":
                refused = True
    return "".join(texts), refused


class OpenAIProvider:
    model = OPENAI_MODEL
    reasoning = {"effort": EFFORT}
    author_provenance = {"model": model, **reasoning}
    review_provenance = {"model": model, **reasoning}

    def __init__(self, audit_dir: Path, max_calls=240, timeout_seconds=900, *, client=None):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError("Set OPENAI_API_KEY for OpenAI generation or review")
        self.audit_dir = audit_dir
        audit_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls, self.calls, self.usage = max_calls, 0, []
        self.client = client or httpx.Client(
            base_url="https://api.openai.com", timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )

    def close(self):
        self.client.close()

    def _write_json(self, path: Path, value):
        # Also protect unexpected credential echoes in server diagnostics.
        path.write_text(_redact_text(json.dumps(value, ensure_ascii=False, indent=2)))

    def _save_usage(self):
        self._write_json(self.audit_dir / "usage.json", self.usage)

    def generate(self, system, payload, schema, label, max_tokens=65536):
        wire_schema = openai_schema(schema)
        schema_name = re.sub(r"[^a-zA-Z0-9_-]", "_", schema.__name__)[:64]
        body = {
            "model": OPENAI_MODEL,
            "instructions": system,
            "input": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "reasoning": {"effort": EFFORT},
            "text": {"format": {"type": "json_schema", "name": schema_name,
                                  "strict": True, "schema": wire_schema}},
            "max_output_tokens": max(65536, max_tokens),
            "stream": True,
            "store": False,
        }
        self._write_json(self.audit_dir / f"{label}.request.json", {
            "provider": "openai", "model": OPENAI_MODEL, "effort": EFFORT,
            "system": system, "payload": payload, "schema": wire_schema,
            "wire_request": body,
        })
        for attempt in range(1, 4):
            if self.calls >= self.max_calls:
                raise ProviderError(f"API call limit reached ({self.max_calls})")
            self.calls += 1
            started = time.monotonic()
            entry = {"label": label, "attempt": attempt, "model": OPENAI_MODEL, "effort": EFFORT}
            raw, streamed, model, status, usage = "", "", None, None, {}
            completed = refused = False
            nonretryable_api_error = False
            try:
                with self.client.stream("POST", "/v1/responses", json=body) as response:
                    entry["request_id"] = response.headers.get("x-request-id")
                    if response.status_code >= 400:
                        error_text = response.read().decode(errors="replace")
                        entry.update(status_code=response.status_code, detail=redact(error_text))
                        try:
                            error_body = json.loads(error_text)
                        except ValueError:
                            error_body = None
                        if isinstance(error_body, dict):
                            entry["api_error"] = _error_diagnostic(error_body)
                            nonretryable_api_error = bool(
                                _error_identifiers(entry["api_error"]) & NONRETRYABLE_API_ERRORS)
                        response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        value = line[5:].strip()
                        if not value or value == "[DONE]":
                            continue
                        event = json.loads(value)
                        kind = event.get("type")
                        if kind in ("response.created", "response.in_progress"):
                            state = event.get("response") or {}
                            model = state.get("model", model)
                            entry["response_id"] = state.get("id", entry.get("response_id"))
                        elif kind == "response.output_text.delta":
                            streamed += event.get("delta", "")
                        elif kind in ("response.refusal.delta", "response.refusal.done"):
                            refused = True
                        elif kind in ("response.completed", "response.incomplete", "response.failed"):
                            state = event.get("response") or {}
                            model = state.get("model", model)
                            entry["response_id"] = state.get("id", entry.get("response_id"))
                            status = state.get("status")
                            usage = state.get("usage") or {}
                            raw, final_refusal = _visible_output(state)
                            refused = refused or final_refusal
                            completed = kind == "response.completed" and status == "completed"
                            if state.get("incomplete_details"):
                                entry["incomplete_details"] = state["incomplete_details"]
                            if state.get("error"):
                                entry["detail"] = redact(json.dumps(state["error"]))
                        elif kind == "error":
                            entry["api_error"] = _error_diagnostic(event)
                            entry["detail"] = redact(json.dumps(entry["api_error"], ensure_ascii=False))
                            if _transient_stream_error(entry["api_error"]):
                                raise httpx.ReadError("Temporary OpenAI stream error")
                            raise ValueError("OpenAI stream error")
            except (httpx.HTTPError, ValueError) as exc:
                entry.update(error_type=type(exc).__name__, seconds=round(time.monotonic() - started, 2),
                             usage=usage, billing_complete=False)
                self.usage.append(entry)
                self._save_usage()
                if streamed:
                    (self.audit_dir / f"{label}.attempt{attempt}.partial.txt").write_text(_redact_text(streamed))
                code = entry.get("status_code")
                retryable = not nonretryable_api_error and (
                    code in (408, 429, 500, 502, 503, 504, 529) or isinstance(exc, httpx.TransportError))
                if retryable and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise ProviderError(f"OpenAI request failed: {type(exc).__name__}, HTTP {code or 'stream/network'}. {entry.get('detail', '')}") from None

            # A terminal response provides canonical output text. Deltas are only
            # a diagnostic fallback when the stream ended without that response.
            visible = raw if raw else streamed
            entry.update(model_version=model, finish_reason=status, usage=usage,
                         seconds=round(time.monotonic() - started, 2), stream_complete=completed,
                         refused=refused, content_sha256=hashlib.sha256(visible.encode()).hexdigest())
            self.usage.append(entry)
            self._save_usage()
            (self.audit_dir / f"{label}.response.txt").write_text(_redact_text(visible))
            if model != OPENAI_MODEL:
                raise ProviderError(f"Unexpected returned model: {redact(str(model))}; no fallback accepted")
            if refused:
                raise OutputError("OpenAI response contains a refusal")
            if not completed or status != "completed" or not raw:
                raise OutputError(f"Incomplete OpenAI response: status={status}, stream_complete={completed}")
            if streamed and streamed != raw:
                raise OutputError("OpenAI final text does not match streamed output")
            try:
                return schema.model_validate_json(raw)
            except ValidationError as exc:
                raise OutputError(redact(str(exc))) from None
