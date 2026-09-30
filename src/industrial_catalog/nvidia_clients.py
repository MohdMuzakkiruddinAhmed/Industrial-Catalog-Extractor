"""Small, dependency-light clients for NVIDIA NIM/OpenAI-compatible endpoints.

The production server can point each adapter at a separate local NIM.  Tests and
offline development can use ``dry_run=True`` or inject a ``JsonTransport``; no
network call is made in either case.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

JsonObject = dict[str, Any]


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hash(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value if isinstance(value, bytes) else _stable_json(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Deterministic exponential retry policy (no random jitter)."""

    max_attempts: int = 4
    initial_delay_seconds: float = 0.5
    multiplier: float = 2.0
    max_delay_seconds: float = 8.0
    retry_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 409, 425, 429, 500, 502, 503, 504})
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.initial_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")
        if self.multiplier < 1:
            raise ValueError("retry multiplier must be at least one")

    def delay_before_attempt(self, attempt_number: int) -> float:
        """Return delay before the one-indexed retry attempt.

        ``attempt_number=2`` is the first retry after the initial request.
        """

        if attempt_number <= 1:
            return 0.0
        exponent = attempt_number - 2
        return min(
            self.max_delay_seconds,
            self.initial_delay_seconds * (self.multiplier**exponent),
        )


@dataclass(frozen=True, slots=True)
class NIMEndpointConfig:
    """Connection information for one OpenAI-compatible NVIDIA endpoint."""

    base_url: str
    model: str
    endpoint_path: str = "/v1/chat/completions"
    api_key: str | None = None
    api_key_env: str = "NVIDIA_API_KEY"
    timeout_seconds: float = 120.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    dry_run: bool = False
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url and not self.dry_run:
            raise ValueError("base_url is required unless dry_run is enabled")
        if not self.model:
            raise ValueError("model is required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @property
    def url(self) -> str:
        if self.endpoint_path.startswith(("http://", "https://")):
            return self.endpoint_path
        return f"{self.base_url.rstrip('/')}/{self.endpoint_path.lstrip('/')}"

    @property
    def resolved_api_key(self) -> str | None:
        return self.api_key or os.getenv(self.api_key_env) or None

    def safe_dict(self) -> dict[str, Any]:
        """Serializable configuration with credentials intentionally omitted."""

        return {
            "base_url": self.base_url,
            "model": self.model,
            "endpoint_path": self.endpoint_path,
            "timeout_seconds": self.timeout_seconds,
            "retry": {
                **asdict(self.retry),
                "retry_status_codes": sorted(self.retry.retry_status_codes),
            },
            "dry_run": self.dry_run,
            "extra_headers": dict(self.extra_headers),
        }

    @classmethod
    def from_env(
        cls,
        prefix: str,
        *,
        default_model: str,
        default_base_url: str = "http://127.0.0.1:8000",
        default_endpoint_path: str = "/v1/chat/completions",
    ) -> NIMEndpointConfig:
        """Load ``<PREFIX>_BASE_URL``, ``_MODEL``, and related settings."""

        normalized = prefix.upper().rstrip("_")
        common_dry_run = _env_bool("NVIDIA_DRY_RUN", False)
        max_attempts = int(os.getenv(f"{normalized}_MAX_ATTEMPTS", "4"))
        return cls(
            base_url=os.getenv(f"{normalized}_BASE_URL", default_base_url),
            model=os.getenv(f"{normalized}_MODEL", default_model),
            endpoint_path=os.getenv(
                f"{normalized}_ENDPOINT_PATH", default_endpoint_path
            ),
            api_key=os.getenv(f"{normalized}_API_KEY") or None,
            api_key_env=os.getenv(f"{normalized}_API_KEY_ENV", "NVIDIA_API_KEY"),
            timeout_seconds=float(os.getenv(f"{normalized}_TIMEOUT_SECONDS", "120")),
            retry=RetryPolicy(
                max_attempts=max_attempts,
                initial_delay_seconds=float(
                    os.getenv(f"{normalized}_RETRY_INITIAL_SECONDS", "0.5")
                ),
                multiplier=float(os.getenv(f"{normalized}_RETRY_MULTIPLIER", "2")),
                max_delay_seconds=float(
                    os.getenv(f"{normalized}_RETRY_MAX_SECONDS", "8")
                ),
            ),
            dry_run=_env_bool(f"{normalized}_DRY_RUN", common_dry_run),
        )


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status_code: int
    body: Mapping[str, Any]
    headers: Mapping[str, str] = field(default_factory=dict)


class JsonTransport(Protocol):
    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> TransportResponse:
        """Send one JSON request."""


class UrllibJsonTransport:
    """Standard-library transport used when no custom transport is injected."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> TransportResponse:
        body = _stable_json(payload).encode("utf-8")
        request = urllib_request.Request(
            url=url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read()
                return TransportResponse(
                    status_code=int(response.status),
                    body=_decode_json_response(raw),
                    headers=dict(response.headers.items()),
                )
        except urllib_error.HTTPError as exc:
            raw = exc.read()
            return TransportResponse(
                status_code=int(exc.code),
                body=_decode_json_response(raw, allow_text=True),
                headers=dict(exc.headers.items()) if exc.headers else {},
            )


def _decode_json_response(raw: bytes, *, allow_text: bool = False) -> JsonObject:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        if allow_text:
            return {"error": {"message": raw.decode("utf-8", errors="replace")}}
        raise
    if isinstance(parsed, dict):
        return parsed
    return {"data": parsed}


class NIMRequestError(RuntimeError):
    """Raised after a NIM request cannot be completed or retried."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response: Mapping[str, Any] | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = dict(response or {})
        self.attempts = attempts

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "error": str(self),
            "status_code": self.status_code,
            "response": self.response,
            "attempts": self.attempts,
        }


class UnsupportedAssistantContentError(ValueError):
    """A successful chat response had no assistant payload this client supports."""


class OpenAICompatibleClient:
    """Retrying OpenAI chat-completions client used by all NVIDIA adapters."""

    def __init__(
        self,
        config: NIMEndpointConfig,
        *,
        transport: JsonTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibJsonTransport()
        self._sleep = sleep

    def chat_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
        request_key: str | None = None,
        response_validator: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> JsonObject:
        payload: JsonObject = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if extra_body:
            _deep_merge(payload, dict(extra_body))

        request_hash = _stable_hash(request_key or "", payload)
        if self.config.dry_run:
            return {
                "id": f"dryrun-{request_hash[:24]}",
                "object": "chat.completion",
                "model": self.config.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "dry_run": True,
                "request_sha256": request_hash,
            }

        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **dict(self.config.extra_headers),
        }
        if api_key := self.config.resolved_api_key:
            headers.setdefault("Authorization", f"Bearer {api_key}")

        last_exception: BaseException | None = None
        last_response: TransportResponse | None = None
        unsupported_content_attempts: list[dict[str, Any]] = []
        policy = self.config.retry
        for attempt in range(1, policy.max_attempts + 1):
            if attempt > 1:
                self._sleep(policy.delay_before_attempt(attempt))
            try:
                response = self.transport.post_json(
                    self.config.url,
                    headers=headers,
                    payload=payload,
                    timeout_seconds=self.config.timeout_seconds,
                )
            except OSError as exc:
                last_exception = exc
                if attempt < policy.max_attempts:
                    continue
                response_audit: JsonObject = {}
                if unsupported_content_attempts:
                    response_audit = {
                        "failure_kind": "assistant_content_retry_exhausted",
                        "successful_http_failures": unsupported_content_attempts,
                        "terminal_transport_error": {
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    }
                raise NIMRequestError(
                    f"NIM request failed after {attempt} attempts: {exc}",
                    response=response_audit,
                    attempts=attempt,
                ) from exc

            last_response = response
            if 200 <= response.status_code < 300:
                raw_response = dict(response.body)
                if response_validator is not None:
                    try:
                        response_validator(raw_response)
                    except UnsupportedAssistantContentError as exc:
                        unsupported_content_attempts.append(
                            {
                                "attempt": attempt,
                                "status_code": response.status_code,
                                "error": str(exc),
                                "raw_response": raw_response,
                            }
                        )
                        if attempt < policy.max_attempts:
                            continue
                        raise NIMRequestError(
                            "NIM returned successful HTTP responses without supported "
                            f"assistant content after {attempt} attempts",
                            status_code=response.status_code,
                            response={
                                "failure_kind": "unsupported_assistant_content",
                                "successful_http_failures": unsupported_content_attempts,
                                "last_raw_response": raw_response,
                            },
                            attempts=attempt,
                        ) from exc
                return raw_response
            if (
                response.status_code in policy.retry_status_codes
                and attempt < policy.max_attempts
            ):
                continue
            response_body: Mapping[str, Any] = response.body
            if unsupported_content_attempts:
                response_body = {
                    "failure_kind": "assistant_content_retry_exhausted",
                    "successful_http_failures": unsupported_content_attempts,
                    "terminal_http_response": {
                        "status_code": response.status_code,
                        "body": dict(response.body),
                    },
                }
            raise NIMRequestError(
                _error_message(response.body, response.status_code),
                status_code=response.status_code,
                response=response_body,
                attempts=attempt,
            )

        # Defensive fallback: the loop always returns or raises.
        raise NIMRequestError(
            f"NIM request failed: {last_exception or last_response}",
            status_code=last_response.status_code if last_response else None,
            response=last_response.body if last_response else None,
            attempts=policy.max_attempts,
        )


def _deep_merge(target: MutableMapping[str, Any], incoming: Mapping[str, Any]) -> None:
    for key, value in incoming.items():
        if (
            key in target
            and isinstance(target[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _error_message(body: Mapping[str, Any], status_code: int) -> str:
    error = body.get("error")
    if isinstance(error, Mapping) and error.get("message"):
        return f"NIM returned HTTP {status_code}: {error['message']}"
    return f"NIM returned HTTP {status_code}"


@dataclass(frozen=True, slots=True)
class ResponseContentCandidate:
    """One possible structured payload exposed by an OpenAI-compatible server."""

    source: str
    value: Any
    finish_reason: str | None = None


def _candidate_identity(value: Any) -> str:
    try:
        return _stable_hash(value)
    except (TypeError, ValueError):
        return _stable_hash(repr(value))


def response_content_candidates(
    response: Mapping[str, Any],
) -> tuple[ResponseContentCandidate, ...]:
    """Enumerate JSON-capable content shapes used by vLLM/OpenAI-compatible APIs.

    In addition to ``message.content``, servers and client gateways may expose a
    parsed object, content blocks, tool arguments, legacy completion text, or a
    Responses-API-style output array. Candidate order always prefers the most
    explicitly structured representation.
    """

    candidates: list[ResponseContentCandidate] = []
    seen: set[str] = set()

    def add(value: Any, source: str, finish_reason: str | None = None) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if not value.strip():
                return
            expanded: list[tuple[Any, str]] = [(value, source)]
        elif isinstance(value, Mapping):
            if isinstance(value.get("text"), str) and (
                "type" in value or set(value).issubset({"text", "type", "annotations"})
            ):
                expanded = [(value["text"], f"{source}.text")]
            elif "json" in value:
                expanded = [(value["json"], f"{source}.json")]
            else:
                expanded = [(dict(value), source)]
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            text_blocks: list[str] = []
            for item in value:
                if isinstance(item, str):
                    text_blocks.append(item)
                elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    text_blocks.append(str(item["text"]))
            if text_blocks:
                expanded = [("\n".join(text_blocks), f"{source}.text_blocks")]
            else:
                expanded = [(list(value), source)]
        else:
            return
        for expanded_value, expanded_source in expanded:
            identity = _candidate_identity(expanded_value)
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(
                ResponseContentCandidate(
                    source=expanded_source,
                    value=expanded_value,
                    finish_reason=finish_reason,
                )
            )

    choices = response.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        for choice_index, choice in enumerate(choices):
            if not isinstance(choice, Mapping):
                continue
            finish_reason = (
                str(choice["finish_reason"])
                if choice.get("finish_reason") is not None
                else None
            )
            message = choice.get("message")
            if isinstance(message, Mapping):
                add(
                    message.get("parsed"),
                    f"choices[{choice_index}].message.parsed",
                    finish_reason,
                )
                add(
                    message.get("content"),
                    f"choices[{choice_index}].message.content",
                    finish_reason,
                )
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, Sequence) and not isinstance(
                    tool_calls, (str, bytes)
                ):
                    for tool_index, tool_call in enumerate(tool_calls):
                        if not isinstance(tool_call, Mapping):
                            continue
                        function = tool_call.get("function")
                        if isinstance(function, Mapping):
                            add(
                                function.get("arguments"),
                                f"choices[{choice_index}].message.tool_calls[{tool_index}]"
                                ".function.arguments",
                                finish_reason,
                            )
                function_call = message.get("function_call")
                if isinstance(function_call, Mapping):
                    add(
                        function_call.get("arguments"),
                        f"choices[{choice_index}].message.function_call.arguments",
                        finish_reason,
                    )
                # Some reasoning-model gateways incorrectly place the final answer
                # here. Keep it last so ordinary content always wins.
                add(
                    message.get("reasoning_content"),
                    f"choices[{choice_index}].message.reasoning_content",
                    finish_reason,
                )
            add(choice.get("text"), f"choices[{choice_index}].text", finish_reason)

    add(response.get("parsed"), "parsed")
    add(response.get("output_text"), "output_text")
    output = response.get("output")
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        for output_index, item in enumerate(output):
            if not isinstance(item, Mapping):
                continue
            add(item.get("content"), f"output[{output_index}].content")
            add(item.get("text"), f"output[{output_index}].text")
    add(response.get("content"), "content")
    return tuple(candidates)


def chat_content(response: Mapping[str, Any]) -> str:
    """Return the highest-priority content candidate as text."""

    candidates = response_content_candidates(response)
    if not candidates:
        raise UnsupportedAssistantContentError(
            "response does not contain supported assistant content"
        )
    content = candidates[0].value
    return content if isinstance(content, str) else _stable_json(content)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized model response while retaining the complete provider payload."""

    content: str
    model: str
    request_id: str | None
    raw_response: Mapping[str, Any]
    dry_run: bool = False
    content_source: str | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "request_id": self.request_id,
            "raw_response": dict(self.raw_response),
            "dry_run": self.dry_run,
            "content_source": self.content_source,
            "finish_reason": self.finish_reason,
        }


def _image_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class _ImageChatAdapter:
    def __init__(
        self,
        config: NIMEndpointConfig,
        *,
        transport: JsonTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.client = OpenAICompatibleClient(config, transport=transport, sleep=sleep)

    def _invoke(
        self,
        image_bytes: bytes,
        *,
        prompt: str,
        document_id: str,
        page_number: int,
        mime_type: str,
        max_tokens: int,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        if not image_bytes and not self.config.dry_run:
            raise ValueError("image_bytes cannot be empty")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(image_bytes, mime_type),
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
        response = self.client.chat_completion(
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body=extra_body,
            request_key=f"{document_id}:{page_number}:{_stable_hash(image_bytes)}",
            response_validator=chat_content,
        )
        return ModelResponse(
            content=chat_content(response),
            model=str(response.get("model") or self.config.model),
            request_id=str(response["id"]) if response.get("id") else None,
            raw_response=response,
            dry_run=bool(response.get("dry_run", False)),
        )


class NemotronParseClient(_ImageChatAdapter):
    """Adapter for Nemotron Parse or another document-understanding VLM NIM."""

    # Nemotron Parse v1.2 was trained on this four-token task prompt. Natural-language
    # OCR instructions produce severely degraded output even though the endpoint still
    # returns HTTP 200, so keep the official prompt as the production default.
    DEFAULT_PROMPT = (
        "</s><s><predict_bbox><predict_classes><output_markdown>"
        "<predict_no_text_in_pic>"
    )

    def parse_page(
        self,
        image_bytes: bytes,
        *,
        document_id: str,
        page_number: int,
        mime_type: str = "image/png",
        prompt: str | None = None,
    ) -> ModelResponse:
        return self._invoke(
            image_bytes,
            prompt=prompt or self.DEFAULT_PROMPT,
            document_id=document_id,
            page_number=page_number,
            mime_type=mime_type,
            # Leave room for the required task prompt inside Parse v1.2's
            # 9,000-token model window. vLLM rejects prompt + output > 9,000.
            max_tokens=8192,
            extra_body={
                "top_k": 1,
                "repetition_penalty": 1.1,
                "skip_special_tokens": False,
            },
        )


class NemotronOCRClient(_ImageChatAdapter):
    """Adapter for exact-text/identifier verification using an OCR-capable NIM."""

    DEFAULT_PROMPT = (
        "Transcribe the page exactly for verification. Do not correct or infer "
        "characters. Focus on manufacturer names, model codes, part numbers, numeric "
        "specifications, units, and table cells. Return JSON as {\"elements\":["
        "{\"type\":\"ocr_line\",\"text\":\"...\",\"bbox\":[x0,y0,x1,y1],"
        "\"confidence\":0.0}]}."
    )

    def ocr_page(
        self,
        image_bytes: bytes,
        *,
        document_id: str,
        page_number: int,
        mime_type: str = "image/png",
        prompt: str | None = None,
    ) -> ModelResponse:
        return self._invoke(
            image_bytes,
            prompt=prompt or self.DEFAULT_PROMPT,
            document_id=document_id,
            page_number=page_number,
            mime_type=mime_type,
            max_tokens=8192,
        )


class GuidedJsonMode(StrEnum):
    """Provider dialect used to constrain LLM output."""

    NVIDIA_NVEXT = "nvidia_nvext"
    OPENAI_RESPONSE_FORMAT = "openai_response_format"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class GuidedJsonResult:
    data: Any
    response: ModelResponse
    schema_sha256: str
    attempts: tuple[ModelResponse, ...] = ()
    decode_failures: tuple[tuple[JsonCandidateFailure, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "response": self.response.to_dict(),
            "schema_sha256": self.schema_sha256,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "decode_failures": [
                [failure.to_dict() for failure in attempt_failures]
                for attempt_failures in self.decode_failures
            ],
        }


_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL
)
_LEADING_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_TRANSPORT_TOKEN_RE = re.compile(
    r"(?:</s>|<\|eot_id\|>|<\|im_end\|>|<\|endoftext\|>)", re.IGNORECASE
)


class JsonContentError(ValueError):
    """A model content candidate was empty, malformed, or truncated JSON."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        position: int | None = None,
        content_length: int = 0,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.position = position
        self.content_length = content_length


@dataclass(frozen=True, slots=True)
class JsonCandidateFailure:
    source: str
    kind: str
    message: str
    finish_reason: str | None
    content_length: int
    trimmed_content_length: int = 0
    trailing_whitespace_characters: int = 0
    content_sha256: str | None = None
    position: int | None = None
    content_head: str = ""
    content_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text_excerpt(value: Any, *, tail: bool = False, limit: int = 240) -> str:
    text = value if isinstance(value, str) else _stable_json(value)
    text = text.replace("\x00", "\\0")
    if tail:
        text = text.rstrip()
    return text[-limit:] if tail else text[:limit]


def _content_diagnostics(value: Any) -> tuple[int, int, int, str]:
    text = value if isinstance(value, str) else _stable_json(value)
    trimmed = text.rstrip()
    return (
        len(text),
        len(trimmed),
        len(text) - len(trimmed),
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _json_structure_state(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            if not stack or stack[-1] != pairs[character]:
                return "mismatched"
            stack.pop()
    return "unclosed" if in_string or escaped or stack else "complete"


def _classify_json_error(candidate: str, error: json.JSONDecodeError) -> str:
    stripped = candidate.rstrip()
    if not stripped:
        return "empty"
    structure_state = _json_structure_state(stripped)
    if structure_state == "mismatched":
        return "malformed"
    if structure_state == "unclosed":
        return "truncated"
    near_end = error.pos >= max(0, len(stripped) - 2)
    if near_end and error.msg in {
        "Expecting value",
        "Expecting property name enclosed in double quotes",
        "Expecting ',' delimiter",
        "Unterminated string starting at",
    }:
        return "truncated"
    return "malformed"


def _wrapper_is_safe(prefix: str, suffix: str) -> bool:
    """Allow prose/fence wrappers, but never discard a second JSON value."""

    cleaned_suffix = _TRANSPORT_TOKEN_RE.sub("", suffix).replace("```", "").strip()
    if cleaned_suffix:
        return False
    return not any(character in prefix for character in "{[")


def parse_json_content(content: Any) -> Any:
    """Decode one candidate without modifying or completing its JSON data.

    Complete Markdown fences, leading reasoning blocks, and recognized transport
    tokens are wrappers rather than data and may be removed. Missing commas,
    braces, quotes, or values are never inserted heuristically.
    """

    if isinstance(content, Mapping):
        return dict(content)
    if isinstance(content, list):
        return content
    if isinstance(content, (int, float, bool)) or content is None:
        return content
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="strict")
    if not isinstance(content, str):
        raise JsonContentError(
            f"unsupported JSON content type: {type(content).__name__}",
            kind="malformed",
        )

    raw = content.lstrip("\ufeff").strip()
    if not raw:
        raise JsonContentError("model returned empty JSON content", kind="empty")
    without_thinking = _LEADING_THINK_RE.sub("", raw).strip()
    variants: list[str] = [match.group(1).strip() for match in _JSON_FENCE_RE.finditer(raw)]
    for value in (without_thinking, raw):
        value = _TRANSPORT_TOKEN_RE.sub("", value).strip()
        if value and value not in variants:
            variants.append(value)

    failures: list[tuple[str, json.JSONDecodeError]] = []
    decoder = json.JSONDecoder()
    for candidate in variants:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as error:
            failures.append((candidate, error))

        starts = [index for index in (candidate.find("{"), candidate.find("[")) if index >= 0]
        if not starts:
            continue
        start = min(starts)
        try:
            decoded, end = decoder.raw_decode(candidate, start)
        except json.JSONDecodeError as error:
            failures.append((candidate[start:], error))
            continue
        if _wrapper_is_safe(candidate[:start], candidate[end:]):
            return decoded

    candidate, error = failures[0] if failures else (raw, None)
    if error is None:
        raise JsonContentError(
            "model content did not contain a JSON value",
            kind="malformed",
            content_length=len(candidate),
        )
    kind = _classify_json_error(candidate, error)
    raise JsonContentError(
        f"model returned {kind} JSON at character {error.pos}: {error.msg}",
        kind=kind,
        position=error.pos,
        content_length=len(candidate),
    ) from error


class _ResponseJsonDecodeError(ValueError):
    def __init__(
        self,
        response: ModelResponse,
        failures: tuple[JsonCandidateFailure, ...],
    ) -> None:
        self.response = response
        self.failures = failures
        super().__init__(failures[0].message if failures else "no JSON content candidate")


class GuidedJsonDecodeError(ValueError):
    """All bounded schema-constrained generation attempts failed JSON decoding."""

    def __init__(
        self,
        attempts: tuple[ModelResponse, ...],
        failures: tuple[tuple[JsonCandidateFailure, ...], ...],
        *,
        regeneration_request_error: NIMRequestError | None = None,
    ) -> None:
        self.attempts = attempts
        self.failures = failures
        self.regeneration_request_error = regeneration_request_error
        summaries = []
        for index, attempt_failures in enumerate(failures, start=1):
            first = attempt_failures[0] if attempt_failures else None
            if first is None:
                summaries.append(f"attempt {index}: no supported content")
            else:
                summaries.append(
                    f"attempt {index}: {first.kind} JSON from {first.source} "
                    f"(finish_reason={first.finish_reason!r}, chars={first.content_length}, "
                    f"trimmed={first.trimmed_content_length}, "
                    f"trailing_ws={first.trailing_whitespace_characters})"
                )
        message = (
            "structured model did not return valid JSON after "
            f"{len(attempts)} bounded attempt(s); " + "; ".join(summaries)
        )
        if regeneration_request_error is not None:
            message += (
                "; regeneration request failed before a response could be decoded: "
                f"{regeneration_request_error}"
            )
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "error_type": type(self).__name__,
            "error": str(self),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "failures": [
                [failure.to_dict() for failure in attempt_failures]
                for attempt_failures in self.failures
            ],
        }
        if self.regeneration_request_error is not None:
            payload["regeneration_request_error"] = (
                self.regeneration_request_error.to_dict()
            )
        return payload


def _model_response(
    raw: Mapping[str, Any], candidate: ResponseContentCandidate | None
) -> ModelResponse:
    value = candidate.value if candidate is not None else ""
    content = value if isinstance(value, str) else _stable_json(value)
    return ModelResponse(
        content=content,
        model=str(raw.get("model") or "unknown"),
        request_id=str(raw["id"]) if raw.get("id") else None,
        raw_response=raw,
        dry_run=bool(raw.get("dry_run", False)),
        content_source=candidate.source if candidate is not None else None,
        finish_reason=candidate.finish_reason if candidate is not None else None,
    )


def decode_json_response(raw: Mapping[str, Any]) -> tuple[Any, ModelResponse]:
    """Decode the first valid structured candidate in a provider response."""

    candidates = response_content_candidates(raw)
    failures: list[JsonCandidateFailure] = []
    for candidate in candidates:
        content_length, trimmed_length, trailing_whitespace, content_hash = (
            _content_diagnostics(candidate.value)
        )
        if candidate.finish_reason not in {
            None,
            "stop",
            "tool_calls",
            "function_call",
        }:
            kind = (
                "truncated"
                if candidate.finish_reason in {"length", "max_tokens"}
                else "incomplete"
            )
            failures.append(
                JsonCandidateFailure(
                    source=candidate.source,
                    kind=kind,
                    message=(
                        "provider marked the structured response incomplete with "
                        f"finish_reason={candidate.finish_reason!r}"
                    ),
                    finish_reason=candidate.finish_reason,
                    content_length=content_length,
                    trimmed_content_length=trimmed_length,
                    trailing_whitespace_characters=trailing_whitespace,
                    content_sha256=content_hash,
                    content_head=_text_excerpt(candidate.value),
                    content_tail=_text_excerpt(candidate.value, tail=True),
                )
            )
            continue
        try:
            data = parse_json_content(candidate.value)
        except JsonContentError as error:
            kind = (
                "truncated"
                if candidate.finish_reason in {"length", "max_tokens"}
                else error.kind
            )
            failures.append(
                JsonCandidateFailure(
                    source=candidate.source,
                    kind=kind,
                    message=str(error),
                    finish_reason=candidate.finish_reason,
                    content_length=content_length,
                    trimmed_content_length=trimmed_length,
                    trailing_whitespace_characters=trailing_whitespace,
                    content_sha256=content_hash,
                    position=error.position,
                    content_head=_text_excerpt(candidate.value),
                    content_tail=_text_excerpt(candidate.value, tail=True),
                )
            )
            continue
        return data, _model_response(raw, candidate)

    fallback = candidates[0] if candidates else None
    if not failures:
        failures.append(
            JsonCandidateFailure(
                source="response",
                kind="empty",
                message="provider response contained no supported JSON content",
                finish_reason=None,
                content_length=0,
            )
        )
    raise _ResponseJsonDecodeError(
        _model_response(raw, fallback), tuple(failures)
    )


class GuidedJsonClient:
    """Schema-constrained Nemotron LLM adapter for product record parsing."""

    def __init__(
        self,
        config: NIMEndpointConfig,
        *,
        mode: GuidedJsonMode = GuidedJsonMode.NVIDIA_NVEXT,
        serialize_nvext_schema: bool = True,
        schema_name: str = "industrial_catalog_extraction",
        json_regeneration_attempts: int = 1,
        transport: JsonTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if json_regeneration_attempts not in {0, 1}:
            raise ValueError("json_regeneration_attempts must be zero or one")
        self.config = config
        self.mode = mode
        self.serialize_nvext_schema = serialize_nvext_schema
        self.schema_name = schema_name
        self.json_regeneration_attempts = json_regeneration_attempts
        self.client = OpenAICompatibleClient(config, transport=transport, sleep=sleep)

    def generate_json(
        self,
        *,
        schema: Mapping[str, Any],
        user_prompt: str,
        system_prompt: str | None = None,
        source_key: str = "",
        max_tokens: int = 8192,
    ) -> GuidedJsonResult:
        schema_copy = json.loads(_stable_json(schema))
        schema_hash = _stable_hash(schema_copy)
        if self.config.dry_run:
            mock_data = _mock_value_for_schema(schema_copy)
            raw: JsonObject = {
                "id": f"dryrun-{_stable_hash(source_key, user_prompt, schema_copy)[:24]}",
                "object": "chat.completion",
                "model": self.config.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": _stable_json(mock_data),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "dry_run": True,
            }
            response = ModelResponse(
                content=chat_content(raw),
                model=self.config.model,
                request_id=str(raw["id"]),
                raw_response=raw,
                dry_run=True,
                content_source="choices[0].message.content",
                finish_reason="stop",
            )
            return GuidedJsonResult(
                mock_data,
                response,
                schema_hash,
                attempts=(response,),
            )

        messages: list[Mapping[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        extra_body: JsonObject = {}
        if self.mode in {GuidedJsonMode.NVIDIA_NVEXT, GuidedJsonMode.BOTH}:
            guided_schema: Any = schema_copy
            if self.serialize_nvext_schema:
                guided_schema = _stable_json(schema_copy)
            extra_body["nvext"] = {"guided_json": guided_schema}
        if self.mode in {GuidedJsonMode.OPENAI_RESPONSE_FORMAT, GuidedJsonMode.BOTH}:
            extra_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema_name,
                    "strict": True,
                    "schema": schema_copy,
                },
            }

        attempts: list[ModelResponse] = []
        failure_groups: list[tuple[JsonCandidateFailure, ...]] = []
        request_messages = list(messages)
        total_attempts = 1 + self.json_regeneration_attempts
        for attempt_index in range(total_attempts):
            try:
                raw = self.client.chat_completion(
                    request_messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    request_key=(
                        f"{source_key}:{schema_hash}:json-attempt-{attempt_index + 1}"
                    ),
                )
            except NIMRequestError as error:
                if attempts:
                    raise GuidedJsonDecodeError(
                        tuple(attempts),
                        tuple(failure_groups),
                        regeneration_request_error=error,
                    ) from error
                raise
            try:
                data, response = decode_json_response(raw)
            except _ResponseJsonDecodeError as error:
                attempts.append(error.response)
                failure_groups.append(error.failures)
                if attempt_index + 1 >= total_attempts:
                    raise GuidedJsonDecodeError(
                        tuple(attempts), tuple(failure_groups)
                    ) from error
                truncated = any(
                    failure.kind == "truncated" for failure in error.failures
                )
                diagnosis = "truncated" if truncated else "malformed"
                regeneration_prompt = (
                    user_prompt.rstrip()
                    + "\n\nREGENERATION INSTRUCTIONS:\n"
                    + f"The previous structured response was {diagnosis} and could "
                    "not be decoded. Regenerate the complete object from SOURCE "
                    "EVIDENCE. Output exactly one JSON object matching the schema, "
                    "with no Markdown or commentary. Keep raw strings concise, omit "
                    "unknown optional fields, and use [] only for required empty "
                    "collections. Emit compact JSON with no indentation, blank lines, "
                    "or trailing whitespace. Do not continue, quote, or patch the "
                    "previous response."
                )
                request_messages = []
                if system_prompt:
                    request_messages.append(
                        {"role": "system", "content": system_prompt}
                    )
                request_messages.append(
                    {"role": "user", "content": regeneration_prompt}
                )
                continue
            attempts.append(response)
            return GuidedJsonResult(
                data=data,
                response=response,
                schema_sha256=schema_hash,
                attempts=tuple(attempts),
                decode_failures=tuple(failure_groups),
            )

        raise AssertionError("bounded guided JSON generation loop did not terminate")


# More explicit alias used in design documents.
NemotronGuidedJsonClient = GuidedJsonClient


def _mock_value_for_schema(schema: Mapping[str, Any]) -> Any:
    if "const" in schema:
        return schema["const"]
    if enum := schema.get("enum"):
        return enum[0]
    for alternative_key in ("oneOf", "anyOf"):
        alternatives = schema.get(alternative_key)
        if isinstance(alternatives, Sequence) and alternatives:
            first = alternatives[0]
            if isinstance(first, Mapping):
                return _mock_value_for_schema(first)
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        schema_type = non_null[0] if non_null else "null"
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties))
        return {
            str(name): _mock_value_for_schema(properties[name])
            for name in required
            if name in properties and isinstance(properties[name], Mapping)
        }
    if schema_type == "array":
        return []
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    return ""


@dataclass(slots=True)
class NvidiaClientBundle:
    """All model adapters needed by the extraction pipeline."""

    parser: NemotronParseClient
    ocr: NemotronOCRClient
    llm: GuidedJsonClient

    @classmethod
    def from_env(cls) -> NvidiaClientBundle:
        parse_config = NIMEndpointConfig.from_env(
            "NVIDIA_PARSE",
            default_model="nvidia/nemotron-parse-v1.2",
            default_base_url=os.getenv("NVIDIA_NIM_BASE_URL", "http://127.0.0.1:8001"),
        )
        ocr_config = NIMEndpointConfig.from_env(
            "NVIDIA_OCR",
            default_model="nvidia/nemotron-ocr",
            default_base_url=os.getenv("NVIDIA_NIM_BASE_URL", "http://127.0.0.1:8002"),
        )
        llm_config = NIMEndpointConfig.from_env(
            "NVIDIA_LLM",
            default_model="nvidia/nemotron-3-nano-30b-a3b",
            default_base_url=os.getenv("NVIDIA_NIM_BASE_URL", "http://127.0.0.1:8003"),
        )
        mode = GuidedJsonMode(
            os.getenv("NVIDIA_LLM_GUIDED_JSON_MODE", GuidedJsonMode.NVIDIA_NVEXT.value)
        )
        serialize = _env_bool("NVIDIA_LLM_SERIALIZE_GUIDED_SCHEMA", True)
        return cls(
            parser=NemotronParseClient(parse_config),
            ocr=NemotronOCRClient(ocr_config),
            llm=GuidedJsonClient(
                llm_config,
                mode=mode,
                serialize_nvext_schema=serialize,
            ),
        )


__all__ = [
    "GuidedJsonClient",
    "GuidedJsonDecodeError",
    "GuidedJsonMode",
    "GuidedJsonResult",
    "JsonCandidateFailure",
    "JsonContentError",
    "JsonTransport",
    "ModelResponse",
    "NIMEndpointConfig",
    "NIMRequestError",
    "NemotronGuidedJsonClient",
    "NemotronOCRClient",
    "NemotronParseClient",
    "NvidiaClientBundle",
    "OpenAICompatibleClient",
    "ResponseContentCandidate",
    "RetryPolicy",
    "TransportResponse",
    "UnsupportedAssistantContentError",
    "UrllibJsonTransport",
    "chat_content",
    "decode_json_response",
    "parse_json_content",
    "response_content_candidates",
]
