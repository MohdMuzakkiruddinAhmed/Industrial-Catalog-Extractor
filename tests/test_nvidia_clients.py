from __future__ import annotations

import json

import pytest

from industrial_catalog.extraction import DocumentExtraction, ExtractionPipeline
from industrial_catalog.nvidia_clients import (
    GuidedJsonClient,
    GuidedJsonDecodeError,
    GuidedJsonMode,
    JsonContentError,
    NIMEndpointConfig,
    RetryPolicy,
    TransportResponse,
    decode_json_response,
    parse_json_content,
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"products": {"type": "array", "items": {"type": "object"}}},
    "required": ["products"],
}


class SequenceTransport:
    def __init__(self, responses: list[TransportResponse]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, object]] = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.payloads.append(dict(payload))
        return self.responses.pop(0)


def completion(
    content: object,
    *,
    request_id: str = "completion-1",
    finish_reason: str = "stop",
) -> TransportResponse:
    return TransportResponse(
        200,
        {
            "id": request_id,
            "model": "nemotron-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
        },
    )


def make_client(transport: SequenceTransport) -> GuidedJsonClient:
    return GuidedJsonClient(
        NIMEndpointConfig(
            base_url="http://llm.test",
            model="nemotron-test",
            retry=RetryPolicy(max_attempts=1),
        ),
        mode=GuidedJsonMode.OPENAI_RESPONSE_FORMAT,
        transport=transport,
    )


def test_parse_json_content_accepts_fence_reasoning_wrapper_and_transport_token() -> None:
    content = (
        "<think>internal analysis that is not the answer</think>\n"
        "Here is the result:\n```json\n{\"products\": []}\n```</s>"
    )

    assert parse_json_content(content) == {"products": []}


@pytest.mark.parametrize(
    "raw, expected_source",
    [
        (
            {
                "model": "test",
                "choices": [
                    {
                        "message": {
                            "parsed": {"products": []},
                            "content": "not json",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            "choices[0].message.parsed",
        ),
        (
            {
                "model": "test",
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "output_text", "text": '{"products": []}'}
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            "choices[0].message.content.text_blocks",
        ),
        (
            {
                "model": "test",
                "choices": [
                    {
                        "message": {
                            "content": "malformed {",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "emit_products",
                                        "arguments": '{"products": []}',
                                    }
                                }
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            "choices[0].message.tool_calls[0].function.arguments",
        ),
        (
            {
                "model": "test",
                "choices": [
                    {"text": '{"products": []}', "finish_reason": "stop"}
                ],
            },
            "choices[0].text",
        ),
        (
            {
                "model": "test",
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": '{"products": []}'}
                        ]
                    }
                ],
            },
            "output[0].content.text_blocks",
        ),
    ],
)
def test_decode_json_response_supports_public_server_content_shapes(
    raw: dict[str, object], expected_source: str
) -> None:
    data, response = decode_json_response(raw)

    assert data == {"products": []}
    assert response.content_source == expected_source


def test_missing_comma_is_rejected_instead_of_locally_repaired() -> None:
    with pytest.raises(JsonContentError) as captured:
        parse_json_content('{"products": [{"a": 1 "b": 2}]}')

    assert captured.value.kind == "malformed"
    assert "Expecting ',' delimiter" in str(captured.value)


def test_guided_client_regenerates_once_after_malformed_json() -> None:
    transport = SequenceTransport(
        [
            completion('{"products": [{"a": 1 "b": 2}]}', request_id="bad"),
            completion('{"products": []}', request_id="good"),
        ]
    )
    client = make_client(transport)

    result = client.generate_json(
        schema=SCHEMA,
        user_prompt="SOURCE EVIDENCE: []",
        system_prompt="detailed thinking off",
        source_key="document-1",
    )

    assert result.data == {"products": []}
    assert result.response.request_id == "good"
    assert [attempt.request_id for attempt in result.attempts] == ["bad", "good"]
    assert len(transport.payloads) == 2
    retry_messages = transport.payloads[1]["messages"]
    assert isinstance(retry_messages, list)
    assert [message["role"] for message in retry_messages] == ["system", "user"]
    assert retry_messages[0]["content"] == "detailed thinking off"
    assert retry_messages[1]["content"].startswith("SOURCE EVIDENCE: []")
    assert "Regenerate the complete object" in retry_messages[-1]["content"]
    assert transport.payloads[0]["response_format"]["type"] == "json_schema"
    assert transport.payloads[1]["response_format"]["type"] == "json_schema"


def test_retry_http_failure_preserves_first_decode_attempt_in_audit() -> None:
    transport = SequenceTransport(
        [
            completion('{"products": [}', request_id="first-malformed"),
            TransportResponse(
                400,
                {
                    "error": {
                        "message": (
                            "Conversation roles must alternate between user/tool and assistant"
                        )
                    }
                },
            ),
        ]
    )

    with pytest.raises(GuidedJsonDecodeError) as captured:
        make_client(transport).generate_json(
            schema=SCHEMA,
            user_prompt="SOURCE EVIDENCE: []",
            system_prompt="detailed thinking off",
        )

    error = captured.value
    assert [attempt.request_id for attempt in error.attempts] == ["first-malformed"]
    assert len(error.failures) == 1
    assert error.failures[0][0].kind == "malformed"
    retry_messages = transport.payloads[1]["messages"]
    assert isinstance(retry_messages, list)
    assert [message["role"] for message in retry_messages] == ["system", "user"]
    audit = error.to_dict()
    assert audit["attempts"][0]["raw_response"]["id"] == "first-malformed"
    assert audit["failures"][0][0]["kind"] == "malformed"
    request_error = audit["regeneration_request_error"]
    assert request_error["error_type"] == "NIMRequestError"
    assert request_error["status_code"] == 400
    assert "Conversation roles must alternate" in request_error["error"]
    assert "regeneration request failed" in audit["error"]


def test_finish_reason_length_is_reported_as_truncation_before_retry() -> None:
    padded_but_incomplete = '{"products": []}' + ("\n" * 100)
    transport = SequenceTransport(
        [
            completion(
                padded_but_incomplete,
                request_id="truncated",
                finish_reason="length",
            ),
            completion('{"products": []}', request_id="complete"),
        ]
    )
    result = make_client(transport).generate_json(
        schema=SCHEMA,
        user_prompt="SOURCE EVIDENCE: []",
    )

    assert result.data == {"products": []}
    assert result.attempts[0].finish_reason == "length"
    failure = result.decode_failures[0][0]
    assert failure.kind == "truncated"
    assert failure.content_length == len(padded_but_incomplete)
    assert failure.trimmed_content_length == len('{"products": []}')
    assert failure.trailing_whitespace_characters == 100
    assert failure.content_sha256 is not None
    assert "previous structured response was truncated" in str(
        transport.payloads[1]["messages"][-1]["content"]
    )
    assert "no indentation, blank lines, or trailing whitespace" in str(
        transport.payloads[1]["messages"][-1]["content"]
    )


def test_persistent_malformed_json_raises_auditable_error_after_two_attempts() -> None:
    transport = SequenceTransport(
        [
            completion('{"products": [}', request_id="bad-1"),
            completion('{"products": [}', request_id="bad-2"),
        ]
    )

    with pytest.raises(GuidedJsonDecodeError) as captured:
        make_client(transport).generate_json(
            schema=SCHEMA,
            user_prompt="SOURCE EVIDENCE: []",
        )

    error = captured.value
    assert len(transport.payloads) == 2
    assert [attempt.request_id for attempt in error.attempts] == ["bad-1", "bad-2"]
    assert len(error.failures) == 2
    audit = error.to_dict()
    assert audit["failures"][0][0]["kind"] == "malformed"
    assert audit["attempts"][0]["raw_response"]["id"] == "bad-1"
    assert "after 2 bounded attempt(s)" in audit["error"]


def test_direct_parsed_object_does_not_trigger_semantic_retry() -> None:
    transport = SequenceTransport(
        [
            TransportResponse(
                200,
                {
                    "id": "parsed",
                    "model": "nemotron-test",
                    "choices": [
                        {
                            "message": {
                                "parsed": {"products": []},
                                "content": None,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        ]
    )

    result = make_client(transport).generate_json(
        schema=SCHEMA,
        user_prompt="SOURCE EVIDENCE: []",
    )

    assert result.data == {"products": []}
    assert len(transport.payloads) == 1
    assert json.loads(result.response.content) == {"products": []}


def test_structured_pipeline_uses_exact_nemotron_non_reasoning_system_prompt() -> None:
    transport = SequenceTransport([completion('{"products": []}')])
    client = make_client(transport)
    pipeline = ExtractionPipeline(llm=client)
    extraction = DocumentExtraction(
        document_id="document-1",
        source_path="catalog.pdf",
        pages=(),
    )

    pipeline.parse_structured(extraction, schema=SCHEMA, instructions="Extract products.")

    messages = transport.payloads[0]["messages"]
    assert messages[0] == {"role": "system", "content": "detailed thinking off"}
    user_prompt = messages[1]["content"]
    assert "complete SOURCE EVIDENCE array as one catalog-page context" in user_prompt
    assert "both an exact\n   manufacturer name and an exact product or part name" in user_prompt
    assert "Never create products from page headers, footers" in user_prompt
    assert "Emit multiple products only when explicit distinct orderable" in user_prompt
    assert 'return exactly {"products":[]}' in user_prompt
    assert "copy an element_id supplied in SOURCE EVIDENCE" in user_prompt
    assert "explicitly states a key/value\n   relationship" in user_prompt
    assert "prefer an explicit\n   legal company or brand line" in user_prompt
    assert "over a\n   possessive marketing phrase" in user_prompt
    assert "must\n   never create a separate product record" in user_prompt
    assert "ADDITIONAL OUTPUT RULES:\nExtract products." in user_prompt
