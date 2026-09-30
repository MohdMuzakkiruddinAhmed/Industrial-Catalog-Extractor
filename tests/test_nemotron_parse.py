from __future__ import annotations

import pytest

from industrial_catalog.extraction import _elements_from_model_response
from industrial_catalog.nvidia_clients import (
    ModelResponse,
    NemotronParseClient,
    NIMEndpointConfig,
    NIMRequestError,
    RetryPolicy,
    TransportResponse,
)
from industrial_catalog.validation import hydrate_product_evidence


class SequenceTransport:
    def __init__(self, responses: list[TransportResponse]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, object]] = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.payloads.append(dict(payload))
        return self.responses.pop(0)


def test_nemotron_parse_uses_official_v12_prompt() -> None:
    assert NemotronParseClient.DEFAULT_PROMPT == (
        "</s><s><predict_bbox><predict_classes><output_markdown>"
        "<predict_no_text_in_pic>"
    )


def test_nemotron_parse_retries_transient_empty_assistant_content() -> None:
    transport = SequenceTransport(
        [
            TransportResponse(
                200,
                {
                    "id": "empty-first",
                    "model": "parse-test",
                    "choices": [{"message": {"content": "   "}}],
                },
            ),
            TransportResponse(
                200,
                {
                    "id": "recovered",
                    "model": "parse-test",
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "<x_0.1><y_0.2>PANDUIT"
                                    "<x_0.8><y_0.3><class_Page-header>"
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            ),
        ]
    )
    sleeps: list[float] = []
    client = NemotronParseClient(
        NIMEndpointConfig(
            base_url="http://parse.test",
            model="parse-test",
            retry=RetryPolicy(
                max_attempts=3,
                initial_delay_seconds=0.25,
                multiplier=2.0,
                max_delay_seconds=1.0,
            ),
        ),
        transport=transport,
        sleep=sleeps.append,
    )

    response = client.parse_page(
        b"png",
        document_id="document-1",
        page_number=7,
    )

    assert response.request_id == "recovered"
    assert "PANDUIT" in response.content
    assert len(transport.payloads) == 2
    assert sleeps == [0.25]


def test_nemotron_parse_exhausted_empty_content_is_auditable() -> None:
    raw_responses = [
        {
            "id": "empty-1",
            "model": "parse-test",
            "choices": [{"message": {"content": ""}}],
        },
        {
            "id": "empty-2",
            "model": "parse-test",
            "choices": [],
        },
    ]
    transport = SequenceTransport(
        [TransportResponse(200, response) for response in raw_responses]
    )
    sleeps: list[float] = []
    client = NemotronParseClient(
        NIMEndpointConfig(
            base_url="http://parse.test",
            model="parse-test",
            retry=RetryPolicy(max_attempts=2, initial_delay_seconds=0.1),
        ),
        transport=transport,
        sleep=sleeps.append,
    )

    with pytest.raises(NIMRequestError) as captured:
        client.parse_page(
            b"png",
            document_id="document-1",
            page_number=7,
        )

    error = captured.value
    assert error.status_code == 200
    assert error.attempts == 2
    assert sleeps == [0.1]
    assert error.response["failure_kind"] == "unsupported_assistant_content"
    failures = error.response["successful_http_failures"]
    assert [failure["raw_response"]["id"] for failure in failures] == [
        "empty-1",
        "empty-2",
    ]
    assert error.response["last_raw_response"] == raw_responses[-1]
    audit = error.to_dict()
    assert audit["response"]["successful_http_failures"][0]["attempt"] == 1
    assert "after 2 attempts" in audit["error"]


def test_tagged_parse_output_becomes_grounded_elements() -> None:
    response = ModelResponse(
        content=(
            "<x_0.10><y_0.20># Lovejoy Couplings"
            "<x_0.80><y_0.30><class_Title>\n\n"
            "<x_0.12><y_0.35>Part L-110; bore 1 inch"
            "<x_0.88><y_0.44><class_List-item>"
        ),
        model="nvidia/NVIDIA-Nemotron-Parse-v1.2",
        request_id="req-1",
        raw_response={},
    )

    elements = _elements_from_model_response(
        response,
        document_id="doc-1",
        page_number=1,
        extraction_method="nemotron_parse",
        source_path="catalog.pdf",
        sequence_offset=0,
    )

    assert [element.element_type for element in elements] == ["Title", "List-item"]
    assert [element.text for element in elements] == [
        "# Lovejoy Couplings",
        "Part L-110; bore 1 inch",
    ]
    assert elements[0].bbox is not None
    assert elements[0].bbox.to_list() == [0.1, 0.2, 0.8, 0.3]
    assert elements[0].metadata["bbox_coordinate_space"] == "normalized"

    resolution = hydrate_product_evidence(
        {
            "manufacturer": {
                "raw": "Lovejoy",
                "normalized": "Lovejoy",
                "evidence": [{"element_id": elements[0].element_id}],
            }
        },
        extracted_elements=elements,
        source_document_id="doc-1",
        source_path="catalog.pdf",
    )
    evidence = resolution.payload["manufacturer"]["evidence"][0]
    assert evidence["bbox"]["coordinate_space"] == "normalized"
