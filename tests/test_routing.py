from __future__ import annotations

import json

import pytest

from industrial_catalog.extraction import (
    ExtractionPipeline,
    MemoryCheckpointStore,
    PageInput,
)
from industrial_catalog.nvidia_clients import (
    GuidedJsonClient,
    NemotronOCRClient,
    NemotronParseClient,
    NIMEndpointConfig,
    OpenAICompatibleClient,
    RetryPolicy,
    TransportResponse,
)
from industrial_catalog.routing import (
    PageRoute,
    PageRouter,
    RoutingConfig,
    route_page,
    signals_from_text,
)

GOOD_NATIVE_TEXT = """
ACME Industrial Controls — Motor Starter Product Catalog
Part number MS-4400-24V is a three-phase motor starter for factory automation.
Rated voltage 24 VDC. Contact rating 10 A. Operating temperature -20 to 60 C.
The enclosure is IP67 and the product conforms to IEC 60947 requirements.
For installation dimensions and accessories, consult the following product pages.
"""


def test_good_native_text_uses_native_route() -> None:
    decision = route_page(GOOD_NATIVE_TEXT)

    assert decision.route is PageRoute.NATIVE_TEXT
    assert decision.requires_vlm is False
    assert decision.requires_ocr_verification is False
    assert decision.reasons == ("native_text_quality_passed",)
    assert decision.signals.identifier_count >= 1


def test_empty_scanned_page_uses_vlm_and_ocr() -> None:
    decision = route_page("", image_coverage=1.0)

    assert decision.route is PageRoute.DOCUMENT_VLM
    assert decision.requires_ocr_verification is True
    assert decision.reasons == (
        "insufficient_native_characters",
        "insufficient_native_words",
        "low_alphanumeric_ratio",
        "image_heavy_layout",
        "ocr_identifier_verification",
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_reason"),
    [
        ({"table_count": 1}, "table_layout"),
        ({"is_multi_column": True}, "multi_column_layout"),
        ({"image_coverage": 0.45}, "image_heavy_layout"),
    ],
)
def test_complex_page_signals_force_vlm(kwargs: dict[str, object], expected_reason: str) -> None:
    decision = route_page(GOOD_NATIVE_TEXT, **kwargs)

    assert decision.route is PageRoute.DOCUMENT_VLM
    assert expected_reason in decision.reasons


def test_corrupt_native_text_forces_vlm() -> None:
    corrupted = GOOD_NATIVE_TEXT + ("\ufffd" * 20)
    decision = route_page(corrupted)

    assert decision.route is PageRoute.DOCUMENT_VLM
    assert "corrupt_or_replacement_glyphs" in decision.reasons


def test_custom_policy_can_keep_tables_on_native_route() -> None:
    router = PageRouter(
        RoutingConfig(
            use_vlm_for_tables=False,
            use_vlm_for_multi_column=False,
            use_vlm_for_image_heavy_pages=False,
        )
    )
    decision = router.decide(
        GOOD_NATIVE_TEXT,
        table_count=2,
        is_multi_column=True,
        image_coverage=0.9,
    )

    assert decision.route is PageRoute.NATIVE_TEXT


def test_native_identifier_ocr_verification_is_configurable() -> None:
    router = PageRouter(
        RoutingConfig(verify_native_identifiers_with_ocr=True)
    )
    decision = router.decide(GOOD_NATIVE_TEXT)

    assert decision.route is PageRoute.NATIVE_TEXT
    assert decision.requires_ocr_verification is True
    assert decision.reasons[-1] == "ocr_identifier_verification"


def test_signal_generation_is_bounded_and_deterministic() -> None:
    first = signals_from_text(
        GOOD_NATIVE_TEXT,
        image_coverage=float("inf"),
        table_count=-4,
    )
    second = signals_from_text(GOOD_NATIVE_TEXT, image_coverage=0.0)

    assert first == second
    assert 0.0 <= first.printable_ratio <= 1.0
    assert 0.0 <= first.alphanumeric_ratio <= 1.0
    assert first.table_count == 0


class _SequenceTransport:
    def __init__(self, responses: list[TransportResponse]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, object]] = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.payloads.append(dict(payload))
        return self.responses.pop(0)


def test_nim_client_retries_transient_errors_deterministically() -> None:
    transport = _SequenceTransport(
        [
            TransportResponse(503, {"error": {"message": "warming"}}),
            TransportResponse(
                200,
                {
                    "id": "ok",
                    "model": "test-model",
                    "choices": [{"message": {"content": "done"}}],
                },
            ),
        ]
    )
    sleeps: list[float] = []
    client = OpenAICompatibleClient(
        NIMEndpointConfig(
            base_url="http://nim.test",
            model="test-model",
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

    response = client.chat_completion([{"role": "user", "content": "hello"}])

    assert response["id"] == "ok"
    assert len(transport.payloads) == 2
    assert sleeps == [0.25]


def test_guided_json_dry_run_is_schema_valid_and_reproducible() -> None:
    client = GuidedJsonClient(
        NIMEndpointConfig(
            base_url="",
            model="nemotron-test",
            dry_run=True,
        )
    )
    schema = {
        "type": "object",
        "properties": {
            "manufacturer": {"type": ["string", "null"]},
            "products": {"type": "array", "items": {"type": "object"}},
            "page": {"type": "integer"},
        },
        "required": ["manufacturer", "products", "page"],
    }

    first = client.generate_json(schema=schema, user_prompt="extract", source_key="doc")
    second = client.generate_json(schema=schema, user_prompt="extract", source_key="doc")

    assert first.data == {"manufacturer": "", "products": [], "page": 0}
    assert first.response.request_id == second.response.request_id
    assert first.schema_sha256 == second.schema_sha256


def test_page_checkpoint_avoids_repeating_dry_run_model_calls() -> None:
    parse_config = NIMEndpointConfig(
        base_url="", model="parse-test", dry_run=True
    )
    ocr_config = NIMEndpointConfig(base_url="", model="ocr-test", dry_run=True)
    pipeline = ExtractionPipeline(
        parser=NemotronParseClient(parse_config),
        ocr=NemotronOCRClient(ocr_config),
    )
    checkpoints = MemoryCheckpointStore()
    image_load_count = 0

    def first_image() -> bytes:
        nonlocal image_load_count
        image_load_count += 1
        return b"fake-png"

    first = pipeline.extract_pages(
        "doc-1",
        [PageInput(page_number=1, image_loader=first_image, image_coverage=1.0)],
        checkpoint=checkpoints,
    )

    def must_not_render() -> bytes:
        raise AssertionError("checkpoint restore should occur before image rendering")

    second = pipeline.extract_pages(
        "doc-1",
        [PageInput(page_number=1, image_loader=must_not_render, image_coverage=1.0)],
        checkpoint=checkpoints,
    )

    assert image_load_count == 1
    assert first.checkpoint_hits == 0
    assert second.checkpoint_hits == 1
    assert first.pages[0].fingerprint == second.pages[0].fingerprint
    assert len(first.pages[0].model_responses) == 2


def test_model_elements_keep_parse_and_ocr_evidence_separate() -> None:
    def response(model: str, method_text: str) -> TransportResponse:
        content = json.dumps(
            {
                "elements": [
                    {
                        "type": "part_number",
                        "text": method_text,
                        "bbox": [10, 20, 100, 40],
                        "confidence": 0.98,
                    }
                ]
            }
        )
        return TransportResponse(
            200,
            {
                "id": model,
                "model": model,
                "choices": [{"message": {"content": content}}],
            },
        )

    parse_transport = _SequenceTransport([response("parse", "MS-4400-24V")])
    ocr_transport = _SequenceTransport([response("ocr", "MS-4400-24V")])
    config = NIMEndpointConfig(base_url="http://nim.test", model="unused")
    pipeline = ExtractionPipeline(
        parser=NemotronParseClient(config, transport=parse_transport),
        ocr=NemotronOCRClient(config, transport=ocr_transport),
    )

    result = pipeline.extract_pages(
        "doc-evidence",
        [PageInput(page_number=1, image_bytes=b"png", image_coverage=1.0)],
    )
    elements = result.pages[0].elements

    assert len(elements) == 2
    assert {item.extraction_method for item in elements} == {
        "nemotron_parse",
        "nemotron_ocr_verification",
    }
    assert elements[0].element_id != elements[1].element_id
    assert elements[0].bbox is not None
    assert elements[0].raw_payload["text"] == "MS-4400-24V"
