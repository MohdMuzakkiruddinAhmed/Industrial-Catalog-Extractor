from __future__ import annotations

import json

import pytest

from industrial_catalog.extraction import (
    ExtractionPipeline,
    JsonPageCheckpointStore,
    MemoryCheckpointStore,
    NativeTextBlock,
    PageExtraction,
    PageInput,
    PyMuPDFPageSource,
)
from industrial_catalog.nvidia_clients import (
    NemotronOCRClient,
    NemotronParseClient,
    NIMEndpointConfig,
    NIMRequestError,
    RetryPolicy,
    TransportResponse,
)
from industrial_catalog.routing import PageRoute, PageRouter, RoutingConfig

MEANINGFUL_NATIVE_TEXT = """
Lovejoy, LLC Industrial Couplings Catalog
The L-110 flexible jaw coupling is intended for industrial power transmission.
Part number L-110 has a rated torque of 89 newton metres and a maximum speed of
5,000 RPM. Operating temperature is -40 to 100 C. See the dimensional table for
hub bore options, keyway sizes, materials, and installation requirements.
"""


class SequenceTransport:
    def __init__(self, responses: list[TransportResponse]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, object]] = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.payloads.append(dict(payload))
        return self.responses.pop(0)


def empty_parse_response(request_id: str) -> TransportResponse:
    return TransportResponse(
        200,
        {
            "id": request_id,
            "model": "parse-test",
            "choices": [{"message": {"content": "   "}}],
        },
    )


def parser_with_responses(
    responses: list[TransportResponse],
) -> tuple[NemotronParseClient, SequenceTransport]:
    transport = SequenceTransport(responses)
    parser = NemotronParseClient(
        NIMEndpointConfig(
            base_url="http://parse.test",
            model="parse-test",
            retry=RetryPolicy(max_attempts=len(responses), initial_delay_seconds=0),
        ),
        transport=transport,
        sleep=lambda _: None,
    )
    return parser, transport


def test_semantic_vlm_exhaustion_falls_back_to_meaningful_native_evidence() -> None:
    parser, transport = parser_with_responses(
        [empty_parse_response("empty-1"), empty_parse_response("empty-2")]
    )
    pipeline = ExtractionPipeline(
        parser=parser,
        router=PageRouter(RoutingConfig(verify_vlm_with_ocr=False)),
    )
    page = PageInput(
        page_number=6,
        native_text=MEANINGFUL_NATIVE_TEXT,
        native_blocks=(
            NativeTextBlock(
                text=MEANINGFUL_NATIVE_TEXT.strip(),
                metadata={"native_block_number": 0},
            ),
        ),
        image_bytes=b"rendered-page",
        is_multi_column=True,
    )

    result = pipeline.extract_pages("lovejoy", [page])
    extracted = result.pages[0]

    assert extracted.routing.route is PageRoute.DOCUMENT_VLM
    assert len(transport.payloads) == 2
    assert [element.text for element in extracted.elements] == [
        MEANINGFUL_NATIVE_TEXT.strip()
    ]
    assert extracted.elements[0].extraction_method == "pdf_native_text"
    assert extracted.elements[0].metadata == {
        "native_block_number": 0,
        "vlm_fallback": True,
        "vlm_failure_kind": "unsupported_assistant_content",
    }
    audit = extracted.model_responses[0]
    assert audit["event"] == "vlm_unsupported_content_native_fallback"
    assert audit["outcome"] == "fallback_succeeded"
    assert audit["source_route"] == "document_vlm"
    assert audit["fallback_route"] == "native_text"
    assert audit["ocr_verification_required"] is False
    failures = audit["error_details"]["response"]["successful_http_failures"]
    assert [failure["raw_response"]["id"] for failure in failures] == [
        "empty-1",
        "empty-2",
    ]


def test_deterministically_blank_page_is_valid_and_does_not_render() -> None:
    def must_not_render() -> bytes:
        raise AssertionError("a deterministically blank page must not be rendered")

    pipeline = ExtractionPipeline()
    result = pipeline.extract_pages(
        "blank-document",
        [
            PageInput(
                page_number=2,
                image_loader=must_not_render,
                metadata={
                    "drawing_count": 0,
                    "embedded_image_count": 0,
                    "annotation_count": 0,
                    "link_count": 0,
                    "source_object_inventory_complete": True,
                },
            )
        ],
    )
    extracted = result.pages[0]

    assert extracted.routing.route is PageRoute.DOCUMENT_VLM
    assert extracted.elements == ()
    assert extracted.model_responses[0]["event"] == "deterministic_blank_page"
    assert extracted.model_responses[0]["outcome"] == "empty_page_extraction"
    restored = PageExtraction.from_dict(extracted.to_dict())
    assert restored.elements == ()
    assert restored.model_responses == extracted.model_responses


@pytest.mark.parametrize(
    "page",
    [
        PageInput(page_number=1, image_bytes=b"image", image_coverage=1.0),
        PageInput(
            page_number=1,
            image_loader=lambda: b"vector-render",
            metadata={
                "drawing_count": 1,
                "embedded_image_count": 0,
                "annotation_count": 0,
                "link_count": 0,
                "source_object_inventory_complete": True,
            },
        ),
        PageInput(
            page_number=1,
            image_loader=lambda: b"annotation-render",
            metadata={
                "drawing_count": 0,
                "embedded_image_count": 0,
                "annotation_count": 1,
                "link_count": 0,
                "source_object_inventory_complete": True,
            },
        ),
        PageInput(
            page_number=1,
            image_loader=lambda: b"link-render",
            metadata={
                "drawing_count": 0,
                "embedded_image_count": 0,
                "annotation_count": 0,
                "link_count": 1,
                "source_object_inventory_complete": True,
            },
        ),
        PageInput(page_number=1, image_loader=lambda: b"unknown-render"),
    ],
    ids=[
        "image-only",
        "vector-only",
        "annotation-only",
        "link-only",
        "unknown-inventory",
    ],
)
def test_nonblank_page_without_native_evidence_fails_closed(page: PageInput) -> None:
    parser, transport = parser_with_responses(
        [empty_parse_response("empty-1"), empty_parse_response("empty-2")]
    )

    with pytest.raises(NIMRequestError) as captured:
        ExtractionPipeline(parser=parser).extract_pages("nonblank", [page])

    assert captured.value.response["failure_kind"] == "unsupported_assistant_content"
    assert len(transport.payloads) == 2


def test_native_fallback_keeps_independent_ocr_verification() -> None:
    parser, _ = parser_with_responses([empty_parse_response("empty")])
    ocr_transport = SequenceTransport(
        [
            TransportResponse(
                200,
                {
                    "id": "ocr-ok",
                    "model": "ocr-test",
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"elements":[{"type":"ocr_line",'
                                    '"text":"L-110","confidence":0.99}]}'
                                )
                            }
                        }
                    ],
                },
            )
        ]
    )
    ocr = NemotronOCRClient(
        NIMEndpointConfig(base_url="http://ocr.test", model="ocr-test"),
        transport=ocr_transport,
    )
    page = PageInput(
        page_number=1,
        native_text=MEANINGFUL_NATIVE_TEXT,
        image_bytes=b"rendered-page",
        is_multi_column=True,
    )

    result = ExtractionPipeline(parser=parser, ocr=ocr).extract_pages("ocr", [page])

    assert [element.extraction_method for element in result.elements] == [
        "pdf_native_text",
        "nemotron_ocr_verification",
    ]
    assert result.pages[0].model_responses[0]["ocr_verification_required"] is True
    assert result.pages[0].model_responses[1]["request_id"] == "ocr-ok"


def test_mixed_semantic_and_server_exhaustion_does_not_use_native_fallback() -> None:
    parser, transport = parser_with_responses(
        [
            empty_parse_response("empty-first"),
            TransportResponse(503, {"error": {"message": "service unavailable"}}),
        ]
    )
    page = PageInput(
        page_number=1,
        native_text=MEANINGFUL_NATIVE_TEXT,
        image_bytes=b"rendered-page",
        table_count=1,
    )

    with pytest.raises(NIMRequestError) as captured:
        ExtractionPipeline(parser=parser).extract_pages("server-error", [page])

    assert captured.value.status_code == 503
    assert captured.value.response["failure_kind"] == "assistant_content_retry_exhausted"
    assert len(transport.payloads) == 2


def test_non_nim_parse_error_does_not_use_native_fallback() -> None:
    class BrokenParser:
        config = NIMEndpointConfig(base_url="http://parse.test", model="parse-test")

        def parse_page(self, *args, **kwargs):
            raise ValueError("malformed parse payload")

    page = PageInput(
        page_number=1,
        native_text=MEANINGFUL_NATIVE_TEXT,
        image_bytes=b"rendered-page",
        is_multi_column=True,
    )

    with pytest.raises(ValueError, match="malformed parse payload"):
        ExtractionPipeline(parser=BrokenParser()).extract_pages("parse-error", [page])  # type: ignore[arg-type]


def test_pdf_page_source_records_vector_content_for_blank_page_guard(tmp_path) -> None:
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "blank-and-source-objects.pdf"
    document = fitz.open()
    document.new_page()
    vector_page = document.new_page()
    vector_page.draw_rect(fitz.Rect(20, 20, 80, 80), color=(0, 0, 0))
    annotation_page = document.new_page()
    annotation_page.add_text_annot(fitz.Point(50, 50), "review note")
    link_page = document.new_page()
    link_page.insert_link(
        {
            "kind": fitz.LINK_URI,
            "from": fitz.Rect(20, 20, 80, 40),
            "uri": "https://example.com",
        }
    )
    document.save(pdf_path)
    document.close()

    pages = list(PyMuPDFPageSource(pdf_path).iter_pages())

    assert pages[0].metadata["drawing_count"] == 0
    assert pages[0].metadata["embedded_image_count"] == 0
    assert pages[0].metadata["annotation_count"] == 0
    assert pages[0].metadata["link_count"] == 0
    assert pages[0].metadata["source_object_inventory_complete"] is True
    assert pages[1].metadata["drawing_count"] > 0
    assert pages[2].metadata["annotation_count"] > 0
    assert pages[3].metadata["link_count"] > 0
    assert ExtractionPipeline._is_deterministically_blank(pages[0]) is True
    assert ExtractionPipeline._is_deterministically_blank(pages[1]) is False
    assert ExtractionPipeline._is_deterministically_blank(pages[2]) is False
    assert ExtractionPipeline._is_deterministically_blank(pages[3]) is False


def checkpoint_page() -> PageExtraction:
    return PageExtraction(
        document_id="document-1",
        page_number=2,
        fingerprint="fingerprint-1",
        routing=PageRouter().decide(""),
        elements=(),
    )


def test_memory_checkpoint_rejects_payload_identity_mismatch() -> None:
    checkpoint = MemoryCheckpointStore()
    page = checkpoint_page()
    key = (page.document_id, page.page_number)

    for field, invalid_value in (("document_id", "other-document"), ("page_number", 99)):
        checkpoint.save_page(page)
        checkpoint._pages[key][field] = invalid_value
        assert checkpoint.load_page(*key, page.fingerprint) is None


def test_json_checkpoint_rejects_payload_identity_mismatch(tmp_path) -> None:
    checkpoint = JsonPageCheckpointStore(tmp_path / "checkpoints")
    page = checkpoint_page()
    path = checkpoint._page_path(page.document_id, page.page_number)

    for field, invalid_value in (("document_id", "other-document"), ("page_number", 99)):
        checkpoint.save_page(page)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[field] = invalid_value
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert (
            checkpoint.load_page(page.document_id, page.page_number, page.fingerprint)
            is None
        )
