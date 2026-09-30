from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from industrial_catalog.extraction import (
    DocumentExtraction,
    ExtractedElement,
    ExtractionPipeline,
    PageExtraction,
    _table_rows,
    derive_structured_source_hints,
    manufacturer_candidates_from_extraction,
    product_cardinality_from_extraction,
)
from industrial_catalog.models import product_batch_guided_json_schema
from industrial_catalog.nvidia_clients import (
    GuidedJsonClient,
    GuidedJsonMode,
    NIMEndpointConfig,
    RetryPolicy,
    TransportResponse,
)
from industrial_catalog.routing import route_page
from industrial_catalog.runner import (
    enforce_structured_source_hints,
    product_batch_schema_for_hints,
)
from industrial_catalog.validation import prepare_product_payload

NEMOTRON_LATEX_UPC_TABLE = r"""\begin{tabular}{cccccccccc}
\multicolumn{10}{c}{**Stainless Steel Hub UPC Part Numbers (Prefix is 685144)**}\\
**Size** & **1/4*** & **3/8*** & **1/2*** & **5/8*** & **3/4*** & **7/8*** &
**1*** & **1-1/8*** & **1-1/4***\\
SS075 & 70001 & 70002 & 70003 & 70004 & 70005 & 70006 & 70007 & 70008 & 70009\\
SS095 & 70101 & 70102 & 70103 & 70104 & 70105 & 70106 & 70107 & 70108 & 70109\\
SS100 & 70201 & 70202 & 70203 & 70204 & 70205 & 70206 & 70207 & 70208 & 70209\\
SS110 & 70301 & 70302 & 70303 & 70304 & 70305 & 70306 & 70307 & 70308 & 70309\\
SS150 & 70401 & 70402 & 70403 & 70404 & 70405 & 70406 & 70407 & 70408 & 70409\\
\end{tabular}"""


def element(element_id: str, text: str, element_type: str = "text") -> ExtractedElement:
    return ExtractedElement(
        element_id=element_id,
        document_id="document-1",
        page_number=1,
        sequence_number=0,
        element_type=element_type,
        text=text,
        extraction_method="test",
    )


def document(*elements: ExtractedElement) -> DocumentExtraction:
    page = PageExtraction(
        document_id="document-1",
        page_number=1,
        fingerprint="page-1",
        routing=route_page("Industrial catalog source text " * 20),
        elements=tuple(elements),
    )
    return DocumentExtraction(
        document_id="document-1",
        source_path="catalog.pdf",
        pages=(page,),
    )


def test_legal_footer_candidate_outranks_material_title() -> None:
    extraction = document(
        element("el_title", "Stainless Steel Jaw Couplings", "heading"),
        element(
            "el_footer",
            "\u00a9 2025 Lovejoy, LLC | All Rights Reserved",
            "footer",
        ),
    )

    candidates = manufacturer_candidates_from_extraction(extraction)

    assert [(item.raw, item.element_id, item.basis) for item in candidates] == [
        ("Lovejoy, LLC", "el_footer", "copyright")
    ]
    assert all(item.raw != "Stainless Steel" for item in candidates)


def test_labeled_and_international_legal_company_names_are_detected() -> None:
    extraction = document(
        element("el_maker", "Manufactured by Rockwell Automation, Inc.", "paragraph"),
        element("el_legal", "ifm electronic GmbH", "footer"),
    )

    candidates = manufacturer_candidates_from_extraction(extraction)

    assert [item.raw for item in candidates] == [
        "Rockwell Automation, Inc.",
        "ifm electronic GmbH",
    ]
    assert candidates[0].basis == "manufacturer_label"


def test_trademarked_page_header_yields_canonical_brand_candidate() -> None:
    extraction = document(
        element("el_panduit_header", "# PANDUIT<sup>®</sup>", "Page-header")
    )

    candidates = manufacturer_candidates_from_extraction(extraction)

    assert [(item.raw, item.element_id, item.basis) for item in candidates] == [
        ("PANDUIT", "el_panduit_header", "trademark_header")
    ]


def test_trademark_header_strips_markdown_and_html_entity_markup() -> None:
    extraction = document(
        element(
            "el_brand_heading",
            "## **Rockwell Automation<sup>&trade;</sup>**",
            "heading",
        )
    )

    candidates = manufacturer_candidates_from_extraction(extraction)

    assert [item.raw for item in candidates] == ["Rockwell Automation"]
    assert candidates[0].basis == "trademark_header"


def test_trademark_header_rule_rejects_generic_or_non_standalone_titles() -> None:
    rejected_headers = (
        ("# PANDUIT", "Page-header"),
        ("# Product Catalog<sup>®</sup>", "Page-header"),
        ("# Stainless Steel Jaw Couplings™", "heading"),
        ("# PANDUIT® Product Catalog", "Page-header"),
        ("PANDUIT<sup>®</sup>", "paragraph"),
    )

    for index, (text, element_type) in enumerate(rejected_headers):
        extraction = document(element(f"el_rejected_{index}", text, element_type))
        assert manufacturer_candidates_from_extraction(extraction) == ()


def test_codes_in_prose_do_not_unlock_multiple_products() -> None:
    extraction = document(
        element(
            "el_body",
            "The L090 and L095 codes are mentioned in descriptive prose.",
            "paragraph",
        )
    )

    decision = product_cardinality_from_extraction(extraction)

    assert decision.mode == "single_family"
    assert decision.maximum_products == 1
    assert decision.identifiers == ()


def test_explicit_part_number_table_rows_unlock_multiple_products() -> None:
    extraction = document(
        element(
            "el_table",
            """
| Part Number | Bore | Material |
| --- | --- | --- |
| L090-100 | 10 mm | Stainless Steel |
| L095-200 | 12 mm | Stainless Steel |
""",
            "table",
        )
    )

    decision = product_cardinality_from_extraction(extraction)

    assert decision.mode == "multi_product_rows"
    assert decision.maximum_products is None
    assert decision.identifiers == ("L090-100", "L095-200")
    assert decision.evidence_element_ids == ("el_table",)


def test_nemotron_latex_table_decoder_cleans_live_upc_shape() -> None:
    rows = _table_rows(NEMOTRON_LATEX_UPC_TABLE, allow_plain_columns=True)

    assert rows[0] == [
        "Stainless Steel Hub UPC Part Numbers (Prefix is 685144)"
    ]
    assert rows[1] == [
        "Size",
        "1/4",
        "3/8",
        "1/2",
        "5/8",
        "3/4",
        "7/8",
        "1",
        "1-1/8",
        "1-1/4",
    ]
    assert [row[0] for row in rows[2:]] == [
        "SS075",
        "SS095",
        "SS100",
        "SS110",
        "SS150",
    ]


def test_code_like_size_rows_in_nemotron_latex_unlock_multiple_products() -> None:
    extraction = document(element("el_latex_table", NEMOTRON_LATEX_UPC_TABLE, "table"))

    hints = derive_structured_source_hints(extraction)
    decision = hints.cardinality
    schema = product_batch_schema_for_hints(hints)

    assert decision.mode == "multi_product_rows"
    assert decision.identifiers == ("SS075", "SS095", "SS100", "SS110", "SS150")
    assert decision.evidence_element_ids == ("el_latex_table",)
    assert schema["properties"]["products"]["minItems"] == 5
    assert schema["properties"]["products"]["maxItems"] == 5
    product_schema = schema["properties"]["products"]["items"]
    assert set(product_schema["properties"]) == {
        "manufacturer",
        "part_name",
        "part_number",
    }
    assert product_schema["required"] == [
        "manufacturer",
        "part_name",
        "part_number",
    ]
    assert product_schema["properties"]["part_number"]["properties"]["raw"][
        "enum"
    ] == ["SS075", "SS095", "SS100", "SS110", "SS150"]


def test_explicit_part_numbers_in_latex_preserve_hard_identifiers() -> None:
    extraction = document(
        element(
            "el_latex_parts",
            r"""\begin{tabular}{lll}
\multicolumn{3}{c}{\textbf{Jaw \& Hub Assemblies}}\\
\textbf{Part Number} & \textbf{Bore} & \textbf{Material}\\
SS-075/A & \frac{1}{4} & Stainless Steel\\
SS_095-B & \frac{1}{2} & Stainless Steel\\
\end{tabular}""",
            "table",
        )
    )

    decision = product_cardinality_from_extraction(extraction)

    assert decision.mode == "multi_product_rows"
    assert decision.identifiers == ("SS-075/A", "SS_095-B")


def test_natural_language_or_dimension_size_rows_do_not_unlock_products() -> None:
    extraction = document(
        element(
            "el_sizes",
            r"""\begin{tabular}{ll}
Size & Description\\
Small & Compact coupling\\
Large & High capacity coupling\\
\end{tabular}
\begin{tabular}{ll}
Size & Description\\
10mm & First bore\\
12mm & Second bore\\
\end{tabular}""",
            "table",
        )
    )

    decision = product_cardinality_from_extraction(extraction)

    assert decision.mode == "single_family"
    assert decision.identifiers == ()


def test_single_family_schema_caps_products_and_removes_part_number() -> None:
    extraction = document(
        element("el_footer", "Lovejoy, LLC", "footer"),
        element("el_title", "Stainless Steel Jaw Couplings", "heading"),
    )
    hints = derive_structured_source_hints(extraction)

    schema = product_batch_schema_for_hints(hints)
    products = schema["properties"]["products"]
    product_properties = products["items"]["properties"]

    assert products["maxItems"] == 1
    assert "part_number" not in product_properties
    manufacturer = product_properties["manufacturer"]
    assert manufacturer["properties"]["raw"]["enum"] == ["Lovejoy, LLC"]
    assert manufacturer["properties"]["evidence"]["items"]["properties"][
        "element_id"
    ]["enum"] == ["el_footer"]


def test_multi_product_table_schema_keeps_part_number_and_caps_to_rows() -> None:
    extraction = document(
        element("el_footer", "Lovejoy, LLC", "footer"),
        element(
            "el_table",
            "| Model Number | Description |\n| --- | --- |\n| L090 | Jaw |\n| L095 | Jaw |",
            "table",
        ),
    )
    hints = derive_structured_source_hints(extraction)

    schema = product_batch_schema_for_hints(hints)
    products = schema["properties"]["products"]
    product_schema = products["items"]
    part_number = product_schema["properties"]["part_number"]

    assert products["minItems"] == 2
    assert products["maxItems"] == 2
    assert set(product_schema["properties"]) == {
        "manufacturer",
        "part_name",
        "part_number",
    }
    assert product_schema["required"] == [
        "manufacturer",
        "part_name",
        "part_number",
    ]
    assert part_number["properties"]["raw"]["enum"] == ["L090", "L095"]
    assert part_number["properties"]["evidence"]["items"]["properties"][
        "element_id"
    ]["enum"] == ["el_table"]


def test_multi_product_prompt_requires_one_record_per_exact_identifier() -> None:
    hints = derive_structured_source_hints(
        document(
            element(
                "el_table",
                "| Part Number | Description |\n| --- | --- |\n| L090 | Jaw |\n| L095 | Jaw |",
                "table",
            )
        )
    )

    prompt = hints.prompt_text()

    assert '"identifiers":["L090","L095"]' in prompt
    assert "Emit EXACTLY one product for EACH identifier" in prompt
    assert "Cover every listed identifier once" in prompt
    assert "no missing, duplicate, or additional product records" in prompt
    assert "lean identity-only row pass" in prompt
    assert "do not copy table bodies into specifications" in prompt


def test_identity_only_row_payload_hydrates_empty_enrichment_collections() -> None:
    extraction = document(
        element("el_footer", "Lovejoy, LLC", "footer"),
        element("el_title", "Stainless Steel Jaw Coupling", "heading"),
        element(
            "el_table",
            "| Part Number | Description |\n| --- | --- |\n| L090 | Jaw |\n| L095 | Jaw |",
            "table",
        ),
    )
    raw = {
        "manufacturer": {
            "raw": "Lovejoy, LLC",
            "evidence": [{"element_id": "el_footer"}],
        },
        "part_name": {
            "raw": "Stainless Steel Jaw Coupling",
            "evidence": [{"element_id": "el_title"}],
        },
        "part_number": {
            "raw": "L090",
            "evidence": [{"element_id": "el_table"}],
        },
    }

    prepared = prepare_product_payload(
        raw,
        ordinal=0,
        extracted_elements=extraction,
        source_document_id="document-1",
        source_path="catalog.pdf",
    )

    assert prepared.accepted
    assert prepared.product is not None
    assert prepared.product.specifications == []
    assert prepared.product.other_details == {}


class CaptureTransport:
    def __init__(self) -> None:
        self.payload = None

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.payload = payload
        return TransportResponse(
            200,
            {
                "id": "response-1",
                "model": "nemotron-test",
                "choices": [
                    {
                        "message": {"content": '{"products": []}'},
                        "finish_reason": "stop",
                    }
                ],
            },
        )


def test_source_backed_hints_are_prominent_in_structured_prompt() -> None:
    extraction = document(
        element("el_title", "Stainless Steel Jaw Couplings", "heading"),
        element("el_footer", "Lovejoy, LLC", "footer"),
    )
    transport = CaptureTransport()
    llm = GuidedJsonClient(
        NIMEndpointConfig(
            base_url="http://llm.test",
            model="nemotron-test",
            retry=RetryPolicy(max_attempts=1),
        ),
        mode=GuidedJsonMode.OPENAI_RESPONSE_FORMAT,
        transport=transport,
    )

    ExtractionPipeline(llm=llm).parse_structured(
        extraction,
        schema=product_batch_guided_json_schema(),
    )

    assert transport.payload is not None
    prompt = transport.payload["messages"][1]["content"]
    assert (
        'MANUFACTURER_CANDIDATES=[{"basis":"legal_suffix",'
        '"element_id":"el_footer","raw":"Lovejoy, LLC"}]'
    ) in prompt
    assert "Every emitted product MUST use manufacturer.raw exactly" in prompt
    assert "Materials, finishes, product adjectives, and title words" in prompt
    assert '"maximum_products":1' in prompt
    assert "Emit at most ONE product" in prompt


def parsed_product(
    manufacturer: str,
    *,
    manufacturer_element_id: str = "el_footer",
    part_name: str = "Stainless Steel Jaw Couplings",
    part_name_element_id: str = "el_title",
    part_number: str | None = None,
    part_number_element_id: str = "el_body",
) -> dict:
    product = {
        "manufacturer": {
            "raw": manufacturer,
            "evidence": [{"element_id": manufacturer_element_id}],
        },
        "part_name": {
            "raw": part_name,
            "evidence": [{"element_id": part_name_element_id}],
        },
    }
    if part_number is not None:
        product["part_number"] = {
            "raw": part_number,
            "evidence": [{"element_id": part_number_element_id}],
        }
    return product


PANDUIT_TABLE = """| Part Number | Description |
| --- | --- |
| CR2-M | Cable tie mount |
| CR4H-M | Cable tie mount |
| CR4H-M0 | Cable tie mount |
| CR4H-M30 | Cable tie mount |"""
PANDUIT_EXPECTED_IDENTIFIERS = ["CR2-M", "CR4H-M", "CR4H-M0", "CR4H-M30"]
PANDUIT_EMITTED_IDENTIFIERS = ["CR2-M", "CR2-M", "CR4H-M", "CR4H-M0"]
PANDUIT_LIVE_TITLE = "CR2 & CR4H Closed Connector Rings"
PANDUIT_LIVE_DESCRIPTION = (
    "The Panduit CR2 and CR4H closed connector rings organize and protect "
    "conductors in demanding industrial installations while providing durable "
    "performance, simple application, and compatibility with the listed cable "
    "sizes, materials, environmental conditions, installation tools, packaging "
    "quantities, and ordering options; select the required exact part number "
    "from the table below for the intended application."
)


def panduit_multirow_extraction(
    *extra_elements: ExtractedElement,
) -> DocumentExtraction:
    return document(
        element("el_footer", "Panduit Corp.", "footer"),
        element("el_title", "Cable Tie Mounts", "heading"),
        element("el_table", PANDUIT_TABLE, "table"),
        *extra_elements,
    )


def panduit_duplicate_payload() -> dict:
    return {
        "products": [
            parsed_product(
                "Panduit Corp.",
                part_name="Cable Tie Mounts",
                part_number=identifier,
                part_number_element_id="el_table",
            )
            for identifier in PANDUIT_EMITTED_IDENTIFIERS
        ]
    }


def test_source_guard_repairs_exact_source_backed_identity_row_permutation() -> None:
    extraction = panduit_multirow_extraction()
    hints = derive_structured_source_hints(extraction)
    payload = panduit_duplicate_payload()
    original = deepcopy(payload)

    guarded, audit = enforce_structured_source_hints(
        payload,
        hints,
        extraction=extraction,
    )

    assert payload == original
    assert [
        product["part_number"]["raw"] for product in guarded["products"]
    ] == PANDUIT_EXPECTED_IDENTIFIERS
    assert all(
        product["part_name"]["raw"] == "Cable Tie Mounts"
        for product in guarded["products"]
    )
    assert audit["identity_row_repair"]["part_name_selection_mode"] == (
        "identical_across_emitted_rows"
    )
    assert audit["identity_row_repair"]["part_name_source_element_id"] is None


def test_source_guard_repair_selects_unique_live_standalone_product_title() -> None:
    extraction = document(
        element("el_manufacturer", "Manufacturer: PANDUIT", "footer"),
        element("el_live_title", PANDUIT_LIVE_TITLE, "heading"),
        element("el_live_description", PANDUIT_LIVE_DESCRIPTION, "text"),
        element("el_table", PANDUIT_TABLE, "table"),
    )
    hints = derive_structured_source_hints(extraction)
    products = [
        parsed_product(
            "PANDUIT",
            manufacturer_element_id="el_manufacturer",
            part_name=(
                PANDUIT_LIVE_TITLE if index == 0 else PANDUIT_LIVE_DESCRIPTION
            ),
            part_name_element_id=(
                "el_live_title" if index == 0 else "el_live_description"
            ),
            part_number=identifier,
            part_number_element_id="el_table",
        )
        for index, identifier in enumerate(PANDUIT_EMITTED_IDENTIFIERS)
    ]

    guarded, audit = enforce_structured_source_hints(
        {"products": products},
        hints,
        extraction=extraction,
    )

    assert [
        product["part_number"]["raw"] for product in guarded["products"]
    ] == PANDUIT_EXPECTED_IDENTIFIERS
    assert all(
        product["part_name"]
        == {
            "raw": PANDUIT_LIVE_TITLE,
            "evidence": [{"element_id": "el_live_title"}],
        }
        for product in guarded["products"]
    )
    repair = audit["identity_row_repair"]
    assert repair["applied"] is True
    assert repair["part_name_raw"] == PANDUIT_LIVE_TITLE
    assert repair["part_name_selection_mode"] == (
        "unique_concise_standalone_source_title"
    )
    assert repair["part_name_source_element_id"] == "el_live_title"
    assert set(repair["part_name_values_seen"]) == {
        PANDUIT_LIVE_TITLE,
        PANDUIT_LIVE_DESCRIPTION,
    }
    assert all(
        set(product) == {"manufacturer", "part_name", "part_number"}
        for product in guarded["products"]
    )
    assert all(
        product["part_number"]["evidence"] == [{"element_id": "el_table"}]
        for product in guarded["products"]
    )
    assert audit["dropped_product_count"] == 0
    assert audit["cardinality_violation"]["resolved_by"] == (
        "source_backed_identity_row_repair"
    )
    repair = audit["identity_row_repair"]
    assert repair["applied"] is True
    assert repair["expected_identifiers"] == PANDUIT_EXPECTED_IDENTIFIERS
    assert repair["emitted_identifiers"] == PANDUIT_EMITTED_IDENTIFIERS
    assert repair["source_element_by_identifier"] == {
        identifier: "el_table" for identifier in PANDUIT_EXPECTED_IDENTIFIERS
    }
    assert audit["actions"][0]["action"] == (
        "repaired_multi_product_identifier_set"
    )
    assert [action["part_number_raw"] for action in audit["actions"][1:]] == (
        PANDUIT_EXPECTED_IDENTIFIERS
    )
    prepared = [
        prepare_product_payload(
            product,
            ordinal=index,
            extracted_elements=extraction,
            source_document_id="document-1",
            source_path="panduit.pdf",
        )
        for index, product in enumerate(guarded["products"])
    ]
    assert all(item.accepted for item in prepared)
    assert [
        item.product.part_number.raw
        for item in prepared
        if item.product is not None and item.product.part_number is not None
    ] == PANDUIT_EXPECTED_IDENTIFIERS


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("no_extraction", "document_extraction_not_available"),
        ("unexpected_part_number", "emitted_part_number_is_not_expected"),
        (
            "wrong_part_number_evidence",
            "part_number_evidence_is_not_from_cardinality_elements",
        ),
        ("inconsistent_manufacturer", "manufacturer_identity_is_not_consistent"),
        (
            "ambiguous_concise_part_names",
            "multiple_concise_standalone_part_name_candidates",
        ),
        (
            "generic_heading_part_name",
            "no_concise_standalone_part_name_candidate",
        ),
        (
            "no_concise_part_name",
            "no_concise_standalone_part_name_candidate",
        ),
        (
            "manufacturer_not_source_backed",
            "manufacturer_identity_is_not_exactly_source_backed",
        ),
        (
            "part_name_not_source_backed",
            "part_name_identity_is_not_exactly_source_backed",
        ),
        (
            "no_manufacturer_candidates",
            "manufacturer_is_not_a_deterministic_source_candidate",
        ),
        ("extra_product_field", "product_is_not_identity_only"),
        (
            "invalid_identity_wire_shape",
            "identity_field_does_not_match_strict_raw_evidence_shape",
        ),
        (
            "count_mismatch",
            "emitted_product_count_does_not_match_expected_count",
        ),
        (
            "prior_rejection",
            "products_were_rejected_before_identifier_repair",
        ),
        (
            "expected_identifier_not_in_source",
            "expected_identifier_not_exactly_source_backed",
        ),
        (
            "duplicate_expected_identifier",
            "expected_identifiers_not_nonempty_and_unique",
        ),
        (
            "missing_cardinality_evidence",
            "cardinality_evidence_element_not_found",
        ),
    ],
)
def test_source_guard_row_repair_remains_fail_closed_for_unsafe_cases(
    case: str,
    expected_reason: str,
) -> None:
    extraction = panduit_multirow_extraction()
    hints = derive_structured_source_hints(extraction)
    payload = panduit_duplicate_payload()
    enforcement_extraction: DocumentExtraction | None = extraction

    if case == "no_extraction":
        enforcement_extraction = None
    elif case == "unexpected_part_number":
        payload["products"][-1]["part_number"]["raw"] = "CR4H-UNKNOWN"
    elif case == "wrong_part_number_evidence":
        payload["products"][-1]["part_number"]["evidence"] = [
            {"element_id": "el_title"}
        ]
    elif case == "inconsistent_manufacturer":
        extraction = panduit_multirow_extraction(
            element("el_footer_alt", "Panduit Inc.", "footer")
        )
        hints = derive_structured_source_hints(extraction)
        enforcement_extraction = extraction
        payload["products"][-1]["manufacturer"] = {
            "raw": "Panduit Inc.",
            "evidence": [{"element_id": "el_footer_alt"}],
        }
    elif case == "ambiguous_concise_part_names":
        extraction = panduit_multirow_extraction(
            element("el_title_alt", "Cable Tie Accessories", "heading")
        )
        hints = derive_structured_source_hints(extraction)
        enforcement_extraction = extraction
        payload["products"][-1]["part_name"] = {
            "raw": "Cable Tie Accessories",
            "evidence": [{"element_id": "el_title_alt"}],
        }
    elif case == "generic_heading_part_name":
        extraction = panduit_multirow_extraction(
            element("el_generic_heading", "Product Catalog", "heading"),
            element("el_description", PANDUIT_LIVE_DESCRIPTION, "text"),
        )
        hints = derive_structured_source_hints(extraction)
        enforcement_extraction = extraction
        for index, product in enumerate(payload["products"]):
            product["part_name"] = {
                "raw": (
                    "Product Catalog" if index == 0 else PANDUIT_LIVE_DESCRIPTION
                ),
                "evidence": [
                    {
                        "element_id": (
                            "el_generic_heading"
                            if index == 0
                            else "el_description"
                        )
                    }
                ],
            }
    elif case == "no_concise_part_name":
        sentence = "Closed connector ring ordering choices are listed below."
        extraction = panduit_multirow_extraction(
            element("el_sentence", sentence, "text"),
            element("el_description", PANDUIT_LIVE_DESCRIPTION, "text"),
        )
        hints = derive_structured_source_hints(extraction)
        enforcement_extraction = extraction
        for index, product in enumerate(payload["products"]):
            product["part_name"] = {
                "raw": sentence if index == 0 else PANDUIT_LIVE_DESCRIPTION,
                "evidence": [
                    {
                        "element_id": (
                            "el_sentence" if index == 0 else "el_description"
                        )
                    }
                ],
            }
    elif case == "manufacturer_not_source_backed":
        enforcement_extraction = document(
            element("el_footer", "Different Corp.", "footer"),
            element("el_title", "Cable Tie Mounts", "heading"),
            element("el_table", PANDUIT_TABLE, "table"),
        )
    elif case == "part_name_not_source_backed":
        for product in payload["products"]:
            product["part_name"]["raw"] = "Invented Mount Name"
    elif case == "no_manufacturer_candidates":
        hints = replace(hints, manufacturer_candidates=())
    elif case == "extra_product_field":
        payload["products"][0]["description"] = {
            "raw": "Cable tie mount",
            "evidence": [{"element_id": "el_table"}],
        }
    elif case == "invalid_identity_wire_shape":
        payload["products"][0]["manufacturer"]["normalized"] = "PANDUIT"
    elif case == "count_mismatch":
        payload["products"].pop()
    elif case == "prior_rejection":
        payload["products"].append("not a product object")
    elif case == "expected_identifier_not_in_source":
        hints = replace(
            hints,
            cardinality=replace(
                hints.cardinality,
                identifiers=("CR2-M", "CR4H-M", "CR4H-M0", "CR9-M"),
            ),
        )
    elif case == "duplicate_expected_identifier":
        hints = replace(
            hints,
            cardinality=replace(
                hints.cardinality,
                identifiers=("CR2-M", "CR4H-M", "CR4H-M0", "CR4H-M0"),
            ),
        )
    elif case == "missing_cardinality_evidence":
        hints = replace(
            hints,
            cardinality=replace(
                hints.cardinality,
                evidence_element_ids=("el_missing_table",),
            ),
        )
    else:
        raise AssertionError(f"unhandled unsafe repair case: {case}")

    original = deepcopy(payload)
    guarded, audit = enforce_structured_source_hints(
        payload,
        hints,
        extraction=enforcement_extraction,
    )

    assert payload == original
    assert guarded["products"] == []
    assert audit["dropped_product_count"] == len(payload["products"])
    repair = audit["identity_row_repair"]
    assert repair["attempted"] is True
    assert repair["applied"] is False
    assert repair["failed_precondition"] == expected_reason
    assert not any(
        action["action"] == "repaired_multi_product_identifier_set"
        for action in audit["actions"]
    )


def test_source_guard_keeps_exact_candidate_and_removes_unsupported_part_number() -> None:
    hints = derive_structured_source_hints(
        document(
            element("el_title", "Stainless Steel Jaw Couplings", "heading"),
            element("el_footer", "Lovejoy, LLC", "footer"),
        )
    )
    payload = {
        "products": [
            parsed_product("Lovejoy, LLC", part_number="generic-prose-code")
        ]
    }

    guarded, audit = enforce_structured_source_hints(payload, hints)

    assert len(guarded["products"]) == 1
    assert "part_number" not in guarded["products"][0]
    assert audit["dropped_product_count"] == 0
    assert audit["actions"][0]["action"] == "removed_unsupported_part_number"
    assert "part_number" in payload["products"][0]


def test_source_guard_drops_material_claimed_as_manufacturer() -> None:
    hints = derive_structured_source_hints(
        document(
            element("el_title", "Stainless Steel Jaw Couplings", "heading"),
            element("el_footer", "Lovejoy, LLC", "footer"),
        )
    )

    guarded, audit = enforce_structured_source_hints(
        {"products": [parsed_product("Stainless Steel")]},
        hints,
    )

    assert guarded == {"products": []}
    assert audit["dropped_product_count"] == 1
    assert audit["dropped_products"][0]["reason"] == (
        "manufacturer_not_in_source_candidates"
    )


def test_source_guard_requires_candidate_evidence_element_id() -> None:
    hints = derive_structured_source_hints(
        document(element("el_footer", "Lovejoy, LLC", "footer"))
    )

    guarded, audit = enforce_structured_source_hints(
        {
            "products": [
                parsed_product(
                    "Lovejoy, LLC",
                    manufacturer_element_id="el_unrelated",
                )
            ]
        },
        hints,
    )

    assert guarded["products"] == []
    assert audit["dropped_products"][0]["reason"] == (
        "manufacturer_candidate_evidence_not_cited"
    )


def test_source_guard_rejects_ambiguous_excess_single_family_products() -> None:
    hints = derive_structured_source_hints(
        document(element("el_footer", "Lovejoy, LLC", "footer"))
    )

    guarded, audit = enforce_structured_source_hints(
        {
            "products": [
                parsed_product("Lovejoy, LLC", part_name="Jaw Coupling Family"),
                parsed_product("Lovejoy, LLC", part_name="Jaw Coupling Insert"),
            ]
        },
        hints,
    )

    assert guarded["products"] == []
    assert audit["dropped_product_count"] == 2
    assert audit["cardinality_violation"]["maximum_products"] == 1


def test_source_guard_retains_explicit_multi_product_table_rows() -> None:
    hints = derive_structured_source_hints(
        document(
            element("el_footer", "Lovejoy, LLC", "footer"),
            element(
                "el_table",
                "| Part Number | Description |\n| --- | --- |\n| L090 | Jaw |\n| L095 | Jaw |",
                "table",
            ),
        )
    )

    guarded, audit = enforce_structured_source_hints(
        {
            "products": [
                parsed_product("Lovejoy, LLC", part_number="L090"),
                parsed_product("Lovejoy, LLC", part_number="L095"),
            ]
        },
        hints,
    )

    assert [
        product["part_number"]["raw"] for product in guarded["products"]
    ] == ["L090", "L095"]
    assert audit["final_product_count"] == 2
    assert audit["dropped_product_count"] == 0


def test_source_guard_rejects_collapsed_multi_product_response() -> None:
    hints = derive_structured_source_hints(
        document(
            element("el_footer", "Lovejoy, LLC", "footer"),
            element(
                "el_table",
                "| Part Number | Description |\n| --- | --- |\n| L090 | Jaw |\n| L095 | Jaw |",
                "table",
            ),
        )
    )

    guarded, audit = enforce_structured_source_hints(
        {"products": [parsed_product("Lovejoy, LLC", part_number="L090")]},
        hints,
    )

    assert guarded["products"] == []
    assert audit["dropped_product_count"] == 1
    violation = audit["cardinality_violation"]
    assert violation["reason"] == "multi_product_identifier_set_mismatch"
    assert violation["expected_identifiers"] == ["L090", "L095"]
    assert violation["emitted_identifiers"] == ["L090"]
    assert violation["missing_identifiers"] == ["L095"]
    assert violation["duplicate_identifiers"] == []
    assert violation["unexpected_identifiers"] == []


def test_source_guard_rejects_duplicate_multi_product_identifiers() -> None:
    hints = derive_structured_source_hints(
        document(
            element("el_footer", "Lovejoy, LLC", "footer"),
            element(
                "el_table",
                "| Part Number | Description |\n| --- | --- |\n| L090 | Jaw |\n| L095 | Jaw |",
                "table",
            ),
        )
    )

    guarded, audit = enforce_structured_source_hints(
        {
            "products": [
                parsed_product("Lovejoy, LLC", part_number="L090"),
                parsed_product("Lovejoy, LLC", part_number="L090"),
            ]
        },
        hints,
    )

    assert guarded["products"] == []
    violation = audit["cardinality_violation"]
    assert violation["duplicate_identifiers"] == ["L090"]
    assert violation["missing_identifiers"] == ["L095"]


def test_source_guard_rejects_unexpected_or_missing_part_number() -> None:
    hints = derive_structured_source_hints(
        document(
            element("el_footer", "Lovejoy, LLC", "footer"),
            element(
                "el_table",
                "| Part Number | Description |\n| --- | --- |\n| L090 | Jaw |\n| L095 | Jaw |",
                "table",
            ),
        )
    )

    guarded, audit = enforce_structured_source_hints(
        {
            "products": [
                parsed_product("Lovejoy, LLC", part_number="UNLISTED"),
                parsed_product("Lovejoy, LLC"),
            ]
        },
        hints,
    )

    assert guarded["products"] == []
    violation = audit["cardinality_violation"]
    assert violation["emitted_identifiers"] == ["UNLISTED", None]
    assert violation["missing_identifiers"] == ["L090", "L095"]
    assert violation["unexpected_identifiers"] == ["UNLISTED"]
    assert violation["missing_part_number_product_indexes"] == [1]
