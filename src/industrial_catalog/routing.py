"""Deterministic page routing for industrial catalog extraction.

The router deliberately depends only on cheap page signals.  Model calls live in
``extraction.py`` so routing can be unit-tested, benchmarked, and reproduced
without a GPU or network connection.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

_WORD_RE = re.compile(r"\b[\w][\w./+-]*\b", re.UNICODE)
_IDENTIFIER_RE = re.compile(
    r"(?<!\w)(?=[A-Z0-9][A-Z0-9._/-]{3,}\b)(?=[A-Z0-9._/-]*[A-Z])"
    r"(?=[A-Z0-9._/-]*\d)[A-Z0-9]+(?:[._/-][A-Z0-9]+)+(?!\w)",
    re.IGNORECASE,
)


class PageRoute(StrEnum):
    """Primary extraction path selected for a page."""

    NATIVE_TEXT = "native_text"
    DOCUMENT_VLM = "document_vlm"


# Backwards-friendly semantic alias for callers that prefer this name.
ExtractionRoute = PageRoute


@dataclass(frozen=True, slots=True)
class PageSignals:
    """Cheap, serializable facts used by :class:`PageRouter`.

    Ratios are always in the inclusive range ``[0, 1]``.  ``image_coverage``
    should represent the approximate fraction of the page occupied by raster
    images when that information is available.
    """

    character_count: int
    word_count: int
    line_count: int
    printable_ratio: float
    alphanumeric_ratio: float
    replacement_character_ratio: float
    median_line_length: float
    image_coverage: float = 0.0
    table_count: int = 0
    is_multi_column: bool = False
    identifier_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Thresholds for deterministic native-text versus VLM routing."""

    min_native_characters: int = 80
    min_native_words: int = 12
    min_printable_ratio: float = 0.95
    min_alphanumeric_ratio: float = 0.35
    max_replacement_character_ratio: float = 0.01
    complex_image_coverage: float = 0.45
    use_vlm_for_tables: bool = True
    use_vlm_for_multi_column: bool = True
    use_vlm_for_image_heavy_pages: bool = True
    verify_vlm_with_ocr: bool = True
    verify_native_identifiers_with_ocr: bool = False

    def __post_init__(self) -> None:
        if self.min_native_characters < 0 or self.min_native_words < 0:
            raise ValueError("minimum text thresholds must be non-negative")
        for name in (
            "min_printable_ratio",
            "min_alphanumeric_ratio",
            "max_replacement_character_ratio",
            "complex_image_coverage",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """An auditable page-routing result."""

    route: PageRoute
    requires_ocr_verification: bool
    reasons: tuple[str, ...]
    signals: PageSignals

    @property
    def requires_vlm(self) -> bool:
        return self.route is PageRoute.DOCUMENT_VLM

    @property
    def use_native_text(self) -> bool:
        return self.route is PageRoute.NATIVE_TEXT

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.value,
            "requires_ocr_verification": self.requires_ocr_verification,
            "reasons": list(self.reasons),
            "signals": self.signals.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RoutingDecision:
        return cls(
            route=PageRoute(str(value["route"])),
            requires_ocr_verification=bool(value["requires_ocr_verification"]),
            reasons=tuple(str(item) for item in value.get("reasons", ())),
            signals=PageSignals(**dict(value["signals"])),
        )


def _bounded_ratio(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
    if denominator <= 0:
        return empty
    return min(1.0, max(0.0, numerator / denominator))


def signals_from_text(
    native_text: str | None,
    *,
    image_coverage: float = 0.0,
    table_count: int = 0,
    is_multi_column: bool = False,
) -> PageSignals:
    """Compute stable routing signals from PDF-native text and page metadata."""

    text = native_text or ""
    non_whitespace = [character for character in text if not character.isspace()]
    meaningful_count = len(non_whitespace)
    printable_count = sum(
        character.isprintable() or character in "\n\r\t" for character in text
    )
    alphanumeric_count = sum(character.isalnum() for character in non_whitespace)
    replacement_count = sum(character in {"\ufffd", "\x00"} for character in text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_lengths = [len(line) for line in lines]

    coverage = float(image_coverage)
    if not math.isfinite(coverage):
        coverage = 0.0

    return PageSignals(
        character_count=meaningful_count,
        word_count=len(_WORD_RE.findall(text)),
        line_count=len(lines),
        printable_ratio=_bounded_ratio(printable_count, len(text)),
        alphanumeric_ratio=_bounded_ratio(
            alphanumeric_count, meaningful_count, empty=0.0
        ),
        replacement_character_ratio=_bounded_ratio(
            replacement_count, max(meaningful_count, 1), empty=0.0
        ),
        median_line_length=float(statistics.median(line_lengths)) if line_lengths else 0.0,
        image_coverage=min(1.0, max(0.0, coverage)),
        table_count=max(0, int(table_count)),
        is_multi_column=bool(is_multi_column),
        identifier_count=len(_IDENTIFIER_RE.findall(text)),
    )


class PageRouter:
    """Select a primary route and optional OCR verification for each page.

    Reasons are evaluated in a fixed order and are part of the checkpoint
    fingerprint, making equivalent inputs reproducible across processes.
    """

    def __init__(self, config: RoutingConfig | None = None) -> None:
        self.config = config or RoutingConfig()

    def decide(
        self,
        native_text: str | None = None,
        *,
        signals: PageSignals | None = None,
        image_coverage: float = 0.0,
        table_count: int = 0,
        is_multi_column: bool = False,
    ) -> RoutingDecision:
        if signals is None:
            signals = signals_from_text(
                native_text,
                image_coverage=image_coverage,
                table_count=table_count,
                is_multi_column=is_multi_column,
            )

        config = self.config
        reasons: list[str] = []

        if signals.character_count < config.min_native_characters:
            reasons.append("insufficient_native_characters")
        if signals.word_count < config.min_native_words:
            reasons.append("insufficient_native_words")
        if signals.printable_ratio < config.min_printable_ratio:
            reasons.append("low_printable_ratio")
        if signals.alphanumeric_ratio < config.min_alphanumeric_ratio:
            reasons.append("low_alphanumeric_ratio")
        if (
            signals.replacement_character_ratio
            > config.max_replacement_character_ratio
        ):
            reasons.append("corrupt_or_replacement_glyphs")
        if config.use_vlm_for_tables and signals.table_count > 0:
            reasons.append("table_layout")
        if config.use_vlm_for_multi_column and signals.is_multi_column:
            reasons.append("multi_column_layout")
        if (
            config.use_vlm_for_image_heavy_pages
            and signals.image_coverage >= config.complex_image_coverage
        ):
            reasons.append("image_heavy_layout")

        route = PageRoute.DOCUMENT_VLM if reasons else PageRoute.NATIVE_TEXT
        requires_ocr = (
            route is PageRoute.DOCUMENT_VLM and config.verify_vlm_with_ocr
        ) or (
            route is PageRoute.NATIVE_TEXT
            and config.verify_native_identifiers_with_ocr
            and signals.identifier_count > 0
        )

        if not reasons:
            reasons.append("native_text_quality_passed")
        if requires_ocr:
            reasons.append("ocr_identifier_verification")

        return RoutingDecision(
            route=route,
            requires_ocr_verification=requires_ocr,
            reasons=tuple(reasons),
            signals=signals,
        )

    # ``route`` is a convenient synonym for callers and notebooks.
    route = decide


def route_page(
    native_text: str | None,
    *,
    config: RoutingConfig | None = None,
    image_coverage: float = 0.0,
    table_count: int = 0,
    is_multi_column: bool = False,
) -> RoutingDecision:
    """Functional wrapper around :class:`PageRouter`."""

    return PageRouter(config).decide(
        native_text,
        image_coverage=image_coverage,
        table_count=table_count,
        is_multi_column=is_multi_column,
    )


__all__ = [
    "ExtractionRoute",
    "PageRoute",
    "PageRouter",
    "PageSignals",
    "RoutingConfig",
    "RoutingDecision",
    "route_page",
    "signals_from_text",
]
