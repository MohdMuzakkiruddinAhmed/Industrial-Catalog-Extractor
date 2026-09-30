"""Generate a redistribution-safe synthetic industrial catalog for the dry-run demo."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pymupdf as fitz


def create_catalog(output: Path) -> Path:
    """Write a two-page synthetic catalog containing text and a simple data table."""

    output.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    try:
        page = document.new_page(width=612, height=792)
        page.insert_text((54, 64), "ACME INDUSTRIAL COMPONENTS", fontsize=18)
        page.insert_text((54, 96), "FlexCouple 200 Series", fontsize=15)
        page.insert_text((54, 126), "Part number: FC-200-SS-100", fontsize=11)
        page.insert_text((54, 148), "Material: Stainless steel", fontsize=11)
        page.insert_text((54, 170), "Bore diameter: 1.00 in", fontsize=11)
        page.insert_text((54, 192), "Maximum speed: 6000 rpm", fontsize=11)
        page.insert_text(
            (54, 232),
            "Synthetic research fixture. No real manufacturer or product is represented.",
            fontsize=9,
        )

        page = document.new_page(width=612, height=792)
        page.insert_text((54, 64), "ACME INDUSTRIAL COMPONENTS", fontsize=18)
        page.insert_text((54, 96), "FlexCouple 200 selection table", fontsize=14)
        rows = (
            "Part number        Bore       Material          Torque",
            "FC-200-SS-075     0.75 in    Stainless steel   85 N m",
            "FC-200-SS-100     1.00 in    Stainless steel   90 N m",
            "FC-200-AL-100     1.00 in    Aluminum           65 N m",
        )
        for ordinal, row in enumerate(rows):
            page.insert_text((54, 132 + ordinal * 24), row, fontsize=10, fontname="cour")
        page.insert_text(
            (54, 252),
            "All names, identifiers, and specifications on this page are fictional.",
            fontsize=9,
        )

        document.set_metadata(
            {
                "title": "Synthetic Industrial Catalog",
                "author": "Industrial Catalog Extractor contributors",
                "subject": "Redistribution-safe deterministic demo fixture",
            }
        )
        # Suppress PyMuPDF's generated trailer identifier so repeated builds are
        # byte-for-byte stable and therefore receive the same document SHA-256.
        document.save(output, no_new_id=True)
    finally:
        document.close()
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".demo/corpus/sample-catalog.pdf"),
    )
    args = parser.parse_args(argv)
    print(create_catalog(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
