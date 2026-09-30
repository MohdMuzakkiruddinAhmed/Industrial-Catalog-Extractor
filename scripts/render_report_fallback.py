from __future__ import annotations

# ruff: noqa: E501
import argparse
import base64
import html
import sys
from pathlib import Path

STYLE = r"""
:root {
  --navy: #0b1f33;
  --blue: #285a8e;
  --green: #76b900;
  --green-dark: #4c7f00;
  --ink: #172033;
  --muted: #68717d;
  --line: #cdd4dc;
  --pale-blue: #edf4fa;
  --pale-green: #eef7e4;
  --pale-gray: #f3f5f7;
  --pale-amber: #fff5dc;
}

* { box-sizing: border-box; }
html { background: white; }
body {
  margin: 0;
  color: var(--ink);
  font-family: Arial, Helvetica, sans-serif;
  font-size: 10.1pt;
  line-height: 1.34;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

p { margin: 0 0 6pt; orphans: 3; widows: 3; }
strong { color: inherit; }
a { color: var(--blue); text-decoration: underline; text-underline-offset: 1px; }
h1, h2, h3 { break-after: avoid-page; page-break-after: avoid; }
h1 {
  color: var(--blue);
  font-size: 16pt;
  line-height: 1.12;
  margin: 18pt 0 10pt;
  padding-top: 2pt;
  border-top: 3px solid var(--green);
  break-before: auto;
  page-break-before: auto;
}
h2 { color: var(--blue); font-size: 13pt; line-height: 1.15; margin: 14pt 0 7pt; }
h3 { color: var(--navy); font-size: 11.5pt; line-height: 1.15; margin: 10pt 0 5pt; }

main > p:has(> strong:only-child) {
  color: var(--green-dark);
  font-size: 8pt;
  letter-spacing: .08em;
  margin: 5pt 0 3pt;
  text-transform: uppercase;
  break-after: avoid-page;
  break-before: auto;
  page-break-before: auto;
}

ul, ol { margin: 2pt 0 8pt 18pt; padding: 0; }
li { margin: 0 0 3pt 2pt; break-inside: avoid-page; }
li::marker { color: var(--green-dark); font-weight: 700; }

table { border-collapse: collapse; width: 100%; margin: 6pt 0 9pt; }
table.report-table { font-size: 8.1pt; line-height: 1.18; }
table.report-table th,
table.report-table thead td,
table.report-table tr:first-child td {
  background: var(--navy);
  color: white;
  font-weight: 700;
}
table.report-table th,
table.report-table td { border: 1px solid var(--line); padding: 5pt 6pt; vertical-align: top; }
table.report-table tr:nth-child(even) td { background: var(--pale-gray); }
table.report-table tr { break-inside: avoid-page; page-break-inside: avoid; }

main table:not(.report-table):has(tr:only-child > :is(td, th):only-child) {
  background: #17212b;
  color: #e8edf2;
  border-radius: 4px;
  overflow: hidden;
  break-inside: avoid-page;
}
main table:not(.report-table):has(tr:only-child > :is(td, th):only-child) :is(td, th) {
  padding: 8pt 10pt;
  font-family: Consolas, "Courier New", monospace;
  font-size: 7.7pt;
  font-weight: 400;
  text-align: left;
  line-height: 1.25;
}

main table:not(.report-table):has(tr:only-child > :is(td, th):nth-child(2)) {
  background: var(--pale-blue);
  border: 1px solid #b9cede;
  border-radius: 5px;
  overflow: hidden;
  break-inside: avoid-page;
}
main table:not(.report-table):has(tr:only-child > :is(td, th):nth-child(2)) :is(td, th) {
  padding: 8pt 10pt;
  font-weight: 400;
  text-align: left;
}
main table:not(.report-table):has(tr:only-child > :is(td, th):nth-child(2)) :is(td, th):first-child {
  background: var(--green-dark);
  border: 0;
  color: transparent;
  font-size: 0;
  padding: 0;
  width: 8px;
}

p:has(> img) {
  text-align: center;
  margin: 7pt 0 2pt;
  break-inside: avoid-page;
  page-break-inside: avoid;
}
img { display: inline-block; width: 100%; max-width: 100%; height: auto; }
p:has(> img) + p {
  color: var(--muted);
  font-size: 8pt;
  font-style: italic;
  text-align: center;
  margin: 0 0 9pt;
  break-before: avoid-page;
}

.cover {
  min-height: 9.45in;
  page-break-after: always;
  break-after: page;
  padding-top: .2in;
  position: relative;
}
.cover::before {
  content: "";
  display: block;
  height: 9px;
  width: 2.1in;
  background: var(--green);
  margin: 0 0 .55in;
}
.cover > p:nth-of-type(1) {
  color: var(--green-dark);
  font-size: 8.5pt;
  letter-spacing: .09em;
  margin-bottom: .55in;
}
.cover > p:nth-of-type(2) {
  color: var(--navy);
  font-size: 29pt;
  font-weight: 700;
  line-height: 1.04;
  margin: 0 0 13pt;
}
.cover > p:nth-of-type(3) {
  color: var(--blue);
  font-size: 13pt;
  line-height: 1.25;
  max-width: 6.0in;
  margin-bottom: 18pt;
}
.cover > p:nth-of-type(4) { font-size: 9.5pt; margin-bottom: 22pt; }
.cover table:first-of-type { border-collapse: separate; border-spacing: 7px; margin: 0 -7px 12pt; }
.cover table:first-of-type :is(td, th) {
  background: var(--pale-gray);
  border: 0;
  padding: 12pt 10pt;
  text-align: center;
  vertical-align: middle;
}
.cover table:first-of-type :is(td, th) strong { color: var(--navy); font-size: 15pt; }
.cover table:nth-of-type(2) { background: var(--pale-amber); border-left: 8px solid #8a5a00; }
.cover table:nth-of-type(2) :is(td, th):first-child { display: none; }
.cover table:nth-of-type(2) :is(td, th):last-child {
  padding: 9pt 12pt;
  font-size: 8.8pt;
  font-weight: 400;
  text-align: left;
}
.cover > p:last-child { color: var(--muted); font-size: 7.4pt; margin-top: 12pt; }

@media print {
  a { color: var(--blue); }
  thead { display: table-header-group; }
}
"""


def image_converter(image):
    with image.open() as image_bytes:
        encoded = base64.b64encode(image_bytes.read()).decode("ascii")
    return {"src": f"data:{image.content_type};base64,{encoded}"}


def convert(docx_path: Path, html_path: Path) -> list[str]:
    try:
        import mammoth
    except ImportError as exc:
        raise SystemExit("mammoth is required; add its target directory to PYTHONPATH") from exc

    style_map = "\n".join(
        [
            "table[style-name='Table Grid'] => table.report-table:fresh",
            "p[style-name='Heading 1'] => h1:fresh",
            "p[style-name='Heading 2'] => h2:fresh",
            "p[style-name='Heading 3'] => h3:fresh",
        ]
    )
    with docx_path.open("rb") as source:
        result = mammoth.convert_to_html(
            source,
            style_map=style_map,
            convert_image=mammoth.images.img_element(image_converter),
            include_default_style_map=True,
        )
    body = result.value
    marker = "<p><strong>REPORT MAP</strong></p>"
    split_at = body.find(marker)
    if split_at < 0:
        raise RuntimeError("Could not locate the first body heading in converted HTML")
    cover = body[:split_at]
    main = body[split_at:]
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NVIDIA Industrial Catalog Extraction Pipeline - Detailed Report</title>
  <style>{STYLE}</style>
</head>
<body>
  <section class="cover">{cover}</section>
  <main>{main}</main>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")
    return [message.message for message in result.messages]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a print-styled HTML fallback from the generated DOCX report.")
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--mammoth-target", type=Path)
    args = parser.parse_args()
    if args.mammoth_target:
        sys.path.insert(0, str(args.mammoth_target.resolve()))
    messages = convert(args.docx.resolve(), args.html.resolve())
    print(args.html.resolve())
    for message in messages:
        print(f"MAMMOTH: {html.escape(message)}")


if __name__ == "__main__":
    main()
