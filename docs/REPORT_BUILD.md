# Building the detailed report

The reviewed Word and PDF files under `reports/` are generated artifacts. The
Markdown research report is the portable text source for GitHub readers; the Word
builder preserves the publication layout and architecture figures.

## Python report environment

```bash
python -m pip install -e ".[report]"
python scripts/build_detailed_report.py \
  --output reports/NVIDIA_Industrial_Catalog_Extraction_Pipeline_Detailed_Report.docx \
  --asset-dir reports/assets
```

## Preferred DOCX rendering check

Open the DOCX in Word or render it through LibreOffice, then inspect every page for
clipping, overlap, table wrapping, missing glyphs, and header/footer placement.

## Optional browser PDF fallback

The fallback converts DOCX content to HTML with Mammoth and prints it through
Playwright. Install the pinned Node dependency:

```bash
pnpm install --frozen-lockfile
python scripts/render_report_fallback.py \
  --docx reports/NVIDIA_Industrial_Catalog_Extraction_Pipeline_Detailed_Report.docx \
  --html tmp/report.html
pnpm report:pdf -- \
  tmp/report.html \
  reports/NVIDIA_Industrial_Catalog_Extraction_Pipeline_Detailed_Report.pdf
```

Playwright normally uses its managed Chromium. To use an existing compatible
browser, set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to the executable path.

After PDF creation, run `pdfinfo`, extract text, render every page to PNG with
Poppler, and inspect the page images. Before publication, scan both DOCX XML and PDF
text for personal paths, credentials, and private source content.
