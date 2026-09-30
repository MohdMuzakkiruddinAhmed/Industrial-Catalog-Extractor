const { chromium } = require('playwright');
const path = require('path');

async function main() {
  const [htmlPath, pdfPath] = process.argv.slice(2);
  if (!htmlPath || !pdfPath) {
    throw new Error('Usage: node print_report_pdf.js <input.html> <output.pdf>');
  }
  const launchOptions = { headless: true };
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH) {
    launchOptions.executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  }
  const browser = await chromium.launch(launchOptions);
  try {
    const page = await browser.newPage({ viewport: { width: 1365, height: 1800 } });
    await page.goto('file:///' + path.resolve(htmlPath).replace(/\\/g, '/'), { waitUntil: 'load' });
    await page.emulateMedia({ media: 'print' });
    await page.pdf({
      path: path.resolve(pdfPath),
      format: 'Letter',
      printBackground: true,
      displayHeaderFooter: true,
      margin: { top: '0.52in', right: '0.76in', bottom: '0.58in', left: '0.76in' },
      headerTemplate: '<div style="box-sizing:border-box;width:100%;padding:0 0.76in;font:700 7px Arial;color:#68717d;text-align:right;letter-spacing:.04em;">NVIDIA INDUSTRIAL CATALOG EXTRACTION &nbsp;•&nbsp; TECHNICAL REPORT</div>',
      footerTemplate: '<div style="box-sizing:border-box;width:100%;padding:0 0.76in;font:7px Arial;color:#68717d;display:flex;justify-content:space-between;"><span>Architecture and implementation report &nbsp;•&nbsp; 7 August 2026</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>',
      tagged: true,
      outline: true,
    });
  } finally {
    await browser.close();
  }
  process.stdout.write(path.resolve(pdfPath) + '\n');
}

main().catch((error) => {
  process.stderr.write(String(error.stack || error) + '\n');
  process.exit(1);
});
