"""Render instruction.html to the PDF shipped in the package (needs: pip install playwright)."""
import pathlib
import sys

from playwright.sync_api import sync_playwright

here = pathlib.Path(__file__).resolve().parent
out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else here / "ІНСТРУКЦІЯ.pdf"
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page()
    page.goto((here / "instruction.html").as_uri())
    page.wait_for_load_state("networkidle")
    page.pdf(path=str(out), format="A4", print_background=True, prefer_css_page_size=True,
             display_header_footer=True, header_template="<span></span>",
             footer_template='<div style="font-size:8pt;color:#8aa;width:100%;text-align:center"><span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    browser.close()
print(out)
