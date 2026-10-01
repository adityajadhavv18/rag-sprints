"""Download the shared sprint corpus: latest 10-K for each company from SEC EDGAR.

Saves raw HTML to data/raw/ and clean text to data/<doc_id>.txt.
The doc_id (e.g. "aapl_10k_2025") is what golden.jsonl's source_doc refers to.

Run from rag_sprints/:  uv run scripts/download_10ks.py
"""

import re
import warnings
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# 10-Ks are inline-XBRL (XHTML); the HTML parser handles them fine
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# SEC asks every client to identify itself via User-Agent
HEADERS = {"User-Agent": "rag-sprints-learning research@example.com"}
COMPANIES = {"aapl": "0000320193", "msft": "0000789019", "tsla": "0001318605"}

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"


def latest_10k(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    recent = requests.get(url, headers=HEADERS, timeout=30).json()["filings"]["recent"]
    i = recent["form"].index("10-K")
    return {
        "accession": recent["accessionNumber"][i].replace("-", ""),
        "doc": recent["primaryDocument"][i],
        "fiscal_year": recent["reportDate"][i][:4],
    }


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Inline XBRL hides machine-readable metadata in <ix:header>; it's not part of the readable filing
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()
    # Flatten each table row into one "label | value | value" line; otherwise every cell
    # lands on its own line and numbers get separated from their labels
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c not in ("", "$", "%", ")")]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")
    # Break lines only after block elements. get_text("\n") would also break around every
    # inline tag, and 10-Ks wrap each number in an XBRL tag ("$\n34.3\n billion").
    for block in soup.find_all(["p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        block.append("\n")
    text = soup.get_text()
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, cik in COMPANIES.items():
        filing = latest_10k(cik)
        doc_id = f"{ticker}_10k_{filing['fiscal_year']}"
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{filing['accession']}/{filing['doc']}"

        html = requests.get(url, headers=HEADERS, timeout=60).text
        (RAW_DIR / f"{doc_id}.htm").write_text(html)
        text = html_to_text(html)
        (DATA_DIR / f"{doc_id}.txt").write_text(text)
        print(f"{doc_id}: {len(text):,} chars  <- {url}")


if __name__ == "__main__":
    main()
