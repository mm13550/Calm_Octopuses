import requests, io, re
from pypdf import PdfReader

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# First, find the PDF URL from the site
from bs4 import BeautifulSoup
import urllib.parse

resp = requests.get('https://www.ladongnyc.com/', headers=headers)
soup = BeautifulSoup(resp.text, 'html.parser')

# Find all PDF links
pdf_links = []
for a in soup.find_all('a', href=True):
    href = a['href']
    if '.pdf' in href.lower():
        full = urllib.parse.urljoin('https://www.ladongnyc.com/', href)
        pdf_links.append((full, a.get_text().strip()))

# Also regex scan
pdf_pattern = re.compile(r'https?://[^\s"\'>]+\.pdf(?:[^\s"\'>]*)?', re.IGNORECASE)
for u in set(pdf_pattern.findall(resp.text)):
    if u not in [p[0] for p in pdf_links]:
        pdf_links.append((u, '[regex]'))

print(f"Found {len(pdf_links)} PDF links:")
for url, text in pdf_links:
    print(f"  {text}: {url}")

# Download and extract each PDF
for url, label in pdf_links:
    print(f"\n{'='*80}")
    print(f"PDF: {url} ({label})")
    print('='*80)
    try:
        r = requests.get(url, headers=headers, timeout=10)
        reader = PdfReader(io.BytesIO(r.content))
        print(f"Pages: {len(reader.pages)}")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            print(f"\n--- Page {i+1} ---")
            print(text if text else "[NO TEXT EXTRACTED]")
    except Exception as e:
        print(f"Error: {e}")
