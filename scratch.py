import re
from bs4 import BeautifulSoup

with open("/Users/neiljin/.gemini/antigravity/brain/5038f5f9-5203-4f07-a873-5b2c96b5b329/.system_generated/steps/115/content.md", "r") as f:
    text = f.read()

soup = BeautifulSoup(text, "html.parser")

js_html_texts = []
# Match a double-quoted string: \"  then any non-quote chars, then <tag>, then more chars, then \"
# But wait, JSON might escape quotes inside: \"
# A more robust regex for JSON strings containing HTML:
# "(?:[^"\\]|\\.)*<(?:p|h\d|li|div|ul|td|th).*?(?:[^"\\]|\\.)*"
html_string_pattern = re.compile(r'"((?:[^"\\]|\\.)*?<(?:p|h\d|li|div|ul|span|td).*?>.*?)"', re.IGNORECASE | re.DOTALL)

for script in soup.find_all("script"):
    if script.string:
        matches = html_string_pattern.findall(script.string)
        for m in matches:
            # Reconstruct HTML by decoding escaped quotes and newlines
            decoded_m = m.encode('utf-8').decode('unicode_escape', errors='ignore')
            # Using BeautifulSoup to get raw text
            clean_text = BeautifulSoup(decoded_m, "html.parser").get_text(separator=' ', strip=True)
            if len(clean_text) > 10:
                js_html_texts.append(clean_text)

print("Found blocks:", len(js_html_texts))
if js_html_texts:
    print(js_html_texts[0][:200])
