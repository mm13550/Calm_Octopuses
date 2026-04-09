import requests
from bs4 import BeautifulSoup
import re

url = "https://ruffiannyc.com/dinner-menu"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

print(f"Total HTML string length: {len(response.text)}")

for script in soup(["script", "style", "nav", "footer", "header", "meta", "svg", "path"]):
    script.extract()

text = soup.get_text(separator=' ', strip=True)
print(f"Text length after strip: {len(text)}")
print(f"Sample text snippet: {text[:500]}")

# Look for jpg images
images = []
for img in soup.find_all('img'):
    img_src = img.get('src', '') or img.get('data-src', '')
    if 'jpg' in img_src.lower() or 'jpeg' in img_src.lower():
        images.append(img_src)

print(f"Found {len(images)} JPG images:")
for img_src in images[:5]:
    print(" - " + img_src)
    
# Look for a tags with jpg
a_images = []
for a in soup.find_all('a'):
    href = a.get('href', '')
    if 'jpg' in href.lower() or 'jpeg' in href.lower() or 'png' in href.lower():
        a_images.append(href)

print(f"Found {len(a_images)} a tags pointing to images:")
for a_src in a_images[:5]:
    print(" - " + a_src)
