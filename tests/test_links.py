import requests
from bs4 import BeautifulSoup
import urllib.parse

url = "https://www.lepavillonnyc.com/"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

found = []
for link in soup.find_all('a', href=True):
    found.append((link.get_text().strip(), link['href']))

for text, href in found:
    print(f"TEXT: '{text}'  HREF: '{href}'")
