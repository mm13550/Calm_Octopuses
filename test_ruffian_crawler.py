import requests
from bs4 import BeautifulSoup
import urllib.parse

def test_ruffian():
    url = "https://ruffiannyc.com/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    keywords = ['menu', 'food', 'dinner', 'lunch', 'tasting', 'carte', 'sweets', 'dessert']
    skip_keywords = ['drink', 'beverage', 'bev', 'wine', 'cocktail', 'beer', 'liquor', 'catering', 'event']
        
    queue = [url]
    visited = set()
    
    html_texts = []
    
    while queue and len(visited) < 10:
        current_target = queue.pop(0)
        if current_target in visited:
            continue
        visited.add(current_target)
        
        try:
            sub_response = requests.get(current_target, headers=headers, timeout=10)
            if sub_response.status_code != 200:
                continue
                
            target_soup = BeautifulSoup(sub_response.text, 'html.parser')
            print(f"\nEvaluating {current_target}:")
            
            for link in target_soup.find_all('a', href=True):
                if not link.has_attr('href'):
                    continue
                href = link['href']
                text = link.get_text().lower()
                
                path_and_text = (urllib.parse.urlparse(href).path + " " + text).lower()
                if any(k in path_and_text for k in skip_keywords):
                    continue

                if any(k in text or k in href.lower() for k in keywords) or '.pdf' in href.lower():
                    nested_link = urllib.parse.urljoin(current_target, href)
                    if urllib.parse.urlparse(nested_link).netloc == urllib.parse.urlparse(url).netloc:
                        if nested_link not in visited and nested_link not in queue:
                            queue.append(nested_link)
                            print(f"  + Enqueued {nested_link}")
            
            for script in target_soup(["script", "style", "nav", "footer", "header", "meta"]):
                script.extract()
            page_text = target_soup.get_text(separator=' ', strip=True)
            print(f"  -> Extracted text length: {len(page_text)}")
                                
        except Exception as e:
            print(f"Failed {current_target}: {e}")

test_ruffian()
