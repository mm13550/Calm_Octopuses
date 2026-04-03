import re
import os
import json
import io
import urllib.parse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (particularly OPENAI_API_KEY)
load_dotenv()

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def scrape_menu_text(url):
    """
    An upgraded robust scraper using a BFS queue to hunt for PDF menus, 
    and falling back to an HTML crawler. It tracks PDFs and HTML separately,
    prioritizing parsed PDF text over HTML noise.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    keywords = ['menu', 'food', 'dinner', 'lunch', 'tasting', 'carte', 'sweets', 'dessert']
    skip_keywords = ['drink', 'beverage', 'bev', 'wine', 'cocktail', 'beer', 'liquor', 'catering', 'event']
        
    queue = [url]
    visited = set()
    
    pdf_texts = []
    html_texts = []
    raw_html_cache = {}  # cache raw HTML for regex scanning later
    
    while queue and len(visited) < 10:  # Allow up to 10 page visits to find all menus
        current_target = queue.pop(0)
        if current_target in visited:
            continue
        visited.add(current_target)
        
        try:
            sub_response = requests.get(current_target, headers=headers, timeout=10)
            if sub_response.status_code != 200:
                continue
                
            content_type = sub_response.headers.get('Content-Type', '').lower()
            
            # --- HANDLE PDF ---
            if 'application/pdf' in content_type or current_target.split('?')[0].lower().endswith('.pdf'):
                print(f"  -> Parsing PDF: {current_target}")
                pdf_file = io.BytesIO(sub_response.content)
                try:
                    reader = PdfReader(pdf_file)
                    extracted_pdf = ""
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            extracted_pdf += extracted + "\n"
                    if extracted_pdf.strip():
                        pdf_texts.append(f"--- [PDF: {current_target}] ---\n{extracted_pdf}")
                except Exception as e:
                    print(f"  -> PDF parse error for {current_target}: {e}")
                continue # Do not parse HTML links inside a PDF
                
            # --- HANDLE HTML ---
            target_soup = BeautifulSoup(sub_response.text, 'html.parser')
            raw_html_cache[current_target] = sub_response.text  # cache for later regex scan
            
            # Find deeper links FIRST, before we destroy nav/header tags for text extraction
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
                            # Prioritize PDFs
                            if '.pdf' in nested_link.lower() or 'pdf' in text:
                                queue.insert(0, nested_link)
                            else:
                                queue.append(nested_link)
            
            # Extract text to use as fallback (now it's safe to destroy elements)
            for script in target_soup(["script", "style", "nav", "footer", "header", "meta"]):
                script.extract()
            page_text = target_soup.get_text(separator=' ', strip=True)
            if page_text:
                html_texts.append(f"--- [HTML: {current_target}] ---\n{page_text}")
                                
        except Exception as e:
            print(f"  -> Failed to chase subpage {current_target}: {e}")
            
    # Priority: If we successfully extracted PDF text, throw away HTML to save tokens and reduce noise!
    final_pdf_text = "\n".join(pdf_texts).strip()
    if len(final_pdf_text) > 50:
        print(f"  -> Extracted {len(final_pdf_text)} characters from {len(pdf_texts)} PDF(s). Discarding HTML noise.")
        return final_pdf_text[:15000]
    
    # --- ENGINE 3: RAW HTML REGEX PDF SCANNER ---
    # Handles JS-heavy sites (Squarespace, Wix, Webflow) where PDFs are embedded
    # in <script> tags or data attributes, invisible to BeautifulSoup's <a> tag search.
    print(f"  -> No PDFs found via links. Scanning raw HTML source for embedded PDF URLs...")
    pdf_url_pattern = re.compile(r'https?://[^\s"\'>]+\.pdf(?:[^\s"\'>]*)?', re.IGNORECASE)
    seen_pdf_urls = set()
    
    for page_url, raw_html in raw_html_cache.items():
        found_urls = pdf_url_pattern.findall(raw_html)
        for raw_pdf_url in found_urls:
            # Decode any JSON unicode escapes (e.g. \u002F -> /)
            try:
                raw_pdf_url = raw_pdf_url.encode('utf-8').decode('unicode_escape')
            except Exception:
                pass
            raw_pdf_url = raw_pdf_url.rstrip('"\' ')
            
            if raw_pdf_url in seen_pdf_urls:
                continue
            seen_pdf_urls.add(raw_pdf_url)
            
            # Filter out drink menus at URL level
            url_lower = raw_pdf_url.lower()
            if any(k in url_lower for k in skip_keywords):
                print(f"  -> Skipping drink PDF: {raw_pdf_url}")
                continue
            
            print(f"  -> [Regex] Found embedded PDF: {raw_pdf_url}")
            try:
                pdf_resp = requests.get(raw_pdf_url, headers=headers, timeout=10)
                if pdf_resp.status_code == 200 and 'pdf' in pdf_resp.headers.get('Content-Type','').lower():
                    reader = PdfReader(io.BytesIO(pdf_resp.content))
                    extracted_pdf = ""
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            extracted_pdf += t + "\n"
                    if extracted_pdf.strip():
                        pdf_texts.append(f"--- [PDF via Regex: {raw_pdf_url}] ---\n{extracted_pdf}")
            except Exception as e:
                print(f"  -> [Regex] Failed to fetch/parse {raw_pdf_url}: {e}")
    
    final_pdf_text = "\n".join(pdf_texts).strip()
    if len(final_pdf_text) > 50:
        print(f"  -> [Regex] Extracted {len(final_pdf_text)} chars from {len(pdf_texts)} hidden PDF(s).")
        return final_pdf_text[:15000]
        
    final_html_text = "\n".join(html_texts).strip()
    return final_html_text[:15000]

def parse_text_to_json_with_llm(restaurant_name, raw_text):
    """
    Uses OpenAI's gpt-4o-mini to convert the unstructured raw text into structured JSON.
    """
    if not raw_text.strip():
        return []

    # Extremely rigid system prompt to enforce strict JSON array output
    system_prompt = (
        "You are an expert culinary data extractor. Your task is to process raw, "
        "unstructured menu text from a restaurant and extract the menu items into a highly "
        "structured JSON array.\n\n"
        "IMPORTANT RULES:\n"
        "1. Do NOT include any markdown formatting, backticks, or code blocks (e.g., ```json) in your output.\n"
        "2. Your output must strictly be a valid JSON array of objects.\n"
        "3. Each object must exactly contain the following keys: "
        "'restaurant_name', 'dish_name', 'ingredients', and 'price'.\n"
        f"4. For the 'restaurant_name' field, unconditionally use this value: '{restaurant_name}'.\n"
        "5. If a specific field like ingredients or price cannot be found for a dish, assign it an empty string \"\".\n"
        "6. CRITICAL: EXCLUDE ALL DRINKS. Do not extract wines, cocktails, beers, sodas, or beverages of any kind. Only extract food items.\n"
        "7. Output absolutely nothing else besides the raw JSON array string."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Parse the following menu text to JSON:\n\n{raw_text}"}
            ],
            temperature=0.0
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Guard against LLM accidentally retaining markdown anyway
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "", 1)
        if result_text.endswith("```"):
            result_text = result_text[::-1].replace("```", "", 1)[::-1]
            
        parsed_json = json.loads(result_text.strip())
        return parsed_json

    except json.JSONDecodeError as e:
        print(f"  -> Failed to decode JSON from LLM response for {restaurant_name}: {e}")
        # Print a snippet of the failed output for debugging
        print(f"  -> Debug Output Snippet: {result_text[:100]}...")
        return []
    except Exception as e:
        print(f"  -> LLM Processing failed for {restaurant_name}: {e}")
        return []

def main():
    # Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # Automatically create necessary pipeline directories
    images_dir = os.path.join(data_dir, 'images')
    extracted_menus_dir = os.path.join(data_dir, 'extracted_menus')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(extracted_menus_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, 'seeds_resolved.csv')
    output_path = os.path.join(extracted_menus_dir, 'parsed_menus.json')

    print(f"Reading restaurants from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Target 1: Apply df.head(3) for testing
    test_df = df.head(10)
    
    all_extracted_menus = []

    for index, row in test_df.iterrows():
        restaurant_name = row.get('name', 'Unknown')
        url = row.get('homepage')
        
        print(f"\nProcessing [{index + 1}/{len(test_df)}]: {restaurant_name}")
        
        if pd.isna(url) or not str(url).startswith('http'):
            print(f"  -> Invalid URL, skipping.")
            continue
            
        raw_text = scrape_menu_text(url)
        
        if raw_text:
            print(f"  -> Extracted {len(raw_text)} characters. Sending to GPT-4o-mini to standardize JSON...")
            structured_dishes = parse_text_to_json_with_llm(restaurant_name, raw_text)
            print(f"  -> Successfully structured {len(structured_dishes)} dishes.")
            all_extracted_menus.extend(structured_dishes)
        else:
            print(f"  -> No usable text found.")

    # Save output rigidly as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_menus, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Pipeline Complete. Extracted {len(all_extracted_menus)} total dish records.")
    print(f"✅ Data exported successfully to: {output_path}")

if __name__ == "__main__":
    main()