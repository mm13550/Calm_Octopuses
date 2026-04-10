import re
import os
import json
import io
import urllib.parse
import base64
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
    image_urls_list = []
    image_keywords = ['menu', 'a la carte', 'a-la-carte', 'alacarte', 'dinner', 'lunch', 'dessert']
    
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

                if len(visited) <= 1 or any(k in text or k in href.lower() for k in keywords) or '.pdf' in href.lower():
                    nested_link = urllib.parse.urljoin(current_target, href)
                    
                    if any(nested_link.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']) or 'format=jpg' in nested_link.lower() or 'format=webp' in nested_link.lower():
                        is_menu_page = any(k in current_target.lower() for k in ['menu', 'food', 'dinner', 'lunch'])
                        is_img_keyword = any(k in nested_link.lower() or k in text for k in image_keywords)
                        if is_img_keyword or is_menu_page:
                            if nested_link not in image_urls_list and len(image_urls_list) < 5:
                                image_urls_list.append(nested_link)
                        continue

                    if urllib.parse.urlparse(nested_link).netloc == urllib.parse.urlparse(url).netloc:
                        if nested_link not in visited and nested_link not in queue:
                            # Prioritize PDFs
                            if '.pdf' in nested_link.lower() or 'pdf' in text:
                                queue.insert(0, nested_link)
                            else:
                                queue.append(nested_link)
            
            # look for images directly in img tags
            for img in target_soup.find_all('img'):
                src = img.get('src') or img.get('data-src') or ''
                alt = img.get('alt') or ''
                is_menu_page = any(k in current_target.lower() for k in ['menu', 'food', 'dinner', 'lunch'])
                is_img_keyword = any(k in src.lower() or k in alt.lower() for k in image_keywords)
                if is_img_keyword or is_menu_page:
                    if any(src.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']) or 'format=jpg' in src.lower() or 'format=webp' in src.lower():
                        img_link = urllib.parse.urljoin(current_target, src)
                        if img_link not in image_urls_list and len(image_urls_list) < 8:
                            image_urls_list.append(img_link)

            # Extract text to use as fallback (now it's safe to destroy elements)
            for script in target_soup(["script", "style", "nav", "footer", "header", "meta"]):
                script.extract()
            page_text = target_soup.get_text(separator=' ', strip=True)
            if page_text:
                html_texts.append(f"--- [HTML: {current_target}] ---\n{page_text}")
                                
        except Exception as e:
            print(f"  -> Failed to chase subpage {current_target}: {e}")
            
    final_pdf_text = "\n".join(pdf_texts).strip()
    if len(final_pdf_text) > 50:
        print(f"  -> Extracted {len(final_pdf_text)} characters from {len(pdf_texts)} PDF(s).")
    else:
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
            
    final_html_text = "\n".join(html_texts).strip()
    
    combined_text = ""
    if len(final_pdf_text) > 50:
        combined_text += final_pdf_text
    
    # Always include HTML text now
    if len(final_html_text) > 50:
        print(f"  -> Extracted {len(final_html_text)} characters from HTML.")
        if combined_text:
            combined_text += "\n\n" + final_html_text
        else:
            combined_text = final_html_text
            
    return combined_text[:50000], image_urls_list

def encode_image_to_base64(image_url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
            'Referer': image_url
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        
        # Fallback: Some CDNs block "spoofed" standard browsers due to TLS mismatches, 
        # but allow honest python-requests or cURL user-agents.
        if response.status_code == 403:
            # Fall back to default requests user-agent (python-requests/2.x)
            response = requests.get(image_url, timeout=10)
            
        response.raise_for_status()
        encoded = base64.b64encode(response.content).decode('utf-8')
        return encoded
    except Exception as e:
        print(f"  -> Failed to download/encode image {image_url}: {e}")
        return None

def get_place_id(restaurant_name):
    """Fetch the robust Google Place ID (rest_id) dynamically for relational mapping."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key: return None
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key, "X-Goog-FieldMask": "places.id"}
    payload = {"textQuery": f"{restaurant_name} restaurant NYC"}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=5)
        if r.status_code == 200 and r.json().get('places'):
            return r.json()['places'][0]['id']
    except Exception:
        pass
    return None

def parse_text_to_json_with_llm(restaurant_name, rest_id, raw_text, image_urls):
    """
    Uses OpenAI's gpt-4o-mini to convert the unstructured raw text into structured JSON.
    """
    if not raw_text.strip() and not image_urls:
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
        "'rest_id', 'restaurant_name', 'dish_name', 'ingredients', and 'price'.\n"
        f"4. For the 'restaurant_name' field, unconditionally use this value: '{restaurant_name}'.\n"
        f"   For the 'rest_id' field, unconditionally use this value: '{rest_id}'.\n"
        "5. Clean and Fix Prices: Due to PDF extraction errors, prices might appear duplicated (e.g., '2424' instead of '24', or '1616' instead of '16'). You MUST mathematically logicalize and fix these duplicated numbers back to their normal 2-digit menu price (e.g., 24). If a price is listed as 'HALF DOZEN 16 / DOZEN 32', output it exactly as '16 / 32'.\n"
        "6. Match Missing Prices: If the text extraction jumbled the layout and separated prices from dish names, you MUST use contextual logic and ordering to reunite every dish with its correct price. Do not leave prices blank unless they truly do not exist on the menu.\n"
        "7. CRITICAL: EXCLUDE ALL DRINKS. Do not extract wines, cocktails, beers, sodas, or beverages of any kind. Only extract food items.\n"
        "8. CRITICAL: The input text contains MULTIPLE sources (e.g., PDF menus, HTML lunch/dinner menus). You MUST extract the dishes from ALL sources. Do not stop until all food items from all menus are extracted!\n"
        "9. Output absolutely nothing else besides the raw JSON array string."
    )

    try:
        user_content = [{"type": "text", "text": f"Parse the following menu text/images to JSON:\n\n{raw_text}"}]
        
        for img_url in image_urls:
            base64_img = encode_image_to_base64(img_url)
            if base64_img:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}",
                        "detail": "high"
                    }
                })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
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
    test_df = df[63:64]
    
    all_extracted_menus = []

    for index, row in test_df.iterrows():
        restaurant_name = row.get('name', 'Unknown')
        url = row.get('homepage')
        
        print(f"\nProcessing [{index + 1}/{len(test_df)}]: {restaurant_name}")
        
        if pd.isna(url) or not str(url).startswith('http'):
            print(f"  -> Invalid URL, skipping.")
            continue
            
        raw_text, image_urls = scrape_menu_text(url)
        
        if raw_text or image_urls:
            rest_id = get_place_id(restaurant_name) or f"dummy_{index}"
            print(f"  -> Target rest_id resolved to: {rest_id}")
            print(f"  -> Extracted {len(raw_text)} characters and {len(image_urls)} menu images. Sending to GPT-4o-mini...")
            structured_dishes = parse_text_to_json_with_llm(restaurant_name, rest_id, raw_text, image_urls)
            print(f"  -> Successfully structured {len(structured_dishes)} dishes.")
            all_extracted_menus.extend(structured_dishes)
        else:
            print(f"  -> No usable text or images found.")

    # Save output rigidly as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_menus, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Pipeline Complete. Extracted {len(all_extracted_menus)} total dish records.")
    print(f"✅ Data exported successfully to: {output_path}")

if __name__ == "__main__":
    main()