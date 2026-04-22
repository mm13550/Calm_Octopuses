import re

import os
import json
import io
import urllib.parse
import base64
from difflib import SequenceMatcher
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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"macOS"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1'
    }
    
    keywords = ['menu', 'food', 'dinner', 'lunch', 'tasting', 'carte', 'sweets', 'dessert', 'cuisine', 'prix']
    skip_keywords = ['drink', 'beverage', 'bev', 'wine', 'cocktail', 'beer', 'liquor', 'catering', 'event']
    # Common non-menu PDF filenames to skip (legal docs, etc.)
    pdf_skip_names = [
        'terms', 'termsofuse', 'terms-of-use', 'terms_of_use',
        'privacy', 'privacypolicy', 'privacy-policy', 'privacy_policy',
        'disclaimer', 'accessibility', 'sitemap', 'cookie', 'gdpr',
        'press', 'pressrelease', 'press-release', 'annual-report', 'annual_report',
        'covid', 'safe', 'manual', 'guidelines', 'policy', 'safety'
    ]
    # Locations that are definitely NOT NYC - used to filter out menus from other cities
    # for multi-location restaurant chains (e.g. Nami Nori has Atlantic Park, Design District, Montclair)
    non_nyc_locations = [
        'atlantic-park', 'design-district', 'montclair', 'chicago', 'miami', 'boston',
        'los-angeles', 'la-', 'houston', 'dallas', 'seattle', 'denver', 'atlanta',
        'philadelphia', 'dc', 'washington', 'san-francisco', 'sf-', 'las-vegas',
        'toronto', 'london', 'paris', 'tokyo', 'hamptons', 'jersey','singapore','vegas','seoul'
    ]
        
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
            
            # --- HIDDEN SPA / JSON LINK EXTRACTOR ---
            # SPA / Next.js / Prismic CMS often hide navigation links inside huge JSON payloads rather than <a> tags
            hidden_links = re.findall(r"\"(/(?:cuisine|menu|dining|dinner|lunch|prix-?[a-z]*|tasting|food)[^\"]*)\"", sub_response.text, re.IGNORECASE)
            for hl in hidden_links:
                clean_hl = hl.rstrip('\\')
                full_nested = urllib.parse.urljoin(current_target, clean_hl)
                if full_nested not in visited and full_nested not in queue:
                    if not any(k in full_nested.lower() for k in skip_keywords):
                        queue.append(full_nested)

            # Find deeper HTML links FIRST, before we destroy nav/header tags for text extraction
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
                    
                    parsed_img = urllib.parse.urlparse(nested_link.lower())
                    if any(parsed_img.path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']) or 'format=jpg' in nested_link.lower() or 'format=webp' in nested_link.lower():
                        is_menu_page = any(k in current_target.lower() for k in ['menu', 'food', 'dinner', 'lunch'])
                        is_img_keyword = any(k in nested_link.lower() or k in text for k in image_keywords)
                        if is_img_keyword or is_menu_page:
                            if nested_link not in image_urls_list and len(image_urls_list) < 5:
                                image_urls_list.append(nested_link)
                        continue

                    if urllib.parse.urlparse(nested_link).netloc == urllib.parse.urlparse(url).netloc:
                        if nested_link not in visited and nested_link not in queue:
                            # Skip menus that are explicitly for non-NYC locations
                            link_path_lower = urllib.parse.urlparse(nested_link).path.lower()
                            is_non_nyc = any(loc in link_path_lower or loc in text for loc in non_nyc_locations)
                            if is_non_nyc:
                                continue
                            # Skip legal / non-menu PDFs by filename
                            if '.pdf' in nested_link.lower():
                                pdf_fname = link_path_lower.split('/')[-1].split('?')[0]
                                if any(skip in pdf_fname for skip in pdf_skip_names):
                                    continue
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
                    parsed_src = urllib.parse.urlparse(src.lower())
                    if any(parsed_src.path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']) or 'format=jpg' in src.lower() or 'format=webp' in src.lower():
                        img_link = urllib.parse.urljoin(current_target, src)
                        if img_link not in image_urls_list and len(image_urls_list) < 8:
                            image_urls_list.append(img_link)

            # --- RSC / JSON PAYLOAD EXTRACTOR ---
            # Try to grab content from Next.js or React Server Component payloads hidden in the raw HTML string
            json_food_pattern = re.compile(
                r"\\?\"(?:name|title)\\?\"\s*:\s*\\?\"(.*?)\\?\"\s*,\s*(?:\\?\"description\\?\"\s*:\s*\\?\"(.*?)\\?\"\s*,\s*)?\\?\"price\\?\"\s*:\s*\\?\"(.*?)\\?\"", 
                re.IGNORECASE
            )
            raw_matches = json_food_pattern.findall(sub_response.text)
            for m in raw_matches:
                try:
                    name, desc, price = m
                    clean_name = name.encode('utf-8').decode('unicode_escape', errors='ignore').replace('$', '')
                    clean_desc = desc.encode('utf-8').decode('unicode_escape', errors='ignore').replace('$undefined', '')
                    clean_price = price.encode('utf-8').decode('unicode_escape', errors='ignore')
                    if len(clean_name) > 3 and clean_price and clean_price != 'undefined':
                        html_texts.append(f"--- [JSON FOOD ITEM] ---\nName: {clean_name}\nDesc: {clean_desc}\nPrice: {clean_price}\n")
                except Exception:
                    pass

            # --- JS HTML EXTRACTOR ---
            # Try to grab content from common JS variables or blurb fields that contain embedded HTML menus (e.g. Le Bernardin)
            html_string_pattern = re.compile(r'"((?:[^"\\]|\\.)*?<(?:p|h\d|li|div|ul|span|td).*?>.*?)"', re.IGNORECASE | re.DOTALL)
            for script in target_soup.find_all('script'):
                if script.string:
                    matches = html_string_pattern.findall(script.string)
                    for m in matches:
                        try:
                            # Reconstruct HTML by decoding escaped quotes and newlines
                            decoded_m = m.encode('utf-8').decode('unicode_escape', errors='ignore')
                            clean_text = BeautifulSoup(decoded_m, "html.parser").get_text(separator=' ', strip=True)
                            if len(clean_text) > 20:  # Avoid tiny fragments
                                html_texts.append(f"--- [JS CONTENT: {current_target}] ---\n{clean_text}")
                        except Exception:
                            pass

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
        pdf_url_pattern = re.compile(r'https?://[^\s"\'<>]+\.pdf(?:[^\s"\'<>]*)?', re.IGNORECASE)
        seen_pdf_urls = set()
        seen_pdf_filenames = set()  # dedup by filename hash (same file on multiple CDNs)
        MAX_REGEX_PDFS = 5  # cap to avoid downloading dozens of unrelated PDFs (e.g. museum sites)
        
        for page_url, raw_html in raw_html_cache.items():
            found_urls = pdf_url_pattern.findall(raw_html)
            for raw_pdf_url in found_urls:
                if len(pdf_texts) >= MAX_REGEX_PDFS:
                    break
                # Decode any JSON unicode escapes (e.g. \u002F -> /)
                try:
                    raw_pdf_url = raw_pdf_url.encode('utf-8').decode('unicode_escape')
                except Exception:
                    pass
                raw_pdf_url = raw_pdf_url.rstrip('"\'  ')
                
                if raw_pdf_url in seen_pdf_urls:
                    continue
                seen_pdf_urls.add(raw_pdf_url)
                
                # Dedup by filename: same PDF hosted on multiple CDNs (e.g. Sanity + BunnyCDN)
                pdf_filename = urllib.parse.urlparse(raw_pdf_url).path.split('/')[-1].split('?')[0]
                if pdf_filename in seen_pdf_filenames:
                    continue
                seen_pdf_filenames.add(pdf_filename)
                
                # Filter out drink menus at URL level
                url_lower = raw_pdf_url.lower()
                if any(k in url_lower for k in skip_keywords):
                    continue
                # Filter out non-NYC location PDFs
                if any(loc in url_lower for loc in non_nyc_locations):
                    continue
                # Filter out legal / non-menu PDFs by filename
                if any(skip in pdf_filename.lower() for skip in pdf_skip_names):
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
    
    if len(final_pdf_text) > 0 and len(final_html_text) > 0:
        combined_text = final_pdf_text[:25000] + "\n\n" + final_html_text[:25000]
    else:
        combined_text = (final_pdf_text + final_html_text)[:50000]
            
    return combined_text, image_urls_list

def upgrade_image_resolution(image_url):
    """Upgrade CDN image URLs to maximum resolution for better OCR by GPT."""
    # Shopify CDN: replace &width=NNN or ?width=NNN with &width=2000
    image_url = re.sub(r'([?&])width=\d+', r'\g<1>width=2000', image_url)
    # Squarespace CDN: replace ?format=NNNw with ?format=2000w
    image_url = re.sub(r'\?format=\d+w', '?format=2000w', image_url)
    return image_url

def encode_image_to_base64(image_url):
    try:
        # Upgrade to max resolution before downloading
        image_url = upgrade_image_resolution(image_url)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            # Explicitly exclude AVIF - OpenAI does not support it. CDNs like Prismic use
            # content negotiation and will serve AVIF if it's in Accept header.
            'Accept': 'image/webp,image/png,image/jpeg,image/*;q=0.8',
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        
        # Fallback: Some CDNs block "spoofed" standard browsers due to TLS mismatches, 
        # but allow honest python-requests or cURL user-agents.
        if response.status_code == 403:
            response = requests.get(image_url, timeout=10)
            
        response.raise_for_status()
        
        # Validate content type - skip unsupported formats (AVIF, SVG, etc.)
        content_type = response.headers.get('Content-Type', '').lower()
        supported = any(fmt in content_type for fmt in ['jpeg', 'jpg', 'png', 'gif', 'webp'])
        if not supported:
            print(f"  -> Skipping unsupported image format ({content_type}): {image_url[:80]}")
            return None
        
        encoded = base64.b64encode(response.content).decode('utf-8')
        return encoded
    except Exception as e:
        print(f"  -> Failed to download/encode image {image_url}: {e}")
        return None

def normalize_domain(url):
    if not url:
        return ""
    try:
        host = urllib.parse.urlparse(str(url)).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def normalize_name(value):
    if isinstance(value, dict):
        value = value.get("text") or value.get("displayName") or ""
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def get_place_id(restaurant_name, homepage=None, strict_homepage_match=True):
    """Resolve a Google Place ID, preferring candidates that match the known homepage."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return None

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.websiteUri",
    }
    payload = {"textQuery": f"{restaurant_name} restaurant NYC"}
    target_name = normalize_name(restaurant_name)
    target_domain = normalize_domain(homepage)

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        if response.status_code != 200:
            return None

        places = response.json().get("places") or []
        if not places:
            return None

        best_place = None
        best_score = -1.0
        best_accepted_place = None
        best_accepted_score = -1.0

        for place in places:
            candidate_name = normalize_name(place.get("displayName"))
            candidate_domain = normalize_domain(place.get("websiteUri"))
            name_score = SequenceMatcher(None, target_name, candidate_name).ratio()
            homepage_match = bool(target_domain and candidate_domain and target_domain == candidate_domain)
            accepted = homepage_match or name_score >= 0.72
            score = name_score + (1.0 if homepage_match else 0.0)

            if score > best_score:
                best_score = score
                best_place = place
            if accepted and score > best_accepted_score:
                best_accepted_score = score
                best_accepted_place = place

        if best_accepted_place:
            return best_accepted_place.get("id")

        if homepage and strict_homepage_match:
            return None

        return (best_place or {}).get("id")
    except Exception:
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

        import time
        import openai
        
        result_text = ""
        for attempt in range(4):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.0
                )
                result_text = response.choices[0].message.content.strip()
                break  # Success, exit retry loop
            except openai.RateLimitError as e:
                if attempt < 3:
                    wait_time = [5, 10, 20][attempt]
                    print(f"  -> Rate limit hit! Waiting {wait_time}s before retry {attempt+1}/3...")
                    time.sleep(wait_time)
                else:
                    raise e
        
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

    csv_path = os.path.join(data_dir, 'csv', 'seeds_resolved.csv')
    output_path = os.path.join(extracted_menus_dir, 'parsed_menus.json')

    print(f"Reading restaurants from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    import sys
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    test_df = df[start_idx:end_idx]
    print(f"Targeting batch: {start_idx} to {end_idx}")
    
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
            rest_id = get_place_id(restaurant_name, homepage=url) or f"dummy_{index}"
            print(f"  -> Target rest_id resolved to: {rest_id}")
            print(f"  -> Extracted {len(raw_text)} characters and {len(image_urls)} menu images. Sending to GPT-4o-mini...")
            structured_dishes = parse_text_to_json_with_llm(restaurant_name, rest_id, raw_text, image_urls)
            print(f"  -> Successfully structured {len(structured_dishes)} dishes.")
            all_extracted_menus.extend(structured_dishes)
        else:
            print(f"  -> No usable text or images found.")

    # Save output rigidly as JSON (append mode to prevent overwriting previous batches)
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.extend(all_extracted_menus)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Pipeline Complete. Extracted {len(all_extracted_menus)} total dish records.")
    print(f"✅ Data exported successfully to: {output_path}")

if __name__ == "__main__":
    main()
