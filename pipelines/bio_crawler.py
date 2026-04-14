import os
import sys
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv

load_dotenv()

def scrape_restaurant_bio_text(base_url):
    """
    Crawls the homepage and optionally /about to grab text for LLM bio extraction.
    """
    if pd.isna(base_url) or not str(base_url).startswith('http'):
        return ""
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    base_url = base_url.rstrip('/')
    urls_to_try = [base_url, f"{base_url}/about", f"{base_url}/story", f"{base_url}/our-story"]
    combined_text = []
    
    for url in urls_to_try:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Remove irrelevant blocks
                for script in soup(["script", "style", "nav", "footer", "meta", "noscript", "svg"]):
                    script.extract()
                    
                text = soup.get_text(separator=' ', strip=True)
                if len(text) > 50:
                    combined_text.append(f"--- [SOURCE: {url}] ---\n{text}")
        except Exception:
            pass
            
    return "\n".join(combined_text)

def parse_text_to_bio_with_llm(restaurant_name, text_content):
    """
    Feeds the raw combined HTML text to GPT-4o-mini to extract a clean bio.
    """
    if not text_content.strip():
        return {"description": "", "culinary_style": "", "history": ""}
        
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_prompt = """You are a Michelin dining expert. Read the provided messy website text and extract a highly professional, objective "Official Restaurant Description". 
You must return your findings in strict JSON format.

JSON Schema:
{
    "description": "A cohesive 1-2 paragraph summary of the restaurant's vibe, official brand identity, and overall dining experience.",
    "culinary_style": "2-3 short sentences describing their exact culinary focus, cuisine type, menu format (e.g., Omakase, Tasting Menu), and signature techniques.",
    "history": "Any mentioned background about the Chef(s), the timeline of opening, or the inspiration behind the restaurant. If none, leave blank."
}

Do NOT invent information. If the text does not contain it, leave the field empty. Make sure JSON is perfectly valid."""

    user_content = f"Restaurant Name: {restaurant_name}\n\nWebsite Text:\n{text_content[:20000]}"

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
            
            if result_text.startswith("```json"):
                result_text = result_text.replace("```json", "", 1)
            if result_text.endswith("```"):
                result_text = result_text[::-1].replace("```", "", 1)[::-1]
                
            return json.loads(result_text.strip())
            
        except openai.RateLimitError as e:
            if attempt < 3:
                wait_time = [5, 10, 20][attempt]
                print(f"  -> Rate limit hit! Waiting {wait_time}s before retry {attempt+1}/3...")
                time.sleep(wait_time)
            else:
                print(f"  -> Fatal Rate Limit Error for {restaurant_name}.")
                return {"description": "", "culinary_style": "", "history": ""}
        except json.JSONDecodeError:
            print(f"  -> LLM output invalid JSON for {restaurant_name}.")
            return {"description": "", "culinary_style": "", "history": ""}
        except Exception as e:
            print(f"  -> LLM Error for {restaurant_name}: {e}")
            return {"description": "", "culinary_style": "", "history": ""}
            
    return {"description": "", "culinary_style": "", "history": ""}

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    csv_path = os.path.join(data_dir, 'csv', 'seeds_resolved.csv')
    
    out_dir = os.path.join(data_dir, 'extracted_bios')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'restaurant_bios.json')

    print(f"Reading restaurants from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else len(df)
    test_df = df[start_idx:end_idx]
    print(f"Targeting batch: {start_idx} to {end_idx}")

    # Load existing to support resume
    all_extracted = []
    existing_keys = set()
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            try:
                all_extracted = json.load(f)
                existing_keys = {item['name'] for item in all_extracted}
            except json.JSONDecodeError:
                pass

    for index, row in test_df.iterrows():
        restaurant_name = row.get('name', 'Unknown')
        url = row.get('homepage')
        
        print(f"\nProcessing [{index + 1}/{len(df)}]: {restaurant_name} (URL: {url})")
        
        # Overwrite logic inside the slice being run: Remove old entry if this script is re-running it
        all_extracted = [item for item in all_extracted if item.get('name') != restaurant_name]
        
        raw_text = scrape_restaurant_bio_text(url)
        
        if not raw_text:
            print("  -> Could not extract any text from website.")
            bio_json = {"description": "", "culinary_style": "", "history": ""}
        else:
            print(f"  -> Extracted {len(raw_text)} chars of text. Analyzing with LLM...")
            bio_json = parse_text_to_bio_with_llm(restaurant_name, raw_text)
            if bio_json.get("description"):
                print("  -> Successfully pulled Bio and Details.")
            else:
                print("  -> Failed to generate robust Bio.")
                
        # Append structure
        all_extracted.append({
            "name": restaurant_name,
            "bio": bio_json
        })
        
        # Incremental save
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_extracted, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Bio extraction complete for batch {start_idx} to {end_idx}.")
    print(f"✅ Saved to: {out_path}")

if __name__ == "__main__":
    main()
