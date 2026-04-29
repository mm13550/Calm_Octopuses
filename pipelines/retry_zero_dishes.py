"""
pipelines/retry_zero_dishes.py
===============================
Targeted recovery for failed menu crawls.

Identifies restaurants in the catalog that have zero extracted dishes and 
triggers a targeted re-run of ``menu_crawler.py`` to attempt recovery via 
different crawl paths or LLM re-parsing.

Usage::

    python pipelines/retry_zero_dishes.py [start_idx] [end_idx]
"""
import os
import json
import pandas as pd
import sys

# Append the current directory so we can import menu_crawler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from menu_crawler import scrape_menu_text, parse_text_to_json_with_llm, get_place_id

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    csv_path = os.path.join(data_dir, 'csv', 'seeds_resolved.csv')
    parsed_menus_path = os.path.join(data_dir, 'extracted_menus', 'final_parsed_menus.json')
    output_path = os.path.join(data_dir, 'extracted_menus', 'retried_menus.json')

    print(f"Reading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    test_df = df[start_idx:end_idx]
    print(f"Checking missing dishes for batch: {start_idx} to {end_idx}")

    # Load parsed menus to see which restaurants succeeded
    if os.path.exists(parsed_menus_path):
        with open(parsed_menus_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []
        
    successful_restaurants = set([dish.get('restaurant_name') for dish in existing_data if dish.get('restaurant_name')])
    
    failed_rows = []
    for index, row in test_df.iterrows():
        name = row.get('name', 'Unknown')
        # If the name is not in successful_restaurants, it got 0 dishes
        if name not in successful_restaurants:
            failed_rows.append((index, row))

    if not failed_rows:
        print("No zero-dish restaurants found in batch 120-180.")
        return

    print(f"\n--- Found {len(failed_rows)} restaurants with 0 dishes in this batch! ---")
    for index, row in failed_rows:
        print(f"Index [{index + 1}]: {row.get('name')}")
    print("-------------------------------------------------------------------\n")
    
    retried_menus = []
    for index, row in failed_rows:
        restaurant_name = row.get('name', 'Unknown')
        url = row.get('homepage')
        print(f"Retrying [{index + 1}/{len(df)}]: {restaurant_name} (URL: {url})")
        
        if pd.isna(url) or not str(url).startswith('http'):
            print("  -> Invalid URL")
            continue
            
        raw_text, image_urls = scrape_menu_text(url)
        if raw_text or image_urls:
            rest_id = get_place_id(restaurant_name, homepage=url) or f"dummy_{index}"
            structured_dishes = parse_text_to_json_with_llm(restaurant_name, rest_id, raw_text, image_urls)
            print(f"  -> Successfully structured {len(structured_dishes)} dishes.")
            retried_menus.extend(structured_dishes)
        else:
            print("  -> No text or images extracted.")
            
    # Save the retried results into a standalone new JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(retried_menus, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Retried pipeline complete. Extracted {len(retried_menus)} new dishes.")
    print(f"✅ Saved to: {output_path}")

if __name__ == "__main__":
    main()

