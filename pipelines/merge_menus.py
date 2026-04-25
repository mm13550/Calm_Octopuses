"""
pipelines/merge_menus.py
========================
Merges multiple partial menu JSON files into a single deduplicated
``data/extracted_menus/final_parsed_menus.json``.

Run this after any batch of ``menu_crawler.py`` runs to consolidate results.

Usage::

    python pipelines/merge_menus.py
"""
import os
import json
import pandas as pd

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    csv_path = os.path.join(data_dir, 'csv', 'seeds_resolved.csv')
    parsed_menus_path = os.path.join(data_dir, 'extracted_menus', 'parsed_menus.json')
    final_menus_path = os.path.join(data_dir, 'extracted_menus', 'final_parsed_menus.json')
    retried_menus_path = os.path.join(data_dir, 'extracted_menus', 'retried_menus.json')
    output_path = final_menus_path

    print(f"Reading restaurant order from {csv_path}...")
    df = pd.read_csv(csv_path)
    ordered_names = df['name'].tolist()

    # 1. Load raw crawled dishes
    dishes = []
    for file_path, name in [
        (parsed_menus_path, 'parsed_menus.json'),
        (final_menus_path, 'final_parsed_menus.json'),
        (retried_menus_path, 'retried_menus.json')
    ]:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    loaded = json.load(f)
                    dishes.extend(loaded)
                    print(f"Loaded {len(loaded)} dishes from {name}.")
                except json.JSONDecodeError:
                    print(f"Could not decode {name} (might be empty).")
                
    # Group by restaurant name
    grouped_dishes = {}
    for dish in dishes:
        r_name = dish.get('restaurant_name', 'Unknown')
        if r_name not in grouped_dishes:
            grouped_dishes[r_name] = []
        # Minor deduplication by dish name just in case of overlaps
        existing_dish_names = [d.get('dish_name') for d in grouped_dishes[r_name]]
        if dish.get('dish_name') not in existing_dish_names:
            grouped_dishes[r_name].append(dish)
            
    # Rebuild a flat array in exact CSV order
    ordered_flat_menus = []
    found_count = 0
    missing_count = 0
    
    for name in ordered_names:
        if name in grouped_dishes and len(grouped_dishes[name]) > 0:
            ordered_flat_menus.extend(grouped_dishes[name])
            found_count += 1
        else:
            missing_count += 1

    # Save to a new file safely
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ordered_flat_menus, f, indent=4, ensure_ascii=False)
        
    print("\n--- Summary ---")
    print(f"Total structured dishes written: {len(ordered_flat_menus)}")
    print(f"Restaurants with at least 1 dish: {found_count}")
    print(f"Restaurants with 0 dishes (truly missing or inaccessible): {missing_count}")
    print(f"\n✅ Merged correctly and exported to: {output_path}")

if __name__ == "__main__":
    main()

