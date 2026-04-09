import sys
import os
import json

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from pipelines.menu_crawler import scrape_menu_text, parse_text_to_json_with_llm

url = "https://www.lepavillonnyc.com/"
raw_text = scrape_menu_text(url)
print(f"Total Combined Text Length: {len(raw_text)}")

parsed_json = parse_text_to_json_with_llm("Le Pavillon", raw_text)
for item in parsed_json:
    print(f"- {item.get('dish_name')}")
print(f"Total dishes extracted: {len(parsed_json)}")
