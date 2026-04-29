"""
pipelines/audit_restaurant_lookup.py
=====================================
Data quality auditor for the restaurant catalog and menus.

Scans the canonical lookup table and menu JSONs for:
1. Missing or inconsistent fields.
2. Duplicate IDs and domain mismatches.
3. Incomplete menu coverage or alignment issues.

Usage::

    python pipelines/audit_restaurant_lookup.py
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOOKUP_CSV = DATA_DIR / "csv" / "restaurant_lookup.csv"
FINAL_MENUS_JSON = DATA_DIR / "extracted_menus" / "final_parsed_menus.json"
PARSED_MENUS_JSON = DATA_DIR / "extracted_menus" / "parsed_menus.json"
DEFAULT_OUTPUT = DATA_DIR / "csv" / "restaurant_lookup_audit.csv"


def normalize_text(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def normalize_domain(url: str) -> str:
    host = urlparse(str(url or "")).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def significant_tokens(value: str) -> list[str]:
    return [token for token in normalize_text(value).split() if len(token) >= 4]


def load_lookup(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_menu_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def audit_lookup_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for row in rows:
        name = row.get("name", "")
        title = row.get("candidate_title", "")
        homepage = row.get("homepage", "")
        source = row.get("bio_match_source", "")
        name_tokens = significant_tokens(name)
        title_text = normalize_text(title)
        host_text = normalize_domain(homepage)

        reasons: list[str] = []
        severity = 0

        if source == "places_api_disambiguated_menu_conflict":
            reasons.append("lookup_rest_id_conflicts_with_menu_place_result")
            severity += 3
        elif source == "places_api_missing_from_menu":
            reasons.append("menu_pipeline_did_not_recover_place_id")
            severity += 1

        if name_tokens and not any(token in title_text for token in name_tokens):
            reasons.append("candidate_title_missing_name_tokens")
            severity += 2

        if name_tokens and host_text and not any(token in host_text for token in name_tokens):
            reasons.append("homepage_domain_missing_name_tokens")
            severity += 1

        if any(token in normalize_text(name) for token in ["restaurant", "cafe", "house", "club", "room", "kitchen"]):
            reasons.append("generic_name_high_ambiguity")
            severity += 1

        if severity == 0:
            continue

        findings.append(
            {
                "issue_type": "lookup_row",
                "severity": str(severity),
                "name": name,
                "rest_id": row.get("rest_id", ""),
                "related_rest_id": "",
                "bio_match_source": source,
                "homepage": homepage,
                "candidate_title": title,
                "reason": "; ".join(reasons),
            }
        )
    return findings


def audit_menu_rows(
    lookup_rows: list[dict[str, str]],
    menu_rows: list[dict],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    lookup_by_name = {row.get("name", ""): row for row in lookup_rows if row.get("name")}
    lookup_ids = {row.get("rest_id", "") for row in lookup_rows if row.get("rest_id")}

    bad_menu_rest_ids = Counter()
    bad_menu_examples: dict[str, dict] = {}
    for row in menu_rows:
        rest_id = str(row.get("rest_id", "")).strip()
        if rest_id and rest_id not in lookup_ids:
            bad_menu_rest_ids[rest_id] += 1
            bad_menu_examples.setdefault(rest_id, row)

    for bad_rest_id, count in bad_menu_rest_ids.most_common():
        example = bad_menu_examples[bad_rest_id]
        findings.append(
            {
                "issue_type": "menu_rest_id_not_in_lookup",
                "severity": "4",
                "name": str(example.get("restaurant_name", "")),
                "rest_id": bad_rest_id,
                "related_rest_id": lookup_by_name.get(str(example.get("restaurant_name", "")), {}).get("rest_id", ""),
                "bio_match_source": "",
                "homepage": "",
                "candidate_title": "",
                "reason": f"{count} menu rows use a rest_id that is absent from restaurant_lookup.csv",
            }
        )

    menu_ids_by_name: dict[str, set[str]] = defaultdict(set)
    for row in menu_rows:
        restaurant_name = str(row.get("restaurant_name", "")).strip()
        rest_id = str(row.get("rest_id", "")).strip()
        if restaurant_name and rest_id:
            menu_ids_by_name[restaurant_name].add(rest_id)

    for restaurant_name, menu_rest_ids in sorted(menu_ids_by_name.items()):
        lookup_row = lookup_by_name.get(restaurant_name)
        if not lookup_row:
            continue
        lookup_rest_id = lookup_row.get("rest_id", "")
        if len(menu_rest_ids) > 1 or (lookup_rest_id and lookup_rest_id not in menu_rest_ids):
            findings.append(
                {
                    "issue_type": "menu_name_rest_id_mismatch",
                    "severity": "3",
                    "name": restaurant_name,
                    "rest_id": lookup_rest_id,
                    "related_rest_id": ",".join(sorted(menu_rest_ids)),
                    "bio_match_source": lookup_row.get("bio_match_source", ""),
                    "homepage": lookup_row.get("homepage", ""),
                    "candidate_title": lookup_row.get("candidate_title", ""),
                    "reason": "menu rows for this restaurant_name do not cleanly align to the canonical lookup rest_id",
                }
            )

    return findings


def write_findings(path: Path, findings: list[dict[str, str]]) -> None:
    fieldnames = [
        "issue_type",
        "severity",
        "name",
        "rest_id",
        "related_rest_id",
        "bio_match_source",
        "homepage",
        "candidate_title",
        "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(findings)


def parse_args() -> argparse.Namespace:
    default_menus_json = FINAL_MENUS_JSON if FINAL_MENUS_JSON.exists() else PARSED_MENUS_JSON
    parser = argparse.ArgumentParser(description="Audit canonical restaurant lookup and menu rest_id alignment.")
    parser.add_argument("--lookup-csv", default=str(LOOKUP_CSV))
    parser.add_argument("--menus-json", default=str(default_menus_json))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lookup_rows = load_lookup(Path(args.lookup_csv))
    menu_rows = load_menu_rows(Path(args.menus_json))

    findings = audit_lookup_rows(lookup_rows)
    findings.extend(audit_menu_rows(lookup_rows, menu_rows))
    findings.sort(key=lambda row: (-int(row["severity"]), row["issue_type"], row["name"]))

    write_findings(Path(args.output_csv), findings)
    print(f"Loaded {len(lookup_rows)} lookup rows and {len(menu_rows)} menu rows.")
    print(f"Wrote {len(findings)} findings to {args.output_csv}")

    severity_counts = Counter(row["severity"] for row in findings)
    if severity_counts:
        summary = ", ".join(f"severity {severity}: {count}" for severity, count in sorted(severity_counts.items(), reverse=True))
        print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

