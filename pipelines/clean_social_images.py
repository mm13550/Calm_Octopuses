"""
pipelines/clean_social_images.py
=================================
Quality-filters ``data/csv/social_images.csv`` using a zero-shot CLIP classifier
to remove non-food, non-interior images (e.g. screenshots, menus, logos).

Images below the quality threshold are marked for removal; the updated CSV
is written back in-place.

Usage::

    python pipelines/clean_social_images.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "pipelines" else Path.cwd()
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "csv" / "social_images.csv"
DEFAULT_LOOKUP = DATA_DIR / "csv" / "restaurant_lookup.csv"
DEFAULT_OUTPUT = DATA_DIR / "csv" / "social_images_cleaned.csv"
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"

CATEGORY_PROMPTS: dict[str, list[str]] = {
    "food": [
        "a close-up photo of plated food at a restaurant",
        "a restaurant dish or meal",
        "sushi, pasta, steak, dessert, or plated cuisine",
    ],
    "interior": [
        "a restaurant interior or dining room",
        "restaurant ambiance, tables, chairs, bar, or decor",
        "inside a fine dining restaurant",
    ],
    "people": [
        "people dining in a restaurant",
        "a portrait or selfie of people",
        "a chef or diners in a restaurant",
    ],
    "text_or_menu": [
        "a menu, poster, flyer, receipt, or screenshot with text",
        "a QR code or a page mostly containing words",
        "text-heavy signage or restaurant menu board",
    ],
    "other_noise": [
        "a logo, icon, map screenshot, unrelated graphic, or noise",
        "a blurry or unusable image",
        "an outdoor storefront, building exterior, or unrelated scene",
    ],
}

TEXT_MENU_HINTS = {
    "menu", "qr", "receipt", "poster", "flyer", "scan", "screenshot", "text", "sign", "board"
}
NOISE_HINTS = {"logo", "icon", "map", "exterior", "storefront", "outside"}


@dataclass
class ImageQuality:
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    quality_score: float
    notes: list[str]


class ZeroShotImageClassifier:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.labels: list[str] = []
        self.prompts: list[str] = []
        for label, prompts in CATEGORY_PROMPTS.items():
            for prompt in prompts:
                self.labels.append(label)
                self.prompts.append(prompt)

    @torch.no_grad()
    def classify(self, image: Image.Image) -> tuple[str, dict[str, float]]:
        inputs = self.processor(
            text=self.prompts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu().numpy()

        grouped: dict[str, list[float]] = {}
        for label, prob in zip(self.labels, probs):
            grouped.setdefault(label, []).append(float(prob))

        label_scores = {label: float(np.mean(vals)) for label, vals in grouped.items()}
        best_label = max(label_scores.items(), key=lambda kv: kv[1])[0]
        return best_label, label_scores


def load_restaurant_lookup(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}

    cols = {c.lower(): c for c in df.columns}
    rest_col = cols.get("rest_id") or cols.get("restaurant_id") or cols.get("place_id")
    name_col = cols.get("name") or cols.get("restaurant_name")
    if not rest_col or not name_col:
        return {}

    out: dict[str, str] = {}
    for row in df.to_dict(orient="records"):
        rest_id = str(row.get(rest_col) or "").strip()
        name = str(row.get(name_col) or "").strip()
        if rest_id and name:
            out[rest_id] = name
    return out


def compute_quality(image: Image.Image) -> ImageQuality:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
    h, w = gray.shape[:2]

    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)

    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    sharpness = float((dx.mean() + dy.mean()) / 255.0)

    notes: list[str] = []
    score = 1.0

    pixels = w * h
    if pixels < 180 * 180:
        score -= 0.35
        notes.append("low_resolution")
    elif pixels < 320 * 320:
        score -= 0.15
        notes.append("medium_resolution")

    if brightness < 0.18:
        score -= 0.25
        notes.append("too_dark")
    elif brightness > 0.92:
        score -= 0.20
        notes.append("overexposed")

    if contrast < 0.08:
        score -= 0.15
        notes.append("low_contrast")

    if sharpness < 0.035:
        score -= 0.25
        notes.append("blurry")

    score = max(0.0, min(1.0, score))
    return ImageQuality(
        width=w,
        height=h,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        quality_score=round(score, 4),
        notes=notes,
    )


def resolve_image_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(str(raw_path).strip())
    if p.is_absolute():
        return p
    if p.exists():
        return p
    candidate = base_dir / p
    if candidate.exists():
        return candidate
    return candidate


def heuristic_override(predicted: str, raw_path: str, quality: ImageQuality) -> tuple[str, list[str]]:
    notes: list[str] = []
    lowered = raw_path.lower()

    if any(token in lowered for token in TEXT_MENU_HINTS):
        predicted = "text_or_menu"
        notes.append("filename_text_hint")
    elif any(token in lowered for token in NOISE_HINTS):
        predicted = "other_noise"
        notes.append("filename_noise_hint")

    if quality.quality_score < 0.25 and predicted not in {"text_or_menu", "other_noise"}:
        predicted = "other_noise"
        notes.append("quality_override_noise")

    return predicted, notes


def keep_flags(category: str, quality_score: float) -> tuple[bool, bool]:
    keep_food = category == "food" and quality_score >= 0.35
    keep_ambiance = category == "interior" and quality_score >= 0.35
    return keep_food, keep_ambiance


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and label csv/social_images.csv into csv/social_images_cleaned.csv")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--lookup-csv", default=str(DEFAULT_LOOKUP))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick testing")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    required = {"image_uid", "rest_id", "source", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"csv/social_images.csv missing required columns: {sorted(missing)}")

    lookup = load_restaurant_lookup(Path(args.lookup_csv))
    classifier = ZeroShotImageClassifier(model_id=args.model_id)

    rows: list[dict[str, Any]] = []
    missing_images = 0
    failed_images = 0

    for record in df.to_dict(orient="records"):
        image_uid = str(record.get("image_uid") or "").strip()
        rest_id = str(record.get("rest_id") or "").strip()
        source = str(record.get("source") or "").strip()
        raw_image_path = str(record.get("image_path") or "").strip()
        restaurant_name = lookup.get(rest_id, rest_id)

        full_path = resolve_image_path(raw_image_path, BASE_DIR)
        notes: list[str] = []

        if not full_path.exists():
            missing_images += 1
            rows.append({
                "image_uid": image_uid,
                "rest_id": rest_id,
                "restaurant_name": restaurant_name,
                "source": source,
                "image_path": raw_image_path,
                "image_category": "other_noise",
                "keep_for_food_embedding": False,
                "keep_for_ambiance_embedding": False,
                "quality_score": 0.0,
                "notes": "file_missing",
            })
            continue

        try:
            with Image.open(full_path) as img:
                image = img.convert("RGB")
                quality = compute_quality(image)
                predicted, label_scores = classifier.classify(image)
        except Exception as exc:
            failed_images += 1
            rows.append({
                "image_uid": image_uid,
                "rest_id": rest_id,
                "restaurant_name": restaurant_name,
                "source": source,
                "image_path": raw_image_path,
                "image_category": "other_noise",
                "keep_for_food_embedding": False,
                "keep_for_ambiance_embedding": False,
                "quality_score": 0.0,
                "notes": f"image_open_error:{type(exc).__name__}",
            })
            continue

        predicted, heuristic_notes = heuristic_override(predicted, raw_image_path, quality)
        notes.extend(quality.notes)
        notes.extend(heuristic_notes)
        notes.append(
            "scores=" + ",".join(f"{k}:{label_scores[k]:.3f}" for k in sorted(label_scores.keys()))
        )

        keep_food, keep_ambiance = keep_flags(predicted, quality.quality_score)

        rows.append({
            "image_uid": image_uid,
            "rest_id": rest_id,
            "restaurant_name": restaurant_name,
            "source": source,
            "image_path": raw_image_path,
            "image_category": predicted,
            "keep_for_food_embedding": keep_food,
            "keep_for_ambiance_embedding": keep_ambiance,
            "quality_score": quality.quality_score,
            "notes": "; ".join(notes),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Loaded {len(df)} raw image rows from: {input_path}")
    print(f"Wrote cleaned image metadata to: {output_path}")
    print(f"Missing image files: {missing_images}")
    print(f"Failed image opens: {failed_images}")
    print("Category counts:")
    print(out_df["image_category"].value_counts(dropna=False).to_string())
    print("\nKeep-for-food count:", int(out_df["keep_for_food_embedding"].sum()))
    print("Keep-for-ambiance count:", int(out_df["keep_for_ambiance_embedding"].sum()))


if __name__ == "__main__":
    main()
