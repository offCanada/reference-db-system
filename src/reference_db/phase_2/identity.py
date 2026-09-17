from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


HF_DATASET = "saraNour/compliments-reference-db"

INPUT_FILE = "phase_1/validated_products.parquet"
OUTPUT_FILE = "phase_2/standardized_products.parquet"
STATISTICS_FILE = "phase_2/statistics/identity_statistics.json"
VALIDATION_FILE = "phase_2/validation/validation_report.json"

VERSION = "4.1.0"
TIMESTAMP = datetime.now(timezone.utc).isoformat()

REQUIRED_COLUMNS = [
    "external_id",
    "source_retailer",
    "brand",
    "title",
]

IDENTITY_FLAG_COLUMNS = [
    "is_organic",
    "is_gluten_free",
    "is_naturally_simple",
    "is_sugar_free",
    "is_unsalted",
    "is_lactose_free",
    "is_peanut_free",
    "is_plant_based",
    "is_reduced_sodium",
]

BRAND_MAP = {
    "compliments organic": ("Compliments", "Organic"),
    "compliments balance": ("Compliments", "Balance"),
    "compliments naturally simple": ("Compliments", "Naturally Simple"),
    "compliments green care": ("Compliments", "Green"),
    "compliments green": ("Compliments", "Green"),
    "compliments little ones": ("Compliments", "Little Ones"),
    "sensations": ("Sensations", "Sensations"),
    "compliments": ("Compliments", "Core"),
    "compliments ": ("Compliments", "Core"),
    " compliments": ("Compliments", "Core"),
}

FLAVOUR_KEYWORDS = [
    "almond",
    "apple",
    "banana",
    "blueberry",
    "caramel",
    "cherry",
    "chocolate",
    "cinnamon",
    "coconut",
    "cranberry",
    "honey",
    "lemon",
    "lime",
    "mango",
    "maple",
    "mixed berry",
    "peach",
    "peanut",
    "peppermint",
    "pineapple",
    "pomegranate",
    "raspberry",
    "strawberry",
    "tropical",
    "vanilla",
    "watermelon",
    "white chocolate",
    "berry",
    "espresso",
    "butterscotch",
    "toffee",
]

FLAVOUR_PLURAL_PATTERNS = {
    "berry": r"berr(?:y|ies)",
    "cherry": r"cherr(?:y|ies)",
    "peach": r"peach(?:es)?",
    "mango": r"mang(?:o|os)",
}

FORMULATION_KEYWORDS = [
    "smooth",
    "crunchy",
    "creamy",
    "chunky",
    "whole",
    "halves",
    "sliced",
    "ground",
    "chopped",
    "breaded",
    "fresh",
    "frozen",
    "roasted",
    "smoked",
]

BRAND_PREFIXES = [
    ("Compliments Naturally Simple ", ""),
    ("Compliments Balance ", ""),
    ("Compliments Organic ", ""),
    ("Compliments Green Care ", ""),
    ("Compliments Little Ones ", ""),
    ("Compliments ", ""),
    ("Sensations ", ""),
]

SIZE_PATTERNS = [
    r"\s+\d+\s*x\s+\d+\s*(?:g|kg|ml|l|oz|lb|feet|yards?)s?\s*$",
    r"\s+\d+\s*x\s+\d+\s*(?:tablets?|caplets?|capsules?|sheets?|bags?|rolls?)\s*$",
    r"\s+\d+\s*x\s*$",
    r"\s+\d+(?:\.\d+)?[\s-]*inch(?:es)?\s*x\s*\d+\s*(?:feet|yards?)s?\s*$",
    r"\s+\d+(?:\.\d+)?[\s-]*inch(?:es)?\s*$",
    r"\s+\d+\s*(?:tablets?|caplets?|capsules?|sachets?|sticks?|bars?|rolls?|sheets?|bags?|bulbs?|lamps?|lozenges?|plugs?|pairs?|liners?|wipes?|strips?|sprays?|cots?|napkins?|boxes?|pods?)\s*$",
    r"\s+\d+\s*(?:tea\s+)?bags?\s*$",
    r"\s+\d+\s*(?:tea\s+)?sachets?\s*$",
    r"\s+\d+\s*(?:count|ea|pack|pieces?|slice|cups?)\s*$",
    r"\s+\d+\s+per\s+pack\s*$",
    r"\s+\d+\s+count\s*$",
    r"\s+\d[\d,.]*\s*(?:g|kg|ml|l|oz|lb|litre|liters?|pound)s?\s*$",
    r"\s+\d[\d,.]*\s*(?:feet|yards?)s?\s*$",
    r"\s+\d+(?:\.\d+)?[\s-]*(?:oz|ounce)s?\s*$",
]

BRACKET_PATTERN = r"\s*\([^)]*\)\s*"


def clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return re.sub(r"\s+", " ", value)


def normalize_text(value: Any) -> str | None:
    value = clean_text(value)

    if value is None:
        return None

    value = value.lower()
    value = value.replace("–", "-")
    value = value.replace("—", "-")

    return re.sub(r"\s+", " ", value).strip()


def normalize_brand(value: Any) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return ("Unknown", "Core")

    key = str(value).strip().lower()

    if key in BRAND_MAP:
        return BRAND_MAP[key]

    if "compliments" in key:
        return ("Compliments", "Core")

    if "sensations" in key:
        return ("Sensations", "Sensations")

    return (str(value).strip(), "Core")


def extract_identity_flags(title: Any) -> dict[str, bool]:
    title_normalized = normalize_text(title) or ""

    return {
        "is_organic": bool(
            re.search(r"\borganic\b", title_normalized)
        ),
        "is_gluten_free": bool(
            re.search(r"\bgluten[\s-]+free\b", title_normalized)
        ),
        "is_naturally_simple": bool(
            re.search(r"\bnaturally\s+simple\b", title_normalized)
        ),
        "is_sugar_free": bool(
            re.search(
                r"\bsugar[\s-]+free\b"
                r"|\b(?:no sugar added|unsweetened)\b"
                r"|\bzero\s+sugar\b",
                title_normalized,
            )
        ),
        "is_unsalted": bool(
            re.search(
                r"\bunsalted\b|\bno salt\b",
                title_normalized,
            )
        ),
        "is_lactose_free": bool(
            re.search(
                r"\blactose[\s-]+free\b",
                title_normalized,
            )
        ),
        "is_peanut_free": bool(
            re.search(
                r"\bpeanut[\s-]+free\b",
                title_normalized,
            )
        ),
        "is_plant_based": bool(
            re.search(
                r"\bplant[\s-]+based\b",
                title_normalized,
            )
        ),
        "is_reduced_sodium": bool(
            re.search(
                r"\breduced\s+sodium\b"
                r"|\blow\s+sodium\b"
                r"|\bno\s+salt\s+added\b",
                title_normalized,
            )
        ),
    }


def extract_fat_info(title: Any) -> dict[str, Any]:
    title_normalized = normalize_text(title) or ""

    fat_level = "regular"

    if re.search(
        r"\b(?:light|lite|reduced fat|low fat|lean)\b",
        title_normalized,
    ):
        fat_level = "reduced_fat"
    elif re.search(
        r"\bfat[\s-]+free\b",
        title_normalized,
    ):
        fat_level = "fat_free"

    fat_percentage = None

    pct_match = re.search(
        r"(\d+(?:\.\d+)?)\s*%",
        title_normalized,
    )

    if pct_match:
        value = float(pct_match.group(1))

        skip_context = any(
            word in title_normalized
            for word in [
                "cocoa",
                "alcohol",
                "isopropyl",
                "peanuts",
            ]
        )

        if value < 100 and not skip_context:
            fat_percentage = value

            if 0 < value <= 0.7:
                fat_level = "fat_free"

    return {
        "fat_level": fat_level,
        "fat_percentage": fat_percentage,
    }


def extract_flavours(title: Any) -> list[str]:
    title_normalized = normalize_text(title) or ""

    found = []

    for keyword in FLAVOUR_KEYWORDS:
        if " " in keyword:
            if keyword in title_normalized:
                found.append(keyword)
            continue

        if keyword in FLAVOUR_PLURAL_PATTERNS:
            pattern = FLAVOUR_PLURAL_PATTERNS[keyword]
        else:
            pattern = re.escape(keyword) + r"s?"

        if re.search(
            r"\b" + pattern + r"\b",
            title_normalized,
        ):
            found.append(keyword)

    return sorted(set(found))


def extract_formulation(title: Any) -> list[str]:
    title_normalized = normalize_text(title) or ""

    found = []

    for keyword in FORMULATION_KEYWORDS:
        if re.search(
            rf"\b{re.escape(keyword)}\b",
            title_normalized,
        ):
            found.append(keyword)

    return sorted(set(found))


def extract_functional_variant(title: Any) -> str | None:
    title_normalized = normalize_text(title) or ""

    if re.search(r"\bdiapers?\b", title_normalized):
        match = re.search(
            r"\bsize\s+(\d+)\b",
            title_normalized,
        )

        if match:
            return f"size_{match.group(1)}"

    if re.search(
        r"\bbin\s+liners?\b|\bcompostable\s+bin\s+liners?\b",
        title_normalized,
    ):
        match = re.search(
            r"\b(small|tall)\b",
            title_normalized,
        )

        if match:
            return match.group(1)

    return None


def build_variant_attributes(
    size: Any,
    title: Any = None,
) -> dict[str, Any]:
    size_clean = clean_text(size)
    title_clean = clean_text(title)
    attributes: dict[str, Any] = {}

    variant_text = size_clean

    if variant_text is None and title_clean is not None:
        search_text = re.sub(r"\s*\([^)]*\)\s*$", "", title_clean).strip()

        patterns = [
            r"(\d+(?:\.\d+)?)\s*(litres?|liters?)\b",
            r"^\s*(\d+(?:\.\d+)?)\s*(litres?|liters?)\b",
            r"(\d+(?:\.\d+)?)\s*[-]?\s*(inch|inches)\b",
            r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|oz|lb|mg|litres?|liters?)\s*$",
            r"(\d+(?:\.\d+)?)\s*(g|kg|ml|l|oz|lb|mg|litres?|liters?)"
            r"(?:\s+(\d+)\s*(?:count|ea|packs?|pieces?|bags?|boxes?))?\s*$",
            r"(\d+(?:\.\d+)?)\s*(?:inch|inches)\s*x\s*"
            r"(\d+(?:\.\d+)?)\s*(feet|foot|yards?)\s*$",
            r"(\d+(?:\.\d+)?)\s*(?:inch|inches)\s*$",
            r"(\d+(?:\.\d+)?)\s*(feet|foot|yards?)s?\s*$",
            r"(\d+(?:\.\d+)?)\s*(m|meter|meters|cm|centimeter|centimeters)\s*$",
            r"(\d+(?:\.\d+)?)\s*(g|kg|ml|l|oz|lb|mg|litres?|liters?)\s+"
            r"(\d+)\s*(?:count|ea|packs?|pieces?|bags?|boxes?|"
            r"softgel\s+capsules?|capsules?|tablets?|pouches?)\s*$",
            r"(\d+)\s*(?:per\s+pack|count|ea|pack|piece|slice|cups?|"
            r"tablets?|caplets?|capsules?|sachets?|sticks?|bars?|"
            r"rolls?|sheets?|bags?|bulbs?|lamps?|lozenges?|plugs?|"
            r"pairs?|liners?|wipes?|strips?|sprays?|cots?|"
            r"napkins?|boxes?|pods?|pouches?|cases?|"
            r"softgel\s+capsules?|tea\s+bags?|k-cups?|thighs?)s?\s*$",
            r"(\d+)\s+(?:sterile\s+)?(?:bandages?|pouches?|cases?|"
            r"k-cups?|thighs?|tests?|softgel\s+capsules?)\s*$",
            r"twin\s+pack\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                variant_text = match.group(0).strip()
                break

    if variant_text is not None:
        text = variant_text.strip()

        match = re.match(
            r"(\d+(?:\.\d+)?)\s*x\s*"
            r"(\d+(?:\.\d+)?)\s*"
            r"(g|kg|ml|l|oz|lb|mg|litres?|liters?)"
            r"(?:\s+(\d+)\s*(?:count|ea|packs?|pieces?|bags?|boxes?))?\s*$",
            text,
            re.IGNORECASE,
        )

        if match:
            unit = match.group(3).lower()

            if unit in {"litre", "litres", "liter", "liters"}:
                unit = "l"

            attributes = {
                "qty": int(float(match.group(1))),
                "amount": float(match.group(2)),
                "unit": unit,
            }

            if match.group(4):
                attributes["count"] = int(match.group(4))

        else:
            match = re.match(
                r"(\d+(?:\.\d+)?)\s*[-]?\s*"
                r"(g|kg|ml|l|oz|lb|mg|litres?|liters?)"
                r"(?:\s+(\d+)\s*(?:count|ea|packs?|pieces?|bags?|boxes?))?\s*$",
                text,
                re.IGNORECASE,
            )

            if match:
                unit = match.group(2).lower()

                if unit in {"litre", "litres", "liter", "liters"}:
                    unit = "l"

                attributes = {
                    "amount": float(match.group(1)),
                    "unit": unit,
                }

                if match.group(3):
                    attributes["count"] = int(match.group(3))

            else:
                match = re.match(
                    r"(\d+(?:\.\d+)?)\s*[-]?\s*"
                    r"(inch|inches|feet|foot|yards?|m|meter|meters|"
                    r"cm|centimeter|centimeters)\s*$",
                    text,
                    re.IGNORECASE,
                )

                if match:
                    unit = match.group(2).lower()

                    if unit in {"inches"}:
                        unit = "inch"
                    elif unit == "foot":
                        unit = "feet"

                    attributes = {
                        "amount": float(match.group(1)),
                        "unit": unit,
                    }

                else:
                    match = re.match(
                        r"(\d+(?:\.\d+)?)\s*"
                        r"(?:inch|inches)\s*x\s*"
                        r"(\d+(?:\.\d+)?)\s*"
                        r"(feet|foot|yards?)s?\s*$",
                        text,
                        re.IGNORECASE,
                    )

                    if match:
                        length_unit = match.group(3).lower()

                        if length_unit == "foot":
                            length_unit = "feet"

                        attributes = {
                            "width": float(match.group(1)),
                            "width_unit": "inch",
                            "length": float(match.group(2)),
                            "length_unit": length_unit,
                        }

                    else:
                        match = re.match(
                            r"(\d+(?:\.\d+)?)\s*"
                            r"(g|kg|ml|l|oz|lb|mg|litres?|liters?)\s+"
                            r"(\d+)\s*"
                            r"(?:count|ea|packs?|pieces?|bags?|boxes?|"
                            r"softgel\s+capsules?|capsules?|tablets?|pouches?)\s*$",
                            text,
                            re.IGNORECASE,
                        )

                        if match:
                            unit = match.group(2).lower()

                            if unit in {"litre", "litres", "liter", "liters"}:
                                unit = "l"

                            attributes = {
                                "amount": float(match.group(1)),
                                "unit": unit,
                                "count": int(match.group(3)),
                            }

                        else:
                            match = re.match(
                                r"(\d+)\s*"
                                r"(?:per\s+pack|count|ea|pack|piece|slice|cups?|"
                                r"tablets?|caplets?|capsules?|sachets?|sticks?|bars?|"
                                r"rolls?|sheets?|bags?|bulbs?|lamps?|lozenges?|plugs?|"
                                r"pairs?|liners?|wipes?|strips?|sprays?|cots?|"
                                r"napkins?|boxes?|pods?|pouches?|cases?|"
                                r"softgel\s+capsules?|tea\s+bags?|k-cups?|thighs?|"
                                r"sterile\s+bandages?|tests?)s?\s*$",
                                text,
                                re.IGNORECASE,
                            )

                            if match:
                                attributes = {
                                    "count": int(match.group(1)),
                                }

                            elif re.fullmatch(
                                r"twin\s+pack",
                                text,
                                re.IGNORECASE,
                            ):
                                attributes = {
                                    "count": 2,
                                }

                            else:
                                attributes = {
                                    "raw": text,
                                }

    functional_variant = extract_functional_variant(title)

    if functional_variant is not None:
        attributes["functional_variant"] = functional_variant

    return attributes

def extract_product_name(
    title: Any,
    brand: Any,
) -> str | None:
    title_clean = clean_text(title)
    brand_clean = clean_text(brand)

    if title_clean is None:
        return None

    if brand_clean is None:
        return title_clean

    if title_clean.lower().startswith(
        brand_clean.lower()
    ):
        product_name = title_clean[len(brand_clean):].strip()

        product_name = re.sub(
            r"^[\s\-|:/]+",
            "",
            product_name,
        )

        return product_name or title_clean

    return title_clean


def extract_core_title(
    title: Any,
    product_line: str,
) -> str:
    title_clean = clean_text(title)

    if title_clean is None:
        return ""

    text = title_clean

    for prefix, replacement in BRAND_PREFIXES:
        if text.lower().startswith(prefix.lower()):
            text = (
                replacement
                + text[len(prefix):]
            )
            break

    text = re.sub(
        BRACKET_PATTERN,
        " ",
        text,
    )

    previous = None

    while previous != text:
        previous = text

        for pattern in SIZE_PATTERNS:
            text = re.sub(
                pattern,
                "",
                text,
                flags=re.IGNORECASE,
            )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def normalize_title_for_grouping(
    title: Any,
) -> str:
    text = normalize_text(title)

    if text is None:
        return ""

    text = re.sub(
        r"\b(?:grams?|gram)\b",
        "g",
        text,
    )

    text = re.sub(
        r"\b(?:milliliters?|millilitre|mls?)\b",
        "ml",
        text,
    )

    text = re.sub(
        r"\b(?:kilograms?|kgs?)\b",
        "kg",
        text,
    )

    text = re.sub(
        r"(\d+)\s*(?:g|gram|grams)\b",
        r"\1g",
        text,
    )

    text = re.sub(
        r"(\d+)\s*(?:ml|milliliter|millilitre)\b",
        r"\1ml",
        text,
    )

    text = re.sub(
        r"\b(?:compliments|sensations)\b",
        "",
        text,
    )

    descriptive = {
        "organic",
        "natural",
        "simple",
        "light",
        "lean",
        "free",
        "reduced",
        "extra",
        "plus",
        "ultra",
        "premium",
        "classic",
        "original",
        "traditional",
        "new",
        "improved",
        "rich",
        "creamy",
        "smooth",
        "crunchy",
        "chunky",
        "old",
        "style",
        "flavour",
        "flavor",
        "artisan",
        "homestyle",
    }

    tokens = [
        word
        for word in text.split()
        if word not in descriptive
    ]

    packaging = {
        "pack",
        "bag",
        "box",
        "twin",
        "triple",
        "value",
        "club",
        "family",
        "size",
    }

    tokens = [
        word
        for word in tokens
        if word not in packaging
    ]

    tokens = [
        re.sub(
            r"[^a-z0-9]",
            "",
            token,
        )
        for token in tokens
    ]

    tokens = [
        token
        for token in tokens
        if len(token) > 1
    ]

    unique_tokens = sorted(set(tokens))

    return " ".join(unique_tokens)


def build_identity_hash(
    row: dict[str, Any],
) -> str:
    variant_attributes = row.get("variant_attributes")

    if isinstance(variant_attributes, str):
        try:
            variant_attributes = json.loads(
                variant_attributes
            )
        except json.JSONDecodeError:
            variant_attributes = {}

    if not isinstance(variant_attributes, dict):
        variant_attributes = {}

    payload = {
        "source_retailer": normalize_text(
            row.get("source_retailer")
        ),
        "brand_norm": normalize_text(
            row.get("brand_norm")
        ),
        "product_line": normalize_text(
            row.get("product_line")
        ),
        "core_title": normalize_title_for_grouping(
            row.get("core_title")
        ),
        "identity_flags": {
            column: bool(row.get(column))
            for column in IDENTITY_FLAG_COLUMNS
        },
        "fat_percentage": (
            float(row["fat_percentage"])
            if pd.notna(row.get("fat_percentage"))
            else None
        ),
        "fat_level": row.get(
            "fat_level",
            "regular",
        ),
        "flavour": sorted(
            row.get("flavour") or []
        ),
        "formulation": sorted(
            row.get("formulation") or []
        ),
        "functional_variant": variant_attributes.get(
            "functional_variant"
        ),
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def validate_input_schema(
    df: pd.DataFrame,
) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def transform_products(
    df: pd.DataFrame,
) -> pd.DataFrame:
    validate_input_schema(df)

    output = df.copy()

    identity_flags_list = []
    fat_info_list = []
    flavour_list = []
    formulation_list = []
    brand_norm_list = []
    product_line_list = []
    product_name_list = []
    core_title_list = []
    barcode_list = []
    size_list = []
    variant_attributes_list = []

    for _, row in df.iterrows():
        title = clean_text(
            row["title"]
        )

        brand = clean_text(
            row["brand"]
        )

        identity_flags = extract_identity_flags(
            title
        )

        fat_info = extract_fat_info(
            title
        )

        flavour = extract_flavours(
            title
        )

        formulation = extract_formulation(
            title
        )

        brand_norm, product_line = normalize_brand(
            brand
        )

        product_name = extract_product_name(
            title,
            brand,
        )

        core_title = extract_core_title(
            title,
            product_line,
        )

        barcode = clean_text(
            row.get(
                "barcode",
                row.get("upc"),
            )
        )

        size = clean_text(
            row.get("size")
        )

        variant_attributes = build_variant_attributes(
            row.get("size"),
            title,
        )

        identity_flags_list.append(
            identity_flags
        )

        fat_info_list.append(
            fat_info
        )

        flavour_list.append(
            flavour
        )

        formulation_list.append(
            formulation
        )

        brand_norm_list.append(
            brand_norm
        )

        product_line_list.append(
            product_line
        )

        product_name_list.append(
            product_name
        )

        core_title_list.append(
            core_title
        )

        barcode_list.append(
            barcode
        )

        size_list.append(
            size
        )

        variant_attributes_list.append(
            json.dumps(
                variant_attributes,
                sort_keys=True,
                ensure_ascii=False,
            )
        )

    identity_flags_df = pd.DataFrame(
        identity_flags_list,
        index=output.index,
    )

    for column in IDENTITY_FLAG_COLUMNS:
        if column in identity_flags_df.columns:
            output[column] = (
                identity_flags_df[column]
                .fillna(False)
                .astype(bool)
            )

    fat_info_df = pd.DataFrame(
        fat_info_list,
        index=output.index,
    )

    output["fat_level"] = (
        fat_info_df["fat_level"]
    )

    output["fat_percentage"] = (
        fat_info_df["fat_percentage"]
    )

    output["brand_norm"] = pd.Series(
        brand_norm_list,
        index=output.index,
    )

    output["product_line"] = pd.Series(
        product_line_list,
        index=output.index,
    )

    output["product_name"] = pd.Series(
        product_name_list,
        index=output.index,
    )

    output["core_title"] = pd.Series(
        core_title_list,
        index=output.index,
    )

    output["barcode"] = pd.Series(
        barcode_list,
        index=output.index,
    )

    output["size"] = pd.Series(
        size_list,
        index=output.index,
    )

    output["variant_attributes"] = pd.Series(
        variant_attributes_list,
        index=output.index,
    )

    output["flavour"] = pd.Series(
        flavour_list,
        index=output.index,
        dtype="object",
    )

    output["formulation"] = pd.Series(
        formulation_list,
        index=output.index,
        dtype="object",
    )

    output["identity_hash"] = output.apply(
        lambda row: build_identity_hash(
            row.to_dict()
        ),
        axis=1,
    )

    return output


def build_statistics(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
) -> dict[str, Any]:
    functional_variant_counts: dict[str, int] = {}

    for value in df_out["variant_attributes"]:
        try:
            attributes = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            attributes = {}

        functional_variant = attributes.get(
            "functional_variant"
        )

        if functional_variant is not None:
            functional_variant_counts[
                str(functional_variant)
            ] = (
                functional_variant_counts.get(
                    str(functional_variant),
                    0,
                )
                + 1
            )

    return {
        "version": VERSION,
        "timestamp": TIMESTAMP,
        "input": {
            "row_count": int(len(df_in)),
            "column_count": int(len(df_in.columns)),
        },
        "output": {
            "row_count": int(len(df_out)),
            "column_count": int(len(df_out.columns)),
            "columns_added": sorted(
                set(df_out.columns)
                - set(df_in.columns)
            ),
        },
        "brand_normalization": {
            "unique_brands": int(
                df_out["brand_norm"].nunique()
            ),
            "product_lines": {
                str(key): int(value)
                for key, value in (
                    df_out["product_line"]
                    .value_counts()
                    .items()
                )
            },
        },
        "identity_flags": {
            column: int(
                df_out[column].sum()
            )
            for column in IDENTITY_FLAG_COLUMNS
        },
        "fat_info": {
            "fat_level_distribution": {
                str(key): int(value)
                for key, value in (
                    df_out["fat_level"]
                    .value_counts()
                    .items()
                )
            },
            "fat_percentage_nulls": int(
                df_out["fat_percentage"].isna().sum()
            ),
            "fat_percentage_non_null": int(
                df_out["fat_percentage"].notna().sum()
            ),
        },
        "flavour": {
            "products_with_flavour": int(
                (
                    df_out["flavour"]
                    .apply(len)
                    > 0
                ).sum()
            ),
            "products_without_flavour": int(
                (
                    df_out["flavour"]
                    .apply(len)
                    == 0
                ).sum()
            ),
        },
        "formulation": {
            "products_with_formulation": int(
                (
                    df_out["formulation"]
                    .apply(len)
                    > 0
                ).sum()
            ),
            "products_without_formulation": int(
                (
                    df_out["formulation"]
                    .apply(len)
                    == 0
                ).sum()
            ),
        },
        "functional_variant": {
            "products_with_functional_variant": int(
                sum(functional_variant_counts.values())
            ),
            "distribution": functional_variant_counts,
        },
        "core_title": {
            "unique_count": int(
                df_out["core_title"].nunique()
            ),
            "empty_count": int(
                (
                    df_out["core_title"]
                    .fillna("")
                    .str.strip()
                    == ""
                ).sum()
            ),
        },
        "identity_hash": {
            "unique_count": int(
                df_out["identity_hash"].nunique()
            ),
            "empty_count": int(
                (
                    df_out["identity_hash"]
                    .fillna("")
                    .str.strip()
                    == ""
                ).sum()
            ),
        },
    }


def build_validation_report(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    checks["row_count"] = {
        "input": int(len(df_in)),
        "output": int(len(df_out)),
        "pass": len(df_in) == len(df_out),
    }

    missing_original = sorted(
        set(df_in.columns)
        - set(df_out.columns)
    )

    checks["original_columns_preserved"] = {
        "missing": missing_original,
        "pass": len(missing_original) == 0,
    }

    expected_columns = {
        "brand_norm",
        "product_line",
        *IDENTITY_FLAG_COLUMNS,
        "fat_level",
        "fat_percentage",
        "flavour",
        "formulation",
        "variant_attributes",
        "core_title",
        "identity_hash",
    }

    missing_new = sorted(
        expected_columns
        - set(df_out.columns)
    )

    checks["required_phase2_columns"] = {
        "expected": sorted(expected_columns),
        "missing": missing_new,
        "pass": len(missing_new) == 0,
    }

    duplicate_source_ids = int(
        df_out.duplicated(
            subset=[
                "source_retailer",
                "external_id",
            ]
        ).sum()
    )

    checks["source_identifier_uniqueness"] = {
        "duplicate_count": duplicate_source_ids,
        "pass": duplicate_source_ids == 0,
    }

    missing_external_id = int(
        df_out["external_id"]
        .isna()
        .sum()
    )

    checks["external_id"] = {
        "missing_count": missing_external_id,
        "pass": missing_external_id == 0,
    }

    empty_core_title = int(
        (
            df_out["core_title"]
            .fillna("")
            .str.strip()
            == ""
        ).sum()
    )

    checks["core_title"] = {
        "empty_count": empty_core_title,
        "pass": empty_core_title == 0,
    }

    empty_identity_hash = int(
        (
            df_out["identity_hash"]
            .fillna("")
            .str.strip()
            == ""
        ).sum()
    )

    checks["identity_hash"] = {
        "empty_count": empty_identity_hash,
        "pass": empty_identity_hash == 0,
    }

    invalid_fat_levels = sorted(
        set(
            df_out["fat_level"]
            .dropna()
            .unique()
        )
        - {
            "fat_free",
            "reduced_fat",
            "regular",
        }
    )

    checks["fat_level"] = {
        "invalid_values": invalid_fat_levels,
        "pass": len(invalid_fat_levels) == 0,
    }

    identity_flags_boolean = all(
        df_out[column].dtype == bool
        for column in IDENTITY_FLAG_COLUMNS
    )

    checks["identity_flags_boolean"] = {
        "pass": identity_flags_boolean,
    }

    invalid_variant_attributes = 0

    for value in df_out["variant_attributes"]:
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                invalid_variant_attributes += 1
        except (TypeError, json.JSONDecodeError):
            invalid_variant_attributes += 1

    checks["variant_attributes"] = {
        "invalid_count": invalid_variant_attributes,
        "pass": invalid_variant_attributes == 0,
    }

    checks["overall"] = {
        "result": (
            "PASS"
            if all(
                check.get("pass", True)
                for check in checks.values()
            )
            else "FAIL"
        )
    }

    return checks


def run_phase2(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    df_out = transform_products(
        df.copy()
    )

    validation = build_validation_report(
        df,
        df_out,
    )

    statistics = build_statistics(
        df,
        df_out,
    )

    return (
        df_out,
        validation,
        statistics,
    )


def load_from_huggingface() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=INPUT_FILE,
        repo_type="dataset",
    )

    return pd.read_parquet(
        local_path
    )


def save_outputs(
    df_out: pd.DataFrame,
    validation: dict[str, Any],
    statistics: dict[str, Any],
) -> None:
    output_path = Path(OUTPUT_FILE)
    statistics_path = Path(STATISTICS_FILE)
    validation_path = Path(VALIDATION_FILE)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_out.to_parquet(
        output_path,
        index=False,
    )

    with open(
        statistics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            statistics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        validation_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            validation,
            file,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    print("Starting Phase 2")

    df_input = load_from_huggingface()

    print(
        f"Loaded {len(df_input)} rows "
        f"with {len(df_input.columns)} columns"
    )

    df_output, validation, statistics = run_phase2(
        df_input
    )

    save_outputs(
        df_output,
        validation,
        statistics,
    )

    print(
        f"Phase 2 complete: "
        f"{len(df_output)} rows"
    )

    print(
        f"Validation: "
        f"{validation['overall']['result']}"
    )