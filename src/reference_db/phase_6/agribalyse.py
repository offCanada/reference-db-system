from __future__ import annotations

import pandas as pd


AGRIBALYSE_VERSION = "3.2"

MAPPING_VERSION = "phase6-taxonomy-v1"


# These are broad taxonomy mappings.
# They are NOT exact product-level Agribalyse matches.
#
# ciqual_family is intentionally omitted because the current
# Reference DB does not contain an exact Ciqual match.
#
# The official Agribalyse database contains individual food
# entries associated with Ciqual codes, but this pipeline currently
# performs only category-level mapping.

TAXONOMY_MAPPING = {
    "DAIRY": {
        "agribalyse_category": "Milk and dairy products",
        "confidence": "MEDIUM",
    },
    "MEAT_SEAFOOD": {
        "agribalyse_category": "Meat, eggs and fish",
        "confidence": "MEDIUM",
    },
    "PRODUCE": {
        "agribalyse_category": "Fruits, vegetables, legumes and oilseeds",
        "confidence": "MEDIUM",
    },
    "BAKERY": {
        "agribalyse_category": "Cereal products / bakery",
        "confidence": "MEDIUM",
    },
    "PASTA_RICE": {
        "agribalyse_category": "Cereal products",
        "confidence": "MEDIUM",
    },
    "BREAKFAST": {
        "agribalyse_category": "Cereal products",
        "confidence": "MEDIUM",
    },
    "CONFECTIONERY": {
        "agribalyse_category": "Sweet products",
        "confidence": "LOW",
    },
    "SNACKS": {
        "agribalyse_category": "Snack products",
        "confidence": "LOW",
    },
    "CONDIMENTS_SAUCES": {
        "agribalyse_category": "Condiments and sauces",
        "confidence": "LOW",
    },
    "OILS_VINEGARS": {
        "agribalyse_category": "Fats and oils",
        "confidence": "MEDIUM",
    },
    "FROZEN": {
        "agribalyse_category": "Frozen foods",
        "confidence": "LOW",
    },
    "GENERAL_GROCERY": {
        "agribalyse_category": "General grocery foods",
        "confidence": "LOW",
    },
    "BAKING_INGREDIENTS": {
        "agribalyse_category": "Baking ingredients",
        "confidence": "LOW",
    },
    "BABY_CARE": {
        "agribalyse_category": None,
        "confidence": "NONE",
    },
}


def map_taxonomy_to_agribalyse(
    taxonomy,
    domain,
) -> dict[str, object]:
    """Map Reference DB taxonomy to a broad Agribalyse category."""

    if domain != "food":
        return {
            "agribalyse_category": None,
            "mapping_method": "not_applicable",
            "mapping_confidence": "NONE",
            "mapping_status": "NOT_APPLICABLE",
            "mapping_note": "Non-food product.",
        }

    if pd.isna(taxonomy):
        return {
            "agribalyse_category": None,
            "mapping_method": "not_available",
            "mapping_confidence": "NONE",
            "mapping_status": "NOT_AVAILABLE",
            "mapping_note": "Taxonomy is unavailable.",
        }

    mapping = TAXONOMY_MAPPING.get(taxonomy)

    if mapping is None:
        return {
            "agribalyse_category": None,
            "mapping_method": "not_available",
            "mapping_confidence": "NONE",
            "mapping_status": "NOT_AVAILABLE",
            "mapping_note": "No deterministic taxonomy mapping defined.",
        }

    if mapping["agribalyse_category"] is None:
        return {
            "agribalyse_category": None,
            "mapping_method": "not_applicable",
            "mapping_confidence": "NONE",
            "mapping_status": "NOT_APPLICABLE",
            "mapping_note": "Category is excluded from Agribalyse mapping.",
        }

    return {
        "agribalyse_category": mapping["agribalyse_category"],
        "mapping_method": "taxonomy_mapping",
        "mapping_confidence": mapping["confidence"],
        "mapping_status": "MAPPED",
        "mapping_note": (
            "Broad taxonomy-level mapping; not an exact product-level "
            "Agribalyse or Ciqual match."
        ),
    }


def add_agribalyse_mapping(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    mapped = result.apply(
        lambda row: map_taxonomy_to_agribalyse(
            row.get("taxonomy"),
            row.get("domain"),
        ),
        axis=1,
    )

    result["agribalyse_category"] = mapped.map(
        lambda x: x["agribalyse_category"]
    )

    result["mapping_method"] = mapped.map(
        lambda x: x["mapping_method"]
    )

    result["mapping_confidence"] = mapped.map(
        lambda x: x["mapping_confidence"]
    )

    result["mapping_status"] = mapped.map(
        lambda x: x["mapping_status"]
    )

    result["mapping_note"] = mapped.map(
        lambda x: x["mapping_note"]
    )

    result["mapping_source"] = (
        "ADEME Agribalyse 3.2 category reference"
    )

    result["mapping_version"] = MAPPING_VERSION

    return result
