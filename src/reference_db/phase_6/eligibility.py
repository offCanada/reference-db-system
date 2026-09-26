from __future__ import annotations

import pandas as pd

REQUIRED_NUTRI_SCORE_FIELDS = [
    "calories_per_100g",
    "sugars_g_per_100g",
    "saturated_fat_g_per_100g",
    "sodium_mg_per_100g",
    "fibre_g_per_100g",
    "protein_g_per_100g",
]

EXCLUDED_TAXONOMIES = {
    "BABY_CARE",
}

EXCLUDED_DOMAINS = {
    "non_food",
}


def _is_missing(value) -> bool:
    return pd.isna(value)


def check_score_eligibility(row: pd.Series) -> tuple[bool, str | None]:
    """Determine whether a product can enter the Nutri-Score calculation."""

    domain = row.get("domain")
    taxonomy = row.get("taxonomy")
    nutrition_status = row.get("nutrition_quality_status")

    if domain in EXCLUDED_DOMAINS:
        return False, "NON_FOOD"

    if pd.isna(domain) or domain == "unknown":
        return False, "UNKNOWN_DOMAIN"

    if taxonomy in EXCLUDED_TAXONOMIES:
        return False, "NUTRI_SCORE_EXCLUDED_CATEGORY"

    if nutrition_status == "NO_SOURCE_DATA":
        return False, "NO_SOURCE_DATA"

    missing_required = [
        field
        for field in REQUIRED_NUTRI_SCORE_FIELDS
        if _is_missing(row.get(field))
    ]

    if missing_required:
        return False, "MISSING_REQUIRED_FIELDS"

    suspicious_fields = row.get("suspicious_fields")

    if isinstance(suspicious_fields, (list, tuple, set)):
        relevant_suspicious = set(suspicious_fields) & {
            "calories",
            "sugars_g",
            "saturated_fat_g",
            "sodium_mg",
            "fibre_g",
            "protein_g",
        }

        if relevant_suspicious:
            return False, "SUSPICIOUS_REQUIRED_INPUT"

    return True, None


def add_eligibility_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    eligibility = result.apply(check_score_eligibility, axis=1)

    result["score_eligibility"] = eligibility.map(lambda x: x[0])
    result["score_exclusion_reason"] = eligibility.map(lambda x: x[1])

    return result
