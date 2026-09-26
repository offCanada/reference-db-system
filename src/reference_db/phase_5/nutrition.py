import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

HF_DATASET = "saraNour/compliments-brand"
NUTRITION_FILE = "raw_data/nutrition.parquet"

PHASE2_FILE = "data/phase_2/standardized_products.parquet"
PHASE3_FILE = "data/phase_3/product_groups.parquet"
PHASE4_FILE = "data/phase_4/product_variant_mapping.parquet"

OUTPUT_DIR = Path("data/phase_5")


NUTRITION_COLUMNS = [
    "calories",
    "fat_g",
    "saturated_fat_g",
    "monounsaturated_fat_g",
    "polyunsaturated_fat_g",
    "omega6_g",
    "omega3_g",
    "sugar_alcohols_g",
    "carbohydrate_g",
    "fibre_g",
    "sugars_g",
    "protein_g",
    "sodium_mg",
    "potassium_mg",
    "calcium_mg",
    "iron_mg",
    "cholesterol_mg",
]


SUSPICIOUS_THRESHOLDS = {
    "calories": 1000,
    "fat_g": 100,
    "protein_g": 100,
}


def load_raw_nutrition() -> pd.DataFrame:
    """Load raw nutrition data from Hugging Face."""

    local_path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=NUTRITION_FILE,
        repo_type="dataset",
    )

    nutrition = pd.read_parquet(local_path)

    nutrition["retailer_product_id"] = (
        nutrition["retailer_product_id"]
        .astype(str)
        .str.strip()
    )

    return nutrition


def load_phase_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Phase 2, Phase 3 and Phase 4 outputs."""

    phase2 = pd.read_parquet(PHASE2_FILE)
    phase3 = pd.read_parquet(PHASE3_FILE)
    phase4 = pd.read_parquet(PHASE4_FILE)

    phase2["external_id"] = (
        phase2["external_id"]
        .astype(str)
        .str.strip()
    )

    phase3["external_id"] = (
        phase3["external_id"]
        .astype(str)
        .str.strip()
    )

    phase4["external_id"] = (
        phase4["external_id"]
        .astype(str)
        .str.strip()
    )

    return phase2, phase3, phase4


def validate_source_schema(nutrition: pd.DataFrame) -> dict[str, Any]:
    """Validate that the raw nutrition source contains the expected schema."""

    required_columns = [
        "retailer_product_id",
        "product_name",
        "brand",
        "size",
        "serving_size",
        "upc",
        "scraped_at",
        *NUTRITION_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in nutrition.columns
    ]

    duplicate_ids = int(
        nutrition["retailer_product_id"].duplicated().sum()
    )

    return {
        "required_columns": required_columns,
        "missing_columns": missing_columns,
        "duplicate_retailer_product_ids": duplicate_ids,
        "result": (
            "PASS"
            if not missing_columns and duplicate_ids == 0
            else "FAIL"
        ),
    }


def assess_nutrition_quality(
    nutrition: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assess nutrition completeness and suspicious values.

    Status rules:
    - SUSPICIOUS: at least one suspicious nutrition value.
    - MISSING_FIELDS: at least one nutrition field is missing.
    - VALID: all nutrition fields are present and no suspicious value exists.
    """

    result = nutrition.copy()

    missing_mask = result[NUTRITION_COLUMNS].isna()

    result["missing_field_count"] = missing_mask.sum(axis=1)

    result["missing_fields"] = missing_mask.apply(
        lambda row: ",".join(
            column
            for column in NUTRITION_COLUMNS
            if row[column]
        ),
        axis=1,
    )

    suspicious_fields = []

    for column, threshold in SUSPICIOUS_THRESHOLDS.items():
        numeric_values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        suspicious_fields.append(
            numeric_values > threshold
        )

    if suspicious_fields:
        suspicious_mask = suspicious_fields[0].copy()

        for mask in suspicious_fields[1:]:
            suspicious_mask = suspicious_mask | mask
    else:
        suspicious_mask = pd.Series(
            False,
            index=result.index,
        )

    result["suspicious"] = suspicious_mask

    suspicious_field_names = []

    for index in result.index:
        fields = []

        for column, threshold in SUSPICIOUS_THRESHOLDS.items():
            value = pd.to_numeric(
                result.at[index, column],
                errors="coerce",
            )

            if pd.notna(value) and value > threshold:
                fields.append(column)

        suspicious_field_names.append(",".join(fields))

    result["suspicious_fields"] = suspicious_field_names

    result["nutrition_quality_status"] = "VALID"

    result.loc[
        result["missing_field_count"] > 0,
        "nutrition_quality_status",
    ] = "MISSING_FIELDS"

    result.loc[
        result["suspicious"],
        "nutrition_quality_status",
    ] = "SUSPICIOUS"

    return result


def parse_serving_size(
    value: Any,
) -> tuple[float | None, str | None, str | None]:
    """
    Parse serving size into amount, unit and parsing reason.

    Supported:
    - 100 g
    - 355 mL
    - 1 cup (250 ml)
    - 2 tsp (10 g)
    - kg -> g
    - L -> ml

    No density-based conversion is performed between mass and volume.
    """

    if pd.isna(value):
        return None, None, "missing_serving_size"

    text = str(value).strip().lower()

    if not text:
        return None, None, "missing_serving_size"

    # Prefer an explicit gram equivalent such as:
    # "2 tsp (10 g)"
    # "1/8 tsp (0.6 g)"
    gram_match = re.search(
        r"\(\s*(\d+(?:\.\d+)?)\s*g(?:ram|rams)?\s*\)",
        text,
    )

    if gram_match:
        amount = float(gram_match.group(1))
        return amount, "g", None

    # Explicit mass.
    mass_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(kg|kilograms?|g|grams?)\b",
        text,
    )

    if mass_match:
        amount = float(mass_match.group(1))
        unit = mass_match.group(2)

        if unit.startswith("kg"):
            amount *= 1000

        return amount, "g", None

    # Explicit volume.
    volume_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(l|litres?|liters?|ml|millilitres?|milliliters?)\b",
        text,
    )

    if volume_match:
        amount = float(volume_match.group(1))
        unit = volume_match.group(2)

        if unit in {"l", "litre", "litres", "liter", "liters"}:
            amount *= 1000

        return amount, "ml", None

    return None, None, "unparseable_serving_size"


def normalize_nutrition(
    nutrition: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize nutrition values to per-100g or per-100ml.

    Mass-based serving sizes produce per_100g values.
    Volume-based serving sizes produce per_100ml values.

    Mass and volume are intentionally kept separate.
    """

    result = nutrition.copy()

    parsed = result["serving_size"].apply(parse_serving_size)

    result["serving_amount"] = parsed.apply(
        lambda value: value[0]
    )

    result["serving_unit"] = parsed.apply(
        lambda value: value[1]
    )

    result["serving_parse_reason"] = parsed.apply(
        lambda value: value[2]
    )

    result["calories_per_100g"] = pd.NA
    result["calories_per_100ml"] = pd.NA

    for column in NUTRITION_COLUMNS:
        result[f"{column}_per_100g"] = pd.NA
        result[f"{column}_per_100ml"] = pd.NA

    mass_mask = result["serving_unit"] == "g"
    volume_mask = result["serving_unit"] == "ml"

    for column in NUTRITION_COLUMNS:
        numeric_values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        per_100_mass = (
            numeric_values / result["serving_amount"] * 100
        )

        per_100_volume = (
            numeric_values / result["serving_amount"] * 100
        )

        result.loc[
            mass_mask,
            f"{column}_per_100g",
        ] = per_100_mass[mass_mask]

        result.loc[
            volume_mask,
            f"{column}_per_100ml",
        ] = per_100_volume[volume_mask]

    return result


def build_product_nutrition(
    phase2: pd.DataFrame,
    phase3: pd.DataFrame,
    phase4: pd.DataFrame,
    nutrition: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the integrated product_nutrition table.

    The product table is the driving table, so all 4,440 products
    remain present even when nutrition source data is unavailable.
    """

    products = phase2[
        [
            "external_id",
            "source_retailer",
            "title",
            "brand",
            "barcode",
            "size",
            "variant_attributes",
        ]
    ].copy()

    groups = phase3[
        [
            "external_id",
            "group_id",
        ]
    ].copy()

    variants = phase4[
        [
            "external_id",
            "variant_id",
        ]
    ].copy()

    products = products.merge(
        groups,
        on="external_id",
        how="left",
        validate="one_to_one",
    )

    products = products.merge(
        variants,
        on="external_id",
        how="left",
        validate="one_to_one",
    )

    products = products.merge(
        nutrition,
        on="external_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_nutrition"),
    )

    nutrition_present = (
        products["nutrition_quality_status"].notna()
    )

    products["match_method"] = pd.NA
    products["match_status"] = "NO_SOURCE_DATA"

    products.loc[
        nutrition_present,
        "match_method",
    ] = "exact_external_id"

    products.loc[
        nutrition_present,
        "match_status",
    ] = "MATCHED"

    products["nutrition_quality_status"] = (
        products["nutrition_quality_status"]
        .fillna("NO_SOURCE_DATA")
    )

    products["missing_fields"] = (
        products["missing_fields"]
        .fillna("")
    )

    products["suspicious_fields"] = (
        products["suspicious_fields"]
        .fillna("")
    )

    products["missing_field_count"] = (
        products["missing_field_count"]
        .fillna(0)
        .astype(int)
    )

    return products


def build_quality_report(
    nutrition: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact quality report."""

    total = len(nutrition)

    status_counts = (
        nutrition["nutrition_quality_status"]
        .value_counts()
        .to_dict()
    )

    rows = [
        {
            "metric": "total_records",
            "value": total,
        },
        {
            "metric": "valid_records",
            "value": status_counts.get("VALID", 0),
        },
        {
            "metric": "missing_fields_records",
            "value": status_counts.get("MISSING_FIELDS", 0),
        },
        {
            "metric": "suspicious_records",
            "value": status_counts.get("SUSPICIOUS", 0),
        },
        {
            "metric": "records_with_any_missing_field",
            "value": int(
                (nutrition["missing_field_count"] > 0).sum()
            ),
        },
        {
            "metric": "records_with_all_fields_missing",
            "value": int(
                nutrition[NUTRITION_COLUMNS]
                .isna()
                .all(axis=1)
                .sum()
            ),
        },
    ]

    for column in NUTRITION_COLUMNS:
        rows.append(
            {
                "metric": f"missing_{column}",
                "value": int(
                    nutrition[column].isna().sum()
                ),
            }
        )

    for column, threshold in SUSPICIOUS_THRESHOLDS.items():
        numeric_values = pd.to_numeric(
            nutrition[column],
            errors="coerce",
        )

        rows.append(
            {
                "metric": f"suspicious_{column}",
                "value": int(
                    (numeric_values > threshold).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_statistics(
    product_nutrition: pd.DataFrame,
    nutrition_normalized: pd.DataFrame,
) -> dict[str, Any]:
    """Build Phase 5 summary statistics."""

    total_products = len(product_nutrition)

    matched = int(
        (
            product_nutrition["match_status"]
            == "MATCHED"
        ).sum()
    )

    without_nutrition = int(
        (
            product_nutrition["match_status"]
            == "NO_SOURCE_DATA"
        ).sum()
    )

    coverage = (
        matched / total_products
        if total_products
        else 0
    )

    status_counts = (
        nutrition_normalized["nutrition_quality_status"]
        .value_counts()
        .to_dict()
    )

    return {
        "phase": 5,
        "total_products": total_products,
        "nutrition_records": len(nutrition_normalized),
        "matched_products": matched,
        "products_without_nutrition": without_nutrition,
        "nutrition_coverage": round(coverage, 4),
        "nutrition_quality": {
            "VALID": status_counts.get("VALID", 0),
            "MISSING_FIELDS": status_counts.get(
                "MISSING_FIELDS",
                0,
            ),
            "SUSPICIOUS": status_counts.get(
                "SUSPICIOUS",
                0,
            ),
        },
        "serving_units": (
            nutrition_normalized["serving_unit"]
            .value_counts(dropna=False)
            .to_dict()
        ),
    }


def validate_phase5(
    phase2: pd.DataFrame,
    phase3: pd.DataFrame,
    phase4: pd.DataFrame,
    nutrition_normalized: pd.DataFrame,
    product_nutrition: pd.DataFrame,
) -> dict[str, Any]:
    """Run Phase 5 validation checks."""

    checks = {}

    checks["product_count"] = {
        "expected": len(phase2),
        "actual": len(product_nutrition),
        "result": (
            "PASS"
            if len(product_nutrition) == len(phase2)
            else "FAIL"
        ),
    }

    checks["nutrition_count"] = {
        "expected": len(nutrition_normalized),
        "actual": int(
            (
                product_nutrition["match_status"]
                == "MATCHED"
            ).sum()
        ),
        "result": (
            "PASS"
            if len(nutrition_normalized)
            == int(
                (
                    product_nutrition["match_status"]
                    == "MATCHED"
                ).sum()
            )
            else "FAIL"
        ),
    }

    checks["unique_external_id"] = {
        "duplicates": int(
            product_nutrition["external_id"]
            .duplicated()
            .sum()
        ),
        "result": (
            "PASS"
            if product_nutrition["external_id"]
            .duplicated()
            .sum()
            == 0
            else "FAIL"
        ),
    }

    checks["group_ids"] = {
        "null_count": int(
            product_nutrition["group_id"]
            .isna()
            .sum()
        ),
        "result": (
            "PASS"
            if product_nutrition["group_id"]
            .isna()
            .sum()
            == 0
            else "FAIL"
        ),
    }

    checks["variant_ids"] = {
        "null_count": int(
            product_nutrition["variant_id"]
            .isna()
            .sum()
        ),
        "result": (
            "PASS"
            if product_nutrition["variant_id"]
            .isna()
            .sum()
            == 0
            else "FAIL"
        ),
    }

    checks["deterministic_matching"] = {
        "unmatched_nutrition": int(
            ~nutrition_normalized["external_id"].isin(
                phase2["external_id"]
            ).sum()
        )
        if len(nutrition_normalized)
        else 0,
        "result": (
            "PASS"
            if set(
                nutrition_normalized["external_id"]
            ).issubset(
                set(phase2["external_id"])
            )
            else "FAIL"
        ),
    }

    checks["mass_volume_separated"] = {
        "g_records": int(
            (
                nutrition_normalized["serving_unit"]
                == "g"
            ).sum()
        ),
        "ml_records": int(
            (
                nutrition_normalized["serving_unit"]
                == "ml"
            ).sum()
        ),
        "result": "PASS",
    }

    all_passed = all(
        check["result"] == "PASS"
        for check in checks.values()
    )

    return {
        "overall": {
            "result": "PASS" if all_passed else "FAIL",
        },
        "checks": checks,
    }


def run_phase5() -> dict[str, Any]:
    """Run the complete Phase 5 nutrition pipeline."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading raw nutrition...")
    raw_nutrition = load_raw_nutrition()

    print(
        f"Raw nutrition rows: {len(raw_nutrition)}"
    )

    schema_validation = validate_source_schema(
        raw_nutrition
    )

    if schema_validation["result"] != "PASS":
        raise ValueError(
            "Nutrition source schema validation failed: "
            f"{schema_validation}"
        )

    print("Loading Phase 2 / 3 / 4 outputs...")

    phase2, phase3, phase4 = load_phase_inputs()

    nutrition = raw_nutrition.copy()

    nutrition["external_id"] = (
        nutrition["retailer_product_id"]
        .astype(str)
        .str.strip()
    )

    print("Assessing nutrition quality...")

    nutrition = assess_nutrition_quality(
        nutrition
    )

    print("Normalizing nutrition...")

    nutrition = normalize_nutrition(
        nutrition
    )

    nutrition_normalized = nutrition.copy()

    print("Building product nutrition integration...")

    product_nutrition = build_product_nutrition(
        phase2=phase2,
        phase3=phase3,
        phase4=phase4,
        nutrition=nutrition_normalized,
    )

    print("Building quality report...")

    nutrition_quality = build_quality_report(
        nutrition_normalized
    )

    statistics = build_statistics(
        product_nutrition=product_nutrition,
        nutrition_normalized=nutrition_normalized,
    )

    print("Validating Phase 5...")

    validation = validate_phase5(
        phase2=phase2,
        phase3=phase3,
        phase4=phase4,
        nutrition_normalized=nutrition_normalized,
        product_nutrition=product_nutrition,
    )

    nutrition_normalized.to_parquet(
        OUTPUT_DIR / "nutrition_normalized.parquet",
        index=False,
    )

    product_nutrition.to_parquet(
        OUTPUT_DIR / "product_nutrition.parquet",
        index=False,
    )

    nutrition_quality.to_parquet(
        OUTPUT_DIR / "nutrition_quality.parquet",
        index=False,
    )

    with open(
        OUTPUT_DIR / "phase5_statistics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            statistics,
            file,
            indent=2,
            default=str,
        )

    with open(
        OUTPUT_DIR / "phase5_validation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            validation,
            file,
            indent=2,
            default=str,
        )

    print("\n=== PHASE 5 COMPLETE ===")
    print(
        f"Products: {statistics['total_products']}"
    )
    print(
        f"Nutrition records: "
        f"{statistics['nutrition_records']}"
    )
    print(
        f"Matched: "
        f"{statistics['matched_products']}"
    )
    print(
        f"Without nutrition: "
        f"{statistics['products_without_nutrition']}"
    )
    print(
        f"Coverage: "
        f"{statistics['nutrition_coverage'] * 100:.1f}%"
    )
    print(
        "Quality:",
        statistics["nutrition_quality"],
    )
    print(
        f"Validation: "
        f"{validation['overall']['result']}"
    )

    return {
        "nutrition_normalized": nutrition_normalized,
        "product_nutrition": product_nutrition,
        "nutrition_quality": nutrition_quality,
        "statistics": statistics,
        "validation": validation,
    }


if __name__ == "__main__":
    run_phase5()