from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
PRODUCTION_DIR = BASE_DIR / "data" / "production"


FILES = {
    "product_metadata": PRODUCTION_DIR / "product_metadata.parquet",
    "product_groups": PRODUCTION_DIR / "product_groups.parquet",
    "product_variants": PRODUCTION_DIR / "product_variants.parquet",
    "product_nutrition": PRODUCTION_DIR / "product_nutrition.parquet",
    "product_scores": PRODUCTION_DIR / "product_scores.parquet",
}


def load_production():
    return {
        name: pd.read_parquet(path)
        for name, path in FILES.items()
    }


def validate_production():
    tables = load_production()

    metadata = tables["product_metadata"]
    groups = tables["product_groups"]
    variants = tables["product_variants"]
    nutrition = tables["product_nutrition"]
    scores = tables["product_scores"]

    errors = []

    # 1. Product-level uniqueness

    for name, df in {
        "product_metadata": metadata,
        "product_nutrition": nutrition,
        "product_scores": scores,
    }.items():
        if "external_id" not in df.columns:
            errors.append(f"{name}: missing external_id")

        elif df["external_id"].duplicated().any():
            errors.append(
                f"{name}: duplicate external_id values"
            )

    # 2. Group-level uniqueness

    if groups["group_id"].duplicated().any():
        errors.append("product_groups: duplicate group_id values")

    # 3. Variant-level uniqueness

    if variants["variant_id"].duplicated().any():
        errors.append("product_variants: duplicate variant_id values")

    # 4. Product coverage

    metadata_ids = set(metadata["external_id"])
    nutrition_ids = set(nutrition["external_id"])
    score_ids = set(scores["external_id"])

    missing_nutrition = metadata_ids - nutrition_ids
    missing_scores = metadata_ids - score_ids

    if missing_nutrition:
        errors.append(
            f"Products missing from nutrition: {len(missing_nutrition)}"
        )

    if missing_scores:
        errors.append(
            f"Products missing from scores: {len(missing_scores)}"
        )

    # 5. Group coverage

    group_ids = set(groups["group_id"])

    for name, df in {
        "product_metadata": metadata,
        "product_nutrition": nutrition,
        "product_scores": scores,
    }.items():
        if "group_id" in df.columns:
            referenced = set(df["group_id"].dropna())

            orphan_groups = referenced - group_ids

            if orphan_groups:
                errors.append(
                    f"{name}: {len(orphan_groups)} orphan group_id values"
                )

    # 6. Variant coverage

    variant_ids = set(variants["variant_id"])

    for name, df in {
        "product_nutrition": nutrition,
        "product_scores": scores,
    }.items():
        if "variant_id" in df.columns:
            referenced = set(df["variant_id"].dropna())

            orphan_variants = referenced - variant_ids

            if orphan_variants:
                errors.append(
                    f"{name}: {len(orphan_variants)} orphan variant_id values"
                )

    # 7. Basic row-count expectations

    expected_counts = {
        "product_metadata": 4440,
        "product_groups": 3845,
        "product_variants": 4320,
        "product_nutrition": 4440,
        "product_scores": 4440,
    }

    for name, expected in expected_counts.items():
        actual = len(tables[name])

        if actual != expected:
            errors.append(
                f"{name}: expected {expected} rows, found {actual}"
            )

    # Final result

    print("\n=== Production Validation ===")

    for name, df in tables.items():
        print(
            f"{name}: {len(df):,} rows × {len(df.columns)} columns"
        )

    if errors:
        print("\n❌ VALIDATION FAILED")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    print("\n✅ VALIDATION PASSED")
    print("All production tables are structurally consistent.")


if __name__ == "__main__":
    validate_production()
