from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


FILES = {
    "p1": DATA_DIR / "phase_1" / "validated_products.parquet",
    "p2": DATA_DIR / "phase_2" / "standardized_products.parquet",
    "p3_groups": DATA_DIR / "phase_3" / "product_groups.parquet",
    "p3_classification": DATA_DIR / "phase_3" / "product_classification.parquet",
    "p3_taxonomy": DATA_DIR / "phase_3" / "product_taxonomy.parquet",
    "p4_variants": DATA_DIR / "phase_4" / "product_variants.parquet",
    "p4_mapping": DATA_DIR / "phase_4" / "product_variant_mapping.parquet",
    "p5_nutrition": DATA_DIR / "phase_5" / "product_nutrition.parquet",
    "p6_scores": DATA_DIR / "phase_6" / "product_scores.parquet",
    "production_metadata": DATA_DIR / "production" / "product_metadata.parquet",
    "production_groups": DATA_DIR / "production" / "product_groups.parquet",
    "production_variants": DATA_DIR / "production" / "product_variants.parquet",
    "production_nutrition": DATA_DIR / "production" / "product_nutrition.parquet",
    "production_scores": DATA_DIR / "production" / "product_scores.parquet",
}


def load_data():
    return {
        name: pd.read_parquet(path)
        for name, path in FILES.items()
    }


def check_unique_ids(df, column, name, errors):
    if column not in df.columns:
        errors.append(f"{name}: missing {column}")
        return

    duplicate_count = df[column].duplicated().sum()

    if duplicate_count:
        errors.append(
            f"{name}: {duplicate_count} duplicate {column} values"
        )


def main():
    print("=== Reference DB End-to-End QA ===")

    errors = []
    data = load_data()

    # ---------------------------------------------------------
    # 1. Required files
    # ---------------------------------------------------------

    missing_files = [
        name
        for name, path in FILES.items()
        if not path.exists()
    ]

    if missing_files:
        errors.append(
            f"Missing files: {', '.join(missing_files)}"
        )

    # ---------------------------------------------------------
    # 2. Print dataset sizes
    # ---------------------------------------------------------

    print("\n--- Dataset Sizes ---")

    for name, df in data.items():
        print(
            f"{name}: {len(df):,} rows × {len(df.columns)} columns"
        )

    # ---------------------------------------------------------
    # 3. External ID coverage
    # ---------------------------------------------------------

    print("\n--- External ID Coverage ---")

    product_tables = {
        "p1": data["p1"],
        "p2": data["p2"],
        "p3_groups": data["p3_groups"],
        "p3_classification": data["p3_classification"],
        "p3_taxonomy": data["p3_taxonomy"],
        "p4_mapping": data["p4_mapping"],
        "p5_nutrition": data["p5_nutrition"],
        "p6_scores": data["p6_scores"],
        "production_metadata": data["production_metadata"],
        "production_nutrition": data["production_nutrition"],
        "production_scores": data["production_scores"],
    }

    reference_ids = set(data["p1"]["external_id"])

    for name, df in product_tables.items():
        if "external_id" not in df.columns:
            errors.append(f"{name}: missing external_id")
            continue

        ids = set(df["external_id"])

        missing = reference_ids - ids
        extra = ids - reference_ids

        print(
            f"{name}: "
            f"{len(ids):,} IDs | "
            f"missing={len(missing)} | "
            f"extra={len(extra)}"
        )

        if missing:
            errors.append(
                f"{name}: {len(missing)} missing external_id values"
            )

        if extra:
            errors.append(
                f"{name}: {len(extra)} unexpected external_id values"
            )

    # ---------------------------------------------------------
    # 4. ID uniqueness
    # ---------------------------------------------------------

    print("\n--- ID Uniqueness ---")

    uniqueness_checks = [
        ("p1", "external_id"),
        ("p2", "external_id"),
        ("p3_groups", "external_id"),
        ("p3_classification", "external_id"),
        ("p3_taxonomy", "external_id"),
        ("p4_mapping", "external_id"),
        ("p5_nutrition", "external_id"),
        ("p6_scores", "external_id"),
        ("production_metadata", "external_id"),
        ("production_nutrition", "external_id"),
        ("production_scores", "external_id"),
        ("production_groups", "group_id"),
        ("production_variants", "variant_id"),
    ]

    for name, column in uniqueness_checks:
        check_unique_ids(
            data[name],
            column,
            name,
            errors,
        )

    # ---------------------------------------------------------
    # 5. Group consistency
    # ---------------------------------------------------------

    print("\n--- Group Consistency ---")

    p3_groups = data["p3_groups"]

    group_ids = set(
        p3_groups["group_id"].dropna()
    )

    null_groups = p3_groups["group_id"].isna().sum()

    print(f"P3 null group_id: {null_groups}")
    print(f"P3 unique groups: {len(group_ids):,}")

    if null_groups:
        errors.append(
            f"P3 contains {null_groups} null group_id values"
        )

    production_groups = data["production_groups"]

    production_group_ids = set(
        production_groups["group_id"].dropna()
    )

    missing_production_groups = (
        group_ids - production_group_ids
    )

    extra_production_groups = (
        production_group_ids - group_ids
    )

    print(
        f"Production groups: "
        f"missing={len(missing_production_groups)} | "
        f"extra={len(extra_production_groups)}"
    )

    if missing_production_groups:
        errors.append(
            f"Production missing {len(missing_production_groups)} groups"
        )

    if extra_production_groups:
        errors.append(
            f"Production contains {len(extra_production_groups)} extra groups"
        )

    # ---------------------------------------------------------
    # 6. Variant consistency
    # ---------------------------------------------------------

    print("\n--- Variant Consistency ---")

    p4_variants = data["p4_variants"]
    p4_mapping = data["p4_mapping"]
    p5 = data["p5_nutrition"]
    p6 = data["p6_scores"]

    variant_ids = set(
        p4_variants["variant_id"].dropna()
    )

    print(f"P4 unique variants: {len(variant_ids):,}")

    for name, df in {
        "P4 mapping": p4_mapping,
        "P5 nutrition": p5,
        "P6 scores": p6,
    }.items():

        if "variant_id" not in df.columns:
            errors.append(f"{name}: missing variant_id")
            continue

        referenced = set(
            df["variant_id"].dropna()
        )

        orphaned = referenced - variant_ids

        print(
            f"{name}: orphan variants={len(orphaned)}"
        )

        if orphaned:
            errors.append(
                f"{name}: {len(orphaned)} orphan variant_id values"
            )

    # ---------------------------------------------------------
    # 7. Production coverage
    # ---------------------------------------------------------

    print("\n--- Production Coverage ---")

    production_checks = {
        "metadata": (
            data["production_metadata"],
            reference_ids,
        ),
        "nutrition": (
            data["production_nutrition"],
            reference_ids,
        ),
        "scores": (
            data["production_scores"],
            reference_ids,
        ),
    }

    for name, (df, expected_ids) in production_checks.items():

        actual_ids = set(df["external_id"])

        missing = expected_ids - actual_ids

        print(
            f"Production {name}: "
            f"missing={len(missing)}"
        )

        if missing:
            errors.append(
                f"Production {name}: "
                f"{len(missing)} missing products"
            )

    # ---------------------------------------------------------
    # 8. Nutrition quality
    # ---------------------------------------------------------

    print("\n--- Nutrition Quality ---")

    nutrition = data["p5_nutrition"]

    if "nutrition_quality_status" in nutrition.columns:
        print(
            nutrition["nutrition_quality_status"]
            .value_counts(dropna=False)
            .to_string()
        )
    else:
        errors.append(
            "P5 nutrition: missing nutrition_quality_status"
        )

    # ---------------------------------------------------------
    # 9. Score eligibility
    # ---------------------------------------------------------

    print("\n--- Score Eligibility ---")

    scores = data["p6_scores"]

    if "score_eligibility" in scores.columns:
        print(
            scores["score_eligibility"]
            .value_counts(dropna=False)
            .to_string()
        )
    else:
        errors.append(
            "P6 scores: missing score_eligibility"
        )

    # ---------------------------------------------------------
    # 10. Final result
    # ---------------------------------------------------------

    print("\n=== QA Result ===")

    if errors:
        print("❌ E2E QA FAILED")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    print("✅ E2E QA PASSED")
    print("All cross-phase and production consistency checks passed.")


if __name__ == "__main__":
    main()
