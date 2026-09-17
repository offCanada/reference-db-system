from __future__ import annotations

import pandas as pd


def validate_phase4(
    df: pd.DataFrame,
    variant_table: pd.DataFrame,
    variant_mapping: pd.DataFrame,
    variant_summary: pd.DataFrame,
) -> dict:
    checks = {}

    variants_per_product = df.groupby("external_id")["variant_id"].nunique()

    checks["product_has_one_variant"] = {
        "max_variants_per_product": int(variants_per_product.max()),
        "products_with_multiple_variants": int(
            (variants_per_product > 1).sum()
        ),
        "pass": bool((variants_per_product == 1).all()),
    }

    groups_per_variant = variant_table.groupby("variant_id")["group_id"].nunique()

    checks["variant_belongs_to_one_group"] = {
        "max_groups_per_variant": int(groups_per_variant.max()),
        "variants_with_multiple_groups": int(
            (groups_per_variant > 1).sum()
        ),
        "pass": bool((groups_per_variant == 1).all()),
    }

    checks["all_products_have_group"] = {
        "null_group_ids": int(df["group_id"].isna().sum()),
        "pass": bool(df["group_id"].notna().all()),
    }

    checks["all_products_have_variant"] = {
        "null_variant_ids": int(df["variant_id"].isna().sum()),
        "pass": bool(df["variant_id"].notna().all()),
    }

    checks["variant_count_matches"] = {
        "mapping_variants": int(variant_mapping["variant_id"].nunique()),
        "table_variants": len(variant_table),
        "pass": (
            variant_mapping["variant_id"].nunique()
            == len(variant_table)
        ),
    }

    checks["product_count_matches"] = {
        "input_products": len(df),
        "mapped_products": len(variant_mapping),
        "pass": len(df) == len(variant_mapping),
    }

    checks["group_count_matches"] = {
        "input_groups": int(df["group_id"].nunique()),
        "summary_groups": len(variant_summary),
        "pass": (
            df["group_id"].nunique()
            == len(variant_summary)
        ),
    }

    all_pass = all(check["pass"] for check in checks.values())

    checks["overall"] = {
        "result": "PASS" if all_pass else "FAIL"
    }

    return checks
