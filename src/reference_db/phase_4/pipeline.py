from __future__ import annotations

import pandas as pd

from reference_db.phase_4.outputs import write_phase4_outputs
from reference_db.phase_4.validation import validate_phase4
from reference_db.phase_4.variants import (
    assign_variants,
    build_variant_mapping,
    build_variant_summary,
    build_variant_table,
)


def run_phase4(
    phase2_path: str = "data/phase_2/standardized_products.parquet",
    phase3_path: str = "data/phase_3/product_groups.parquet",
    output_dir: str = "data/phase_4",
) -> dict:
    phase2 = pd.read_parquet(phase2_path)
    phase3 = pd.read_parquet(phase3_path)

    required_phase2 = {
        "external_id",
        "variant_attributes",
        "core_title",
        "size",
    }

    required_phase3 = {
        "external_id",
        "group_id",
    }

    missing_phase2 = required_phase2 - set(phase2.columns)
    missing_phase3 = required_phase3 - set(phase3.columns)

    if missing_phase2:
        raise ValueError(
            f"Phase 2 missing required columns: {sorted(missing_phase2)}"
        )

    if missing_phase3:
        raise ValueError(
            f"Phase 3 missing required columns: {sorted(missing_phase3)}"
        )

    if phase2["external_id"].duplicated().any():
        raise ValueError("Phase 2 contains duplicate external_id values")

    if phase3["external_id"].duplicated().any():
        raise ValueError("Phase 3 contains duplicate external_id values")

    df = phase3.merge(
        phase2[
            [
                "external_id",
                "variant_attributes",
                "core_title",
                "size",
            ]
        ],
        on="external_id",
        how="left",
        validate="one_to_one",
    )

    if len(df) != len(phase3):
        raise ValueError("Phase 4 merge changed the Phase 3 product count")

    if df["group_id"].isna().any():
        raise ValueError("Phase 4 found products without group_id")

    df = assign_variants(df)

    variant_table = build_variant_table(df)
    variant_mapping = build_variant_mapping(df)
    variant_summary = build_variant_summary(df)

    validation = validate_phase4(
        df,
        variant_table,
        variant_mapping,
        variant_summary,
    )

    write_phase4_outputs(
        variant_table,
        variant_mapping,
        variant_summary,
        validation,
        output_dir,
    )

    return {
        "product_variants": variant_table,
        "product_variant_mapping": variant_mapping,
        "product_variant_summary": variant_summary,
        "validation": validation,
    }


if __name__ == "__main__":
    results = run_phase4()

    print("\n=== PHASE 4 COMPLETE ===")
    print(f"Products: {len(results['product_variant_mapping'])}")
    print(f"Groups: {len(results['product_variant_summary'])}")
    print(f"Variants: {len(results['product_variants'])}")
    print(f"Validation: {results['validation']['overall']['result']}")
