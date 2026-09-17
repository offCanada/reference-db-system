from __future__ import annotations

import pandas as pd


def _calculated_mask(df: pd.DataFrame) -> pd.Series:
    """Return a stable boolean mask for calculated Nutri-Score rows."""
    return df["nutri_score_calculated"].astype("boolean").fillna(False)


def _calculated_mask(df: pd.DataFrame) -> pd.Series:
    """Return a stable boolean mask for calculated Nutri-Score rows."""
    return df["nutri_score_calculated"].astype("boolean").fillna(False)


def validate_phase6(
    result: pd.DataFrame,
    expected_rows: int,
) -> dict:
    """Run Phase 6 integrity checks."""

    checks: dict[str, bool] = {}

    checks["row_count_matches_input"] = (
        len(result) == expected_rows
    )

    checks["external_id_unique"] = (
        result["external_id"].is_unique
    )

    checks["group_id_complete"] = (
        result["group_id"].notna().all()
    )

    checks["variant_id_complete"] = (
        result["variant_id"].notna().all()
    )

    non_food_scored = (
        (result["domain"] == "non_food")
        & _calculated_mask(result)
    )

    checks["non_food_not_scored"] = not non_food_scored.any()

    no_source_scored = (
        (result["nutrition_quality_status"] == "NO_SOURCE_DATA")
        & _calculated_mask(result)
    )

    checks["no_source_data_not_scored"] = not no_source_scored.any()

    baby_care_scored = (
        (result["taxonomy"] == "BABY_CARE")
        & _calculated_mask(result)
    )

    checks["excluded_categories_not_scored"] = not baby_care_scored.any()

    scored = result[
        _calculated_mask(result)
    ]

    if len(scored) > 0:
        checks["scored_rows_have_grade"] = (
            scored["nutri_score_grade"].isin(
                ["A", "B", "C", "D", "E"]
            ).all()
        )
    else:
        checks["scored_rows_have_grade"] = True

    estimated_fvl = result[
        result["fvl_provenance"] == "ESTIMATED"
    ]

    if len(estimated_fvl) > 0:
        checks["estimated_fvl_has_method"] = (
            estimated_fvl["fvl_method"].notna().all()
        )
    else:
        checks["estimated_fvl_has_method"] = True

    checks["mapping_status_present"] = (
        result["mapping_status"].notna().all()
    )

    failed = [
        name for name, passed in checks.items()
        if not passed
    ]

    return {
        "result": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "total_checks": len(checks),
        "passed_checks": len(checks) - len(failed),
    }


def assert_phase6_valid(validation: dict) -> None:
    if validation["result"] != "PASS":
        raise AssertionError(
            "Phase 6 validation failed: "
            + ", ".join(validation["failed_checks"])
        )
