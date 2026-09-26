from __future__ import annotations

from pathlib import Path

import pandas as pd

PRODUCT_SCORE_COLUMNS = [
    "external_id",
    "group_id",
    "variant_id",
    "domain",
    "domain_confidence",
    "domain_status",
    "taxonomy",
    "taxonomy_confidence",
    "taxonomy_rule",
    "taxonomy_status",
    "nutrition_quality_status",
    "score_eligibility",
    "score_exclusion_reason",
    "calories_per_100g",
    "energy_kj_per_100g",
    "sugars_g_per_100g",
    "saturated_fat_g_per_100g",
    "sodium_mg_per_100g",
    "salt_g_per_100g",
    "fibre_g_per_100g",
    "protein_g_per_100g",
    "fvl_percent",
    "fvl_method",
    "fvl_confidence",
    "fvl_provenance",
    "energy_points",
    "sugar_points",
    "saturated_fat_points",
    "salt_points",
    "negative_points",
    "protein_points",
    "fibre_points",
    "fvl_points",
    "positive_points",
    "nutri_score_raw",
    "nutri_score_grade",
    "nutri_score_calculated",
    "nutri_score_algorithm",
    "nutri_score_status",
]


AGRIBALYSE_COLUMNS = [
    "external_id",
    "group_id",
    "domain",
    "taxonomy",
    "agribalyse_category",
    "mapping_method",
    "mapping_confidence",
    "mapping_status",
    "mapping_source",
    "mapping_version",
    "mapping_note",
]


EXCLUSION_COLUMNS = [
    "external_id",
    "group_id",
    "domain",
    "taxonomy",
    "nutrition_quality_status",
    "score_exclusion_reason",
]


def _select_existing(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Select columns while failing loudly if a required output column is absent."""

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required output columns: {missing}"
        )

    return df[columns].copy()


def build_product_scores(df: pd.DataFrame) -> pd.DataFrame:
    return _select_existing(df, PRODUCT_SCORE_COLUMNS)


def build_agribalyse_mapping(df: pd.DataFrame) -> pd.DataFrame:
    return _select_existing(df, AGRIBALYSE_COLUMNS)


def build_score_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    exclusions = df[
        ~df["score_eligibility"].fillna(False)
    ].copy()

    return _select_existing(
        exclusions,
        EXCLUSION_COLUMNS,
    )


def build_summary(
    df: pd.DataFrame,
    algorithm_version: str,
    mapping_version: str,
) -> pd.DataFrame:

    total = len(df)

    scored = df[
        df["nutri_score_calculated"].astype("boolean").fillna(False)
    ]

    grades = scored["nutri_score_grade"]

    summary = {
        "total_products": total,
        "eligible_products": int(
            df["score_eligibility"].fillna(False).sum()
        ),
        "scored_products": len(scored),
        "not_eligible_products": int(
            (~df["score_eligibility"].fillna(False)).sum()
        ),
        "scored_A": int((grades == "A").sum()),
        "scored_B": int((grades == "B").sum()),
        "scored_C": int((grades == "C").sum()),
        "scored_D": int((grades == "D").sum()),
        "scored_E": int((grades == "E").sum()),
        "fvl_available": int(
            df["fvl_percent"].notna().sum()
        ),
        "fvl_not_available": int(
            df["fvl_percent"].isna().sum()
        ),
        "fvl_estimated": int(
            (df["fvl_provenance"] == "ESTIMATED").sum()
        ),
        "agribalyse_mapped": int(
            (df["mapping_status"] == "MAPPED").sum()
        ),
        "agribalyse_not_available": int(
            (df["mapping_status"] == "NOT_AVAILABLE").sum()
        ),
        "algorithm_version": algorithm_version,
        "mapping_version": mapping_version,
    }

    return pd.DataFrame([summary])


def write_phase6_outputs(
    df: pd.DataFrame,
    output_dir: str | Path = "data/phase_6",
    algorithm_version: str = "Nutri-Score-2023",
    mapping_version: str = "phase6-taxonomy-v1",
) -> dict[str, Path]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    product_scores = build_product_scores(df)
    agribalyse_mapping = build_agribalyse_mapping(df)
    exclusions = build_score_exclusions(df)
    summary = build_summary(
        df,
        algorithm_version,
        mapping_version,
    )

    paths = {
        "product_scores": output_dir / "product_scores.parquet",
        "agribalyse_mapping": output_dir / "agribalyse_mapping.parquet",
        "score_exclusions": output_dir / "score_exclusions.parquet",
        "phase6_summary": output_dir / "phase6_summary.parquet",
    }

    product_scores.to_parquet(
        paths["product_scores"],
        index=False,
    )

    agribalyse_mapping.to_parquet(
        paths["agribalyse_mapping"],
        index=False,
    )

    exclusions.to_parquet(
        paths["score_exclusions"],
        index=False,
    )

    summary.to_parquet(
        paths["phase6_summary"],
        index=False,
    )

    return paths
