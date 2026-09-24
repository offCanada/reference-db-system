from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "production"


PHASE_2 = DATA_DIR / "phase_2" / "standardized_products.parquet"
PHASE_3_GROUPS = DATA_DIR / "phase_3" / "product_groups.parquet"
PHASE_4_VARIANTS = DATA_DIR / "phase_4" / "product_variants.parquet"
PHASE_4_MAPPING = DATA_DIR / "phase_4" / "product_variant_mapping.parquet"
PHASE_5_NUTRITION = DATA_DIR / "phase_5" / "product_nutrition.parquet"
PHASE_6_SCORES = DATA_DIR / "phase_6" / "product_scores.parquet"


METADATA_COLUMNS = [
    "external_id",
    "source_retailer",
    "title",
    "brand",
    "barcode",
    "description",
    "ingredients_text",
    "nutrition_text",
    "image_url",
    "size_amount",
    "size_unit",
    "price",
    "currency",
    "scraped_at",
    "is_organic",
    "is_gluten_free",
    "is_naturally_simple",
    "is_sugar_free",
    "is_unsalted",
    "is_lactose_free",
    "is_peanut_free",
    "is_plant_based",
    "is_reduced_sodium",
    "fat_level",
    "fat_percentage",
    "brand_norm",
    "product_line",
    "product_name",
    "core_title",
    "size",
    "variant_attributes",
    "flavour",
    "formulation",
    "identity_hash",
]


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "metadata": pd.read_parquet(PHASE_2),
        "groups": pd.read_parquet(PHASE_3_GROUPS),
        "variants": pd.read_parquet(PHASE_4_VARIANTS),
        "variant_mapping": pd.read_parquet(PHASE_4_MAPPING),
        "nutrition": pd.read_parquet(PHASE_5_NUTRITION),
        "scores": pd.read_parquet(PHASE_6_SCORES),
    }


def build_product_metadata(df: pd.DataFrame) -> pd.DataFrame:
    result = df[METADATA_COLUMNS].copy()

    result = result.drop_duplicates(
        subset=["external_id"],
        keep="first",
    )

    return result


def build_product_groups(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby("group_id", as_index=False)
        .agg(product_count=("external_id", "nunique"))
    )

    return result


def build_product_variants(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variant_id",
        "group_id",
        "variant_key",
        "core_title",
        "size",
        "product_count",
    ]

    result = df[columns].copy()

    result = result.drop_duplicates(
        subset=["variant_id"],
        keep="first",
    )

    return result


def build_product_nutrition(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result = result.drop_duplicates(
        subset=["external_id"],
        keep="first",
    )

    return result


def build_product_scores(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result = result.drop_duplicates(
        subset=["external_id"],
        keep="first",
    )

    return result


def build_production() -> dict[str, pd.DataFrame]:
    inputs = load_inputs()

    outputs = {
        "product_metadata": build_product_metadata(
            inputs["metadata"]
        ),
        "product_groups": build_product_groups(
            inputs["groups"]
        ),
        "product_variants": build_product_variants(
            inputs["variants"]
        ),
        "product_nutrition": build_product_nutrition(
            inputs["nutrition"]
        ),
        "product_scores": build_product_scores(
            inputs["scores"]
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, df in outputs.items():
        path = OUTPUT_DIR / f"{name}.parquet"

        df.to_parquet(
            path,
            index=False,
        )

        print(
            f"Saved {path}: "
            f"{len(df):,} rows × {len(df.columns)} columns"
        )

    return outputs


if __name__ == "__main__":
    build_production()
