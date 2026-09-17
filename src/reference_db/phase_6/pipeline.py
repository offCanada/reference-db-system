from __future__ import annotations

from pathlib import Path

import pandas as pd

from reference_db.phase_6.agribalyse import (
    MAPPING_VERSION,
    add_agribalyse_mapping,
)
from reference_db.phase_6.eligibility import (
    add_eligibility_columns,
)
from reference_db.phase_6.fvl import add_fvl_fields
from reference_db.phase_6.nutri_score import (
    ALGORITHM_VERSION,
    add_nutri_score_components,
)
from reference_db.phase_6.outputs import (
    write_phase6_outputs,
)
from reference_db.phase_6.validation import (
    assert_phase6_valid,
    validate_phase6,
)


PHASE_3_PATH = Path(
    "data/phase_3/product_classification.parquet"
)

PHASE_5_PATH = Path(
    "data/phase_5/product_nutrition.parquet"
)

OUTPUT_DIR = Path("data/phase_6")


def load_inputs() -> pd.DataFrame:
    classification = pd.read_parquet(
        PHASE_3_PATH
    )

    nutrition = pd.read_parquet(
        PHASE_5_PATH
    )

    result = nutrition.merge(
        classification,
        on="external_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_classification"),
    )

    return result


def run_phase6() -> tuple[pd.DataFrame, dict]:
    print("Phase 6: loading inputs...")

    df = load_inputs()

    expected_rows = len(df)

    print(f"Input rows: {expected_rows}")

    print("Applying eligibility...")
    df = add_eligibility_columns(df)

    print("Adding FVL provenance...")
    df = add_fvl_fields(df)

    print("Calculating Nutri-Score components...")
    df = add_nutri_score_components(df)

    print("Applying Agribalyse mapping...")
    df = add_agribalyse_mapping(df)

    print("Running validation...")

    validation = validate_phase6(
        df,
        expected_rows=expected_rows,
    )

    assert_phase6_valid(validation)

    print("Validation: PASS")

    print("Writing outputs...")

    paths = write_phase6_outputs(
        df,
        output_dir=OUTPUT_DIR,
        algorithm_version=ALGORITHM_VERSION,
        mapping_version=MAPPING_VERSION,
    )

    print("\nPhase 6 complete.")
    print("\nOutputs:")

    for name, path in paths.items():
        print(f"  {name}: {path}")

    return df, validation


def main() -> None:
    run_phase6()


if __name__ == "__main__":
    main()
