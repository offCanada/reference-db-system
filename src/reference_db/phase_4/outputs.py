from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_phase4_outputs(
    variant_table: pd.DataFrame,
    variant_mapping: pd.DataFrame,
    variant_summary: pd.DataFrame,
    validation: dict,
    output_dir: str | Path = "phase_4",
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    outputs = {
        "product_variants": variant_table,
        "product_variant_mapping": variant_mapping,
        "product_variant_summary": variant_summary,
    }

    for name, df in outputs.items():
        path = output_path / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths[name] = path

    validation_path = output_path / "phase4_validation.json"

    with validation_path.open("w", encoding="utf-8") as file:
        json.dump(validation, file, indent=2)

    paths["validation"] = validation_path

    return paths
