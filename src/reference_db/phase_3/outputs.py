from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_COLUMNS = {
    "product_classification": [
        "external_id",
        "product_name",
        "domain",
        "domain_confidence",
        "domain_rule",
        "domain_status",
        "taxonomy",
        "taxonomy_confidence",
        "taxonomy_rule",
        "taxonomy_status",
    ],
    "product_taxonomy": [
        "external_id",
        "taxonomy",
        "taxonomy_confidence",
        "taxonomy_rule",
        "taxonomy_status",
    ],
    "product_groups": [
        "external_id",
        "group_id",
    ],
    "grouping_review_queue": [
        "external_id_a",
        "external_id_b",
        "decision",
        "confidence",
        "reason",
    ],
    "grouping_contradictions": [
        "external_id_a",
        "external_id_b",
        "decision",
        "confidence",
        "reason",
    ],
}


def write_phase3_outputs(
    results: dict[str, list[dict[str, Any]]],
    output_dir: str | Path = "phase_3",
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    for dataset_name, records in results.items():
        columns = OUTPUT_COLUMNS.get(dataset_name)
        df = pd.DataFrame(records, columns=columns)
        path = output_path / f"{dataset_name}.parquet"
        df.to_parquet(path, index=False)
        paths[dataset_name] = path

    return paths