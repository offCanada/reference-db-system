from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd

from reference_db.classification.classifier import classify_product
from reference_db.phase_3.grouping import run_grouping
from reference_db.phase_3.outputs import write_phase3_outputs


INPUT_FILE = Path("data/phase_2/standardized_products.parquet")
OUTPUT_DIR = Path("data/phase_3")


def _get_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)

    return getattr(row, key, default)


def classify_products(rows: list[Any]) -> list[dict[str, Any]]:
    results = []

    for row in rows:
        external_id = _get_value(row, "external_id")
        product_name = _get_value(row, "product_name", "")

        result = classify_product(product_name or "")

        results.append(
            {
                "external_id": external_id,
                "product_name": product_name,
                "domain": result.domain,
                "domain_confidence": result.domain_confidence,
                "domain_rule": result.domain_rule,
                "domain_status": result.domain_status,
                "taxonomy": result.taxonomy,
                "taxonomy_confidence": result.taxonomy_confidence,
                "taxonomy_rule": result.taxonomy_rule,
                "taxonomy_status": result.taxonomy_status,
                "taxonomy_resolution": result.taxonomy_resolution,
                "taxonomy_candidates": result.taxonomy_candidates,
            }
        )

    return results


def generate_grouping_results(
    rows: list[Any],
    isolated_ids: Collection[str] = (),
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    (
        products,
        candidates,
        decisions,
        resolution,
        validation,
    ) = run_grouping(rows, isolated_ids=isolated_ids)

    grouping_results = [
        {
            "external_id": external_id,
            "group_id": group_id,
        }
        for external_id, group_id in resolution.assignments.items()
    ]

    review_results = [
        {
            "external_id_a": decision.external_id_a,
            "external_id_b": decision.external_id_b,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "reason": decision.reason,
        }
        for decision in resolution.review_queue
    ]

    contradiction_results = [
        {
            "external_id_a": decision.external_id_a,
            "external_id_b": decision.external_id_b,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "reason": decision.reason,
        }
        for decision in resolution.contradictions
    ]

    if not validation.passed:
        raise ValueError(
            "Phase 3 grouping validation failed: "
            + "; ".join(validation.errors)
        )

    return (
        grouping_results,
        review_results,
        contradiction_results,
    )


def run_phase3(rows: list[Any]) -> dict[str, list[dict[str, Any]]]:
    classification_results = classify_products(rows)

    # Taxonomy-AMBIGUOUS products are never merged: each one keeps its
    # own singleton group instead of grouping on similarity alone.
    ambiguous_ids = {
        str(result["external_id"])
        for result in classification_results
        if result["taxonomy_status"] == "AMBIGUOUS"
    }

    (
        grouping_results,
        review_results,
        contradiction_results,
    ) = generate_grouping_results(rows, isolated_ids=ambiguous_ids)

    taxonomy_results = [
        {
            "external_id": result["external_id"],
            "taxonomy": result["taxonomy"],
            "taxonomy_confidence": result["taxonomy_confidence"],
            "taxonomy_rule": result["taxonomy_rule"],
            "taxonomy_status": result["taxonomy_status"],
            "taxonomy_resolution": result["taxonomy_resolution"],
            "taxonomy_candidates": result["taxonomy_candidates"],
        }
        for result in classification_results
    ]

    taxonomy_review_queue = [
        {
            "external_id": result["external_id"],
            "product_name": result["product_name"],
            "taxonomy": result["taxonomy"],
            "taxonomy_confidence": result["taxonomy_confidence"],
            "taxonomy_rule": result["taxonomy_rule"],
            "taxonomy_status": result["taxonomy_status"],
            "taxonomy_resolution": result["taxonomy_resolution"],
            "taxonomy_candidates": result["taxonomy_candidates"],
        }
        for result in classification_results
        if result["taxonomy_status"] == "AMBIGUOUS"
    ]

    return {
        "product_classification": classification_results,
        "product_taxonomy": taxonomy_results,
        "taxonomy_review_queue": taxonomy_review_queue,
        "product_groups": grouping_results,
        "grouping_review_queue": review_results,
        "grouping_contradictions": contradiction_results,
    }


def main() -> None:
    print("=== PHASE 3 ===")

    print(f"Loading: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Input rows: {len(df)}")

    rows = df.to_dict(orient="records")

    print("Running classification, taxonomy, and grouping...")

    results = run_phase3(rows)

    print("Writing Phase 3 outputs...")

    paths = write_phase3_outputs(
        results,
        output_dir=OUTPUT_DIR,
    )

    print("\n=== PHASE 3 COMPLETE ===")
    print(f"Products: {len(df)}")

    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
