from __future__ import annotations

from typing import Any

from reference_db.classification.classifier import classify_product
from reference_db.phase_3.grouping import run_grouping


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
    ) = run_grouping(rows)

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

    (
        grouping_results,
        review_results,
        contradiction_results,
    ) = generate_grouping_results(rows)

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

    return {
        "product_classification": classification_results,
        "product_taxonomy": taxonomy_results,
        "product_groups": grouping_results,
        "grouping_review_queue": review_results,
        "grouping_contradictions": contradiction_results,
    }