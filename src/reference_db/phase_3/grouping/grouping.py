from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any

from reference_db.phase_3.grouping.candidates import generate_candidates
from reference_db.phase_3.grouping.decision import make_group_decision
from reference_db.phase_3.grouping.models import (
    IDENTITY_COLUMNS,
    ProductRecord,
    ResolutionResult,
)
from reference_db.phase_3.grouping.resolution import resolve_groups
from reference_db.phase_3.grouping.validation import (
    ValidationResult,
    validate_resolution,
)


def _get_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)

    return getattr(row, key, default)


def build_identity_attributes(row: Any) -> dict[str, Any]:
    return {
        column: _get_value(row, column)
        for column in IDENTITY_COLUMNS
    }


def extract_functional_variant(row: Any) -> str | None:
    value = _get_value(row, "variant_attributes")

    if value is None:
        return None

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if not isinstance(value, dict):
        return None

    functional_variant = value.get("functional_variant")

    if functional_variant is None:
        return None

    return str(functional_variant).strip().lower() or None


def build_product_record(row: Any) -> ProductRecord:
    external_id = _get_value(row, "external_id")
    product_name = _get_value(row, "product_name")
    core_title = _get_value(row, "core_title")

    if external_id is None:
        raise ValueError("Product is missing external_id")

    if product_name is None:
        product_name = ""

    if core_title is None:
        core_title = product_name

    return ProductRecord(
        external_id=str(external_id),
        product_name=str(product_name).strip(),
        core_title=str(core_title).strip(),
        identity_attributes=build_identity_attributes(row),
        functional_variant=extract_functional_variant(row),
    )


def prepare_products(rows: list[Any]) -> list[ProductRecord]:
    return [build_product_record(row) for row in rows]


def run_grouping(
    rows: list[Any],
    threshold: float = 0.85,
    isolated_ids: Collection[str] = (),
) -> tuple[
    list[ProductRecord],
    list[Any],
    list[Any],
    ResolutionResult,
    ValidationResult,
]:
    products = prepare_products(rows)

    candidates = generate_candidates(
        products,
        threshold=threshold,
    )

    product_by_id = {
        product.external_id: product
        for product in products
    }

    decisions = []

    for candidate in candidates:
        product_a = product_by_id[candidate.external_id_a]
        product_b = product_by_id[candidate.external_id_b]

        decision = make_group_decision(
            candidate,
            product_a,
            product_b,
        )

        decisions.append(decision)

    external_ids = [
        product.external_id
        for product in products
    ]

    resolution = resolve_groups(
        external_ids,
        decisions,
        isolated_ids=isolated_ids,
    )

    validation = validate_resolution(
        external_ids,
        resolution,
        decisions,
    )

    return (
        products,
        candidates,
        decisions,
        resolution,
        validation,
    )


def group_product(
    core_title: str,
    identity_attributes: dict[str, Any],
) -> str:
    product = ProductRecord(
        external_id="temporary",
        product_name=core_title,
        core_title=core_title,
        identity_attributes=identity_attributes,
        functional_variant=None,
    )

    resolution = resolve_groups(
        [product.external_id],
        [],
    )

    return resolution.groups[0].group_id