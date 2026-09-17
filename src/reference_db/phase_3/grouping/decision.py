from __future__ import annotations

import math
from typing import Any

from reference_db.phase_3.grouping.models import (
    CandidatePair,
    GroupDecision,
    IDENTITY_COLUMNS,
    ProductRecord,
)


def normalize_attribute(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return str(value).strip().lower()


def compare_identity_attributes(
    product_a: ProductRecord,
    product_b: ProductRecord,
) -> GroupDecision:
    variant_a = normalize_attribute(
        product_a.functional_variant
    )
    variant_b = normalize_attribute(
        product_b.functional_variant
    )

    if (
        variant_a is not None
        and variant_b is not None
        and variant_a != variant_b
    ):
        return GroupDecision(
            external_id_a=product_a.external_id,
            external_id_b=product_b.external_id,
            decision="DIFFERENT",
            confidence=1.0,
            reason="functional_variant_conflict",
        )

    missing_attributes: list[str] = []
    compared_attributes = 0

    for attribute in IDENTITY_COLUMNS:
        value_a = normalize_attribute(
            product_a.identity_attributes.get(attribute)
        )
        value_b = normalize_attribute(
            product_b.identity_attributes.get(attribute)
        )

        if value_a is None and value_b is None:
            continue

        if value_a is None or value_b is None:
            missing_attributes.append(attribute)
            continue

        compared_attributes += 1

        if value_a != value_b:
            return GroupDecision(
                external_id_a=product_a.external_id,
                external_id_b=product_b.external_id,
                decision="DIFFERENT",
                confidence=1.0,
                reason=f"identity_conflict:{attribute}",
            )

    if compared_attributes == 0 and not missing_attributes:
        return GroupDecision(
            external_id_a=product_a.external_id,
            external_id_b=product_b.external_id,
            decision="AMBIGUOUS",
            confidence=0.0,
            reason="missing_identity_evidence",
        )

    if missing_attributes:
        return GroupDecision(
            external_id_a=product_a.external_id,
            external_id_b=product_b.external_id,
            decision="AMBIGUOUS",
            confidence=0.0,
            reason="incomplete_identity_evidence:"
            + ",".join(missing_attributes),
        )

    return GroupDecision(
        external_id_a=product_a.external_id,
        external_id_b=product_b.external_id,
        decision="SAME",
        confidence=1.0,
        reason="identity_attributes_match",
    )


def make_group_decision(
    candidate: CandidatePair,
    product_a: ProductRecord,
    product_b: ProductRecord,
) -> GroupDecision:
    return compare_identity_attributes(product_a, product_b)