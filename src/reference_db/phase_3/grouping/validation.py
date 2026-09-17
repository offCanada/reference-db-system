from __future__ import annotations

from dataclasses import dataclass

from reference_db.phase_3.grouping.models import (
    GroupDecision,
    ResolutionResult,
)


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    errors: tuple[str, ...]
    product_count: int
    assigned_count: int
    group_count: int
    singleton_count: int
    multi_product_group_count: int
    contradiction_count: int


def validate_resolution(
    external_ids: list[str],
    result: ResolutionResult,
    decisions: list[GroupDecision],
) -> ValidationResult:
    errors: list[str] = []

    input_ids = set(external_ids)
    assigned_ids = set(result.assignments)

    if len(external_ids) != len(input_ids):
        errors.append("duplicate_external_ids_in_input")

    missing_ids = input_ids - assigned_ids
    if missing_ids:
        errors.append(
            f"missing_assignments:{len(missing_ids)}"
        )

    unexpected_ids = assigned_ids - input_ids
    if unexpected_ids:
        errors.append(
            f"unexpected_assignments:{len(unexpected_ids)}"
        )

    if result.contradictions:
        errors.append(
            f"contradictions_detected:{len(result.contradictions)}"
        )

    groups_by_id = {
        group.group_id: group
        for group in result.groups
    }

    assigned_group_ids = set(result.assignments.values())

    if set(groups_by_id) != assigned_group_ids:
        errors.append("group_assignment_mismatch")

    grouped_product_ids = [
        external_id
        for group in result.groups
        for external_id in group.external_ids
    ]

    if len(grouped_product_ids) != len(set(grouped_product_ids)):
        errors.append("product_appears_in_multiple_groups")

    if set(grouped_product_ids) != input_ids:
        errors.append("group_membership_does_not_match_input")

    decision_by_pair = {
        frozenset(
            (decision.external_id_a, decision.external_id_b)
        ): decision
        for decision in decisions
    }

    for pair, decision in decision_by_pair.items():
        if decision.decision != "DIFFERENT":
            continue

        product_a, product_b = tuple(pair)

        if (
            product_a in result.assignments
            and product_b in result.assignments
            and result.assignments[product_a]
            == result.assignments[product_b]
        ):
            errors.append(
                "different_pair_in_same_group:"
                f"{product_a},{product_b}"
            )

    singleton_count = sum(
        1
        for group in result.groups
        if len(group.external_ids) == 1
    )

    multi_product_group_count = sum(
        1
        for group in result.groups
        if len(group.external_ids) > 1
    )

    return ValidationResult(
        passed=not errors,
        errors=tuple(errors),
        product_count=len(external_ids),
        assigned_count=len(result.assignments),
        group_count=len(result.groups),
        singleton_count=singleton_count,
        multi_product_group_count=multi_product_group_count,
        contradiction_count=len(result.contradictions),
    )