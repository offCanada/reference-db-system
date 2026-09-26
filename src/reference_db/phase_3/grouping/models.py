from __future__ import annotations

from dataclasses import dataclass
from typing import Any

IDENTITY_COLUMNS = (
    "is_organic",
    "is_gluten_free",
    "is_naturally_simple",
    "is_sugar_free",
    "is_unsalted",
    "is_lactose_free",
    "is_peanut_free",
    "is_plant_based",
    "is_reduced_sodium",
    "fat_percentage",
    "fat_level",
    "flavour",
    "formulation",
)


@dataclass(frozen=True)
class ProductRecord:
    external_id: str
    product_name: str
    core_title: str
    identity_attributes: dict[str, Any]
    functional_variant: str | None


@dataclass(frozen=True)
class CandidatePair:
    external_id_a: str
    external_id_b: str
    product_name_a: str
    product_name_b: str
    similarity: float


@dataclass(frozen=True)
class GroupDecision:
    external_id_a: str
    external_id_b: str
    decision: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class ResolvedGroup:
    group_id: str
    external_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolutionResult:
    groups: tuple[ResolvedGroup, ...]
    assignments: dict[str, str]
    review_queue: tuple[GroupDecision, ...]
    contradictions: tuple[GroupDecision, ...]