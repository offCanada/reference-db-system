from dataclasses import dataclass

from reference_db.classification.domain_rules import classify_product_domain
from reference_db.taxonomy.taxonomy_rules import classify_taxonomy


@dataclass(frozen=True)
class ClassificationResult:
    domain: str
    domain_confidence: float
    domain_rule: str | None
    domain_status: str

    taxonomy: str | None
    taxonomy_confidence: float | None
    taxonomy_rule: str | None
    taxonomy_status: str | None
    taxonomy_resolution: str | None
    taxonomy_candidates: tuple[str, ...]


def classify_product(product_name: str) -> ClassificationResult:
    """
    Classify a product into food/non-food/unknown and,
    for food products, assign a Reference DB taxonomy category.
    """

    domain_match = classify_product_domain(product_name)

    # Non-food products must not receive a food taxonomy.
    if domain_match.domain == "non_food":
        return ClassificationResult(
            domain=domain_match.domain,
            domain_confidence=domain_match.confidence,
            domain_rule=domain_match.matched_rule,
            domain_status=domain_match.status,
            taxonomy=None,
            taxonomy_confidence=None,
            taxonomy_rule=None,
            taxonomy_status=None,
            taxonomy_resolution=None,
            taxonomy_candidates=(),
        )

    # Ambiguous / unknown domain → preserve ambiguity.
    if domain_match.domain != "food":
        return ClassificationResult(
            domain=domain_match.domain,
            domain_confidence=domain_match.confidence,
            domain_rule=domain_match.matched_rule,
            domain_status=domain_match.status,
            taxonomy=None,
            taxonomy_confidence=None,
            taxonomy_rule=None,
            taxonomy_status="AMBIGUOUS",
            taxonomy_resolution=None,
            taxonomy_candidates=(),
        )

    # Food → assign taxonomy.
    taxonomy_match = classify_taxonomy(product_name)

    return ClassificationResult(
        domain=domain_match.domain,
        domain_confidence=domain_match.confidence,
        domain_rule=domain_match.matched_rule,
        domain_status=domain_match.status,
        taxonomy=(
            taxonomy_match.category.value
            if taxonomy_match.category is not None
            else None
        ),
        taxonomy_confidence=taxonomy_match.confidence,
        taxonomy_rule=taxonomy_match.matched_rule,
        taxonomy_status=taxonomy_match.status,
        taxonomy_resolution=taxonomy_match.resolution,
        taxonomy_candidates=taxonomy_match.candidates,
    )

