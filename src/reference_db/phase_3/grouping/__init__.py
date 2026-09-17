from .models import (
    CandidatePair,
    GroupDecision,
    ProductRecord,
    ResolutionResult,
    ResolvedGroup,
)

from .grouping import (
    build_identity_attributes,
    build_product_record,
    prepare_products,
    run_grouping,
)

from .candidates import (
    calculate_similarity,
    generate_candidate,
    generate_candidates,
    normalize_title,
)

from .decision import (
    compare_identity_attributes,
    make_group_decision,
    normalize_attribute,
)

from .resolution import (
    generate_group_id,
    resolve_groups,
)

__all__ = [
    "CandidatePair",
    "GroupDecision",
    "ProductRecord",
    "ResolutionResult",
    "ResolvedGroup",
    "build_identity_attributes",
    "build_product_record",
    "prepare_products",
    "run_grouping",
    "calculate_similarity",
    "generate_candidate",
    "generate_candidates",
    "normalize_title",
    "compare_identity_attributes",
    "make_group_decision",
    "normalize_attribute",
    "generate_group_id",
    "resolve_groups",
]