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
from .grouping import (
    build_identity_attributes,
    build_product_record,
    prepare_products,
    run_grouping,
)
from .models import (
    CandidatePair,
    GroupDecision,
    ProductRecord,
    ResolutionResult,
    ResolvedGroup,
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
    "calculate_similarity",
    "compare_identity_attributes",
    "generate_candidate",
    "generate_candidates",
    "generate_group_id",
    "make_group_decision",
    "normalize_attribute",
    "normalize_title",
    "prepare_products",
    "resolve_groups",
    "run_grouping",
]