from __future__ import annotations

from difflib import SequenceMatcher

from reference_db.phase_3.grouping.models import CandidatePair, ProductRecord


def normalize_title(title: str) -> str:
    return " ".join(str(title).lower().split())


def calculate_similarity(title_a: str, title_b: str) -> float:
    normalized_a = normalize_title(title_a)
    normalized_b = normalize_title(title_b)
    return SequenceMatcher(None, normalized_a, normalized_b).ratio()


def generate_candidate(
    product_a: ProductRecord,
    product_b: ProductRecord,
    threshold: float = 0.85,
) -> CandidatePair | None:
    similarity = calculate_similarity(product_a.core_title, product_b.core_title)

    if similarity < threshold:
        return None

    return CandidatePair(
        external_id_a=product_a.external_id,
        external_id_b=product_b.external_id,
        product_name_a=product_a.product_name,
        product_name_b=product_b.product_name,
        similarity=similarity,
    )


def build_block_key(title: str) -> str:
    normalized = normalize_title(title)
    tokens = [token for token in normalized.split() if not token.isdigit()]
    return " ".join(tokens[:4])


def generate_candidates(
    products: list[ProductRecord],
    threshold: float = 0.85,
) -> list[CandidatePair]:
    blocks: dict[str, list[ProductRecord]] = {}

    for product in products:
        block_key = build_block_key(product.core_title)

        if not block_key:
            continue

        blocks.setdefault(block_key, []).append(product)

    candidates: list[CandidatePair] = []

    for block_products in blocks.values():
        for index, product_a in enumerate(block_products):
            for product_b in block_products[index + 1:]:
                candidate = generate_candidate(
                    product_a,
                    product_b,
                    threshold=threshold,
                )

                if candidate is not None:
                    candidates.append(candidate)

    return candidates