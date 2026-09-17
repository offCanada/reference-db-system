from __future__ import annotations

import hashlib

from reference_db.phase_3.grouping.models import (
    GroupDecision,
    ResolutionResult,
    ResolvedGroup,
)


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item_a: str, item_b: str) -> None:
        root_a = self.find(item_a)
        root_b = self.find(item_b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def generate_group_id(external_ids: tuple[str, ...]) -> str:
    identity = "|".join(sorted(external_ids))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def resolve_groups(
    external_ids: list[str],
    decisions: list[GroupDecision],
) -> ResolutionResult:
    union_find = UnionFind(external_ids)

    different_pairs: list[GroupDecision] = []
    review_queue: list[GroupDecision] = []

    for decision in decisions:
        if decision.decision == "SAME":
            union_find.union(
                decision.external_id_a,
                decision.external_id_b,
            )
        elif decision.decision == "DIFFERENT":
            different_pairs.append(decision)
        elif decision.decision == "AMBIGUOUS":
            review_queue.append(decision)

    components: dict[str, list[str]] = {}

    for external_id in external_ids:
        root = union_find.find(external_id)
        components.setdefault(root, []).append(external_id)

    resolved_groups: list[ResolvedGroup] = []
    assignments: dict[str, str] = {}
    contradictions: list[GroupDecision] = []

    for members in components.values():
        member_ids = tuple(sorted(members))
        group_id = generate_group_id(member_ids)

        resolved_groups.append(
            ResolvedGroup(
                group_id=group_id,
                external_ids=member_ids,
            )
        )

        for external_id in member_ids:
            assignments[external_id] = group_id

    for decision in different_pairs:
        group_a = assignments[decision.external_id_a]
        group_b = assignments[decision.external_id_b]

        if group_a == group_b:
            contradictions.append(decision)

    resolved_groups.sort(key=lambda group: group.group_id)

    return ResolutionResult(
        groups=tuple(resolved_groups),
        assignments=assignments,
        review_queue=tuple(review_queue),
        contradictions=tuple(contradictions),
    )