"""Deterministic pedigree moralization and canonical junction-tree structure."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True)
class Clique:
    clique_id: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class JunctionTreeEdge:
    left: str
    right: str
    separator: tuple[str, ...]


@dataclass(frozen=True)
class JunctionTreeCandidateEdge:
    """One edge in the complete, deterministic Kruskal candidate sequence."""

    left: str
    right: str
    separator: tuple[str, ...]
    separator_size: int
    decision: Literal["SELECTED", "REJECTED_CYCLE"]


@dataclass(frozen=True)
class CanonicalJunctionTree:
    people: tuple[str, ...]
    relationships: tuple[tuple[str, str, str], ...]
    moral_edges: tuple[tuple[str, str], ...]
    fill_edges: tuple[tuple[str, str], ...]
    elimination_order: tuple[str, ...]
    cliques: tuple[Clique, ...]
    edges: tuple[JunctionTreeEdge, ...]
    candidate_edges: tuple[JunctionTreeCandidateEdge, ...]
    home_cliques: Mapping[str, str]
    factor_scopes: Mapping[str, tuple[str, ...]]
    factor_homes: Mapping[str, str]
    treewidth: int
    running_intersection_valid: bool
    family_scope_coverage_valid: bool


def _ordered_pair(
    left: str, right: str, *, order: Mapping[str, int]
) -> tuple[str, str]:
    if order[left] < order[right]:
        return (left, right)
    return (right, left)


def _validate_population(
    people: Sequence[str], relationships: Sequence[tuple[str, str, str]]
) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...], dict[str, int]]:
    ordered_people = tuple(str(person) for person in people)
    if not ordered_people or len(set(ordered_people)) != len(ordered_people):
        raise ValueError("People must be a nonempty unique ordered sequence")
    order = {person: index for index, person in enumerate(ordered_people)}
    normalized_rows: list[tuple[str, str, str]] = []
    for child_value, parent1_value, parent2_value in relationships:
        child = str(child_value)
        parent1 = str(parent1_value)
        parent2 = str(parent2_value)
        if any(person not in order for person in (child, parent1, parent2)):
            raise ValueError(
                "Relationship references a person outside the population: "
                f"{(child, parent1, parent2)!r}"
            )
        parent1, parent2 = sorted(
            (parent1, parent2),
            key=lambda person: (order[person], person),
        )
        normalized_rows.append((child, parent1, parent2))
    normalized_relationships = tuple(
        sorted(
            normalized_rows,
            key=lambda item: (
                order[item[0]],
                order[item[1]],
                order[item[2]],
                item,
            ),
        )
    )
    children: set[str] = set()
    for child, parent1, parent2 in normalized_relationships:
        if child in children:
            raise ValueError(f"Duplicate parent assignment for {child!r}")
        children.add(child)
        if len({child, parent1, parent2}) != 3:
            raise ValueError(f"Invalid pedigree triple: {(child, parent1, parent2)!r}")
        if order[parent1] >= order[child] or order[parent2] >= order[child]:
            raise ValueError("People must be in topological order")
    return ordered_people, normalized_relationships, order


def _is_connected_clique_subset(
    clique_ids: set[str], edges: Sequence[JunctionTreeEdge]
) -> bool:
    if len(clique_ids) <= 1:
        return True
    adjacency: dict[str, set[str]] = {clique_id: set() for clique_id in clique_ids}
    for edge in edges:
        if edge.left in clique_ids and edge.right in clique_ids:
            adjacency[edge.left].add(edge.right)
            adjacency[edge.right].add(edge.left)
    visited: set[str] = set()
    queue = deque([min(clique_ids)])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        queue.extend(sorted(adjacency[current].difference(visited)))
    return visited == clique_ids


def build_canonical_junction_tree(
    *,
    people: Sequence[str],
    relationships: Sequence[tuple[str, str, str]],
) -> CanonicalJunctionTree:
    """Build the canonical junction tree using stable min-fill and Kruskal."""

    ordered_people, normalized_relationships, order = _validate_population(
        people, relationships
    )
    adjacency: dict[str, set[str]] = {person: set() for person in ordered_people}
    moral_edge_set: set[tuple[str, str]] = set()
    for child, parent1, parent2 in normalized_relationships:
        for left, right in (
            (parent1, child),
            (parent2, child),
            (parent1, parent2),
        ):
            edge = _ordered_pair(left, right, order=order)
            moral_edge_set.add(edge)
            adjacency[left].add(right)
            adjacency[right].add(left)

    remaining = set(ordered_people)
    fill_edges: set[tuple[str, str]] = set()
    elimination_order: list[str] = []
    elimination_cliques: list[frozenset[str]] = []
    while remaining:
        candidate_details: list[
            tuple[int, int, str, tuple[str, ...], tuple[tuple[str, str], ...]]
        ] = []
        for person in ordered_people:
            if person not in remaining:
                continue
            neighbors = tuple(
                sorted(adjacency[person].intersection(remaining), key=order.__getitem__)
            )
            missing: list[tuple[str, str]] = []
            for left_index, left in enumerate(neighbors):
                for right in neighbors[left_index + 1 :]:
                    if right not in adjacency[left]:
                        missing.append(_ordered_pair(left, right, order=order))
            candidate_details.append(
                (len(missing), order[person], person, neighbors, tuple(missing))
            )
        _fill_count, _person_index, chosen, neighbors, missing_edges = min(
            candidate_details
        )
        elimination_order.append(chosen)
        elimination_cliques.append(frozenset((chosen, *neighbors)))
        for left, right in missing_edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
            fill_edges.add((left, right))
        remaining.remove(chosen)

    maximal_sets = tuple(
        clique
        for clique in elimination_cliques
        if not any(clique < other for other in elimination_cliques)
    )
    unique_maximal = {clique for clique in maximal_sets}
    ordered_clique_members = tuple(
        sorted(
            (tuple(sorted(clique, key=order.__getitem__)) for clique in unique_maximal),
            key=lambda members: (
                len(members),
                tuple(order[person] for person in members),
                members,
            ),
        )
    )
    cliques = tuple(
        Clique(clique_id=f"C{index:03d}", members=members)
        for index, members in enumerate(ordered_clique_members)
    )
    if not cliques:
        raise ValueError("Canonical junction tree contains no cliques")

    candidate_specs: list[tuple[int, int, int, tuple[str, ...]]] = []
    for left_index, left_clique in enumerate(cliques):
        for right_index in range(left_index + 1, len(cliques)):
            right_clique = cliques[right_index]
            right_members = set(right_clique.members)
            separator = tuple(
                person for person in left_clique.members if person in right_members
            )
            candidate_specs.append(
                (-len(separator), left_index, right_index, separator)
            )
    candidate_specs.sort()

    parent = list(range(len(cliques)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges: list[JunctionTreeEdge] = []
    candidate_edges: list[JunctionTreeCandidateEdge] = []
    for negative_weight, left_index, right_index, separator in candidate_specs:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root == right_root:
            decision: Literal["SELECTED", "REJECTED_CYCLE"] = "REJECTED_CYCLE"
            candidate_edges.append(
                JunctionTreeCandidateEdge(
                    left=cliques[left_index].clique_id,
                    right=cliques[right_index].clique_id,
                    separator=separator,
                    separator_size=-negative_weight,
                    decision=decision,
                )
            )
            continue
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root
        edges.append(
            JunctionTreeEdge(
                left=cliques[left_index].clique_id,
                right=cliques[right_index].clique_id,
                separator=separator,
            )
        )
        candidate_edges.append(
            JunctionTreeCandidateEdge(
                left=cliques[left_index].clique_id,
                right=cliques[right_index].clique_id,
                separator=separator,
                separator_size=-negative_weight,
                decision="SELECTED",
            )
        )
    if len(edges) != len(cliques) - 1:
        raise ValueError("Canonical clique graph could not be connected")

    home_cliques: dict[str, str] = {}
    for person in ordered_people:
        containing = tuple(clique for clique in cliques if person in clique.members)
        if not containing:
            raise ValueError(f"No clique contains person {person!r}")
        home_cliques[person] = min(
            containing, key=lambda clique: (len(clique.members), clique.clique_id)
        ).clique_id

    child_people = {child for child, _parent1, _parent2 in normalized_relationships}
    factor_scope_sets: dict[str, frozenset[str]] = {
        f"founder:{person}": frozenset((person,))
        for person in ordered_people
        if person not in child_people
    }
    factor_scope_sets.update(
        {
            f"family:{child}": frozenset((child, parent1, parent2))
            for child, parent1, parent2 in normalized_relationships
        }
    )
    factor_scopes = {
        factor_id: tuple(person for person in ordered_people if person in scope)
        for factor_id, scope in sorted(factor_scope_sets.items())
    }
    factor_homes: dict[str, str] = {}
    for factor_id, ordered_scope in factor_scopes.items():
        scope = set(ordered_scope)
        containing = tuple(
            clique for clique in cliques if scope.issubset(clique.members)
        )
        if not containing:
            raise ValueError(
                f"No clique contains factor {factor_id!r} scope {sorted(scope)!r}"
            )
        factor_homes[factor_id] = min(
            containing, key=lambda clique: (len(clique.members), clique.clique_id)
        ).clique_id

    running_intersection_valid = all(
        _is_connected_clique_subset(
            {clique.clique_id for clique in cliques if person in clique.members},
            edges,
        )
        for person in ordered_people
    )
    family_scope_coverage_valid = all(
        any({child, parent1, parent2}.issubset(clique.members) for clique in cliques)
        for child, parent1, parent2 in normalized_relationships
    )
    if not running_intersection_valid or not family_scope_coverage_valid:
        raise ValueError(
            "Canonical junction-tree validation failed: "
            f"running_intersection={running_intersection_valid!r}, "
            f"family_scope_coverage={family_scope_coverage_valid!r}"
        )

    return CanonicalJunctionTree(
        people=ordered_people,
        relationships=normalized_relationships,
        moral_edges=tuple(
            sorted(
                moral_edge_set,
                key=lambda edge: (order[edge[0]], order[edge[1]]),
            )
        ),
        fill_edges=tuple(
            sorted(fill_edges, key=lambda edge: (order[edge[0]], order[edge[1]]))
        ),
        elimination_order=tuple(elimination_order),
        cliques=cliques,
        edges=tuple(edges),
        candidate_edges=tuple(candidate_edges),
        home_cliques=MappingProxyType(home_cliques),
        factor_scopes=MappingProxyType(factor_scopes),
        factor_homes=MappingProxyType(factor_homes),
        treewidth=max(len(clique.members) - 1 for clique in cliques),
        running_intersection_valid=running_intersection_valid,
        family_scope_coverage_valid=family_scope_coverage_valid,
    )


__all__ = [
    "CanonicalJunctionTree",
    "Clique",
    "JunctionTreeCandidateEdge",
    "JunctionTreeEdge",
    "build_canonical_junction_tree",
]
