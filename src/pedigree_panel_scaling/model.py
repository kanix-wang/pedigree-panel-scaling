"""Self-contained inputs for the original independent-gene pedigree model."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PedigreeCase:
    """People precede their children; each relationship is (child, parent, parent)."""

    people: tuple[str, ...]
    relationships: tuple[tuple[str, str, str], ...]
    genes: tuple[str, ...]
    allele_freqs: Mapping[str, float]
    a_gene: Mapping[str, float]
    b_gene: Mapping[str, float]
    omega_gene: Mapping[str, float]
    delta_gene: Mapping[str, float]
    fixed_cost: float
    variable_cost: float
    initial_evidence: tuple[tuple[str, tuple[int, ...]], ...] = ()


def load_case(path: str | Path) -> PedigreeCase:
    """Read an editable JSON case; the inference engine validates its values."""
    data = json.loads(Path(path).read_text())
    data["people"] = tuple(data["people"])
    data["genes"] = tuple(data["genes"])
    data["relationships"] = tuple(tuple(row) for row in data["relationships"])
    data["initial_evidence"] = tuple(
        (person, tuple(panel)) for person, panel in data.get("initial_evidence", ())
    )
    return PedigreeCase(**data)


def synthetic_case(people: int = 15, genes: int = 15,
                   family: str = "nuclear") -> PedigreeCase:
    """Create an illustrative low-width pedigree; parameters are synthetic."""
    if (isinstance(people, bool) or not isinstance(people, int) or people < 3
            or isinstance(genes, bool) or not isinstance(genes, int) or genes < 1):
        raise ValueError("Examples require at least 3 people and 1 gene")
    if family == "nuclear":
        names = ("F", "M", *(f"C{i}" for i in range(1, people - 1)))
        relationships = tuple((p, "F", "M") for p in names[2:])
    elif family == "multigeneration":
        if people < 5:
            raise ValueError("Multigeneration examples require at least 5 people")
        names = ("F", "M", "S", "C", *(f"G{i}" for i in range(1, people - 3)))
        relationships = (("C", "F", "M"), *((p, "C", "S") for p in names[4:]))
    else:
        raise ValueError("Family must be nuclear or multigeneration")
    gene_names = tuple(f"g{i}" for i in range(1, genes + 1))
    return PedigreeCase(
        people=names, relationships=relationships, genes=gene_names,
        allele_freqs={g: 0.01 + 0.001 * (i % 15) for i, g in enumerate(gene_names)},
        a_gene=dict.fromkeys(gene_names, -0.4),
        b_gene=dict.fromkeys(gene_names, -0.1),
        omega_gene=dict.fromkeys(gene_names, 0.01),
        delta_gene=dict.fromkeys(gene_names, 0.5),
        fixed_cost=0.03, variable_cost=0.04,
    )
