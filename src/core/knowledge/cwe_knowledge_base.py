"""CWE Knowledge Base built from the full MITRE corpus via cwe2."""

from __future__ import annotations

import csv
import logging

from cwe2.database import Database
from pydantic import BaseModel

from .graph import Graph
from .parser import (
    parse_alternate_terms,
    parse_consequences,
    parse_mitigations,
    parse_observed_examples,
    parse_platforms,
    parse_relationships,
)
from .typedefs import (
    AlternateTerm,
    Consequence,
    LinkType,
    Mitigation,
    ObservedExample,
    Platform,
    Relationship,
)

logger = logging.getLogger(name=__name__)


class CWEEntry(BaseModel):
    cwe_id: int
    name: str
    abstraction: str  # Pillar, Class, Base, Variant
    status: str
    description: str
    extended_description: str
    relationships: list[Relationship]
    platforms: list[Platform]
    consequences: list[Consequence]
    mitigations: list[Mitigation]
    alternate_terms: list[AlternateTerm]
    observed_examples: list[ObservedExample]


class _Col:
    CWE_ID = "CWE-ID"
    NAME = "Name"
    ABSTRACTION = "Weakness Abstraction"
    STATUS = "Status"
    DESCRIPTION = "Description"
    EXTENDED_DESCRIPTION = "Extended Description"
    RELATED_WEAKNESSES = "Related Weaknesses"
    WEAKNESS_ORDINALITIES = "Weakness Ordinalities"
    PLATFORMS = "Applicable Platforms"
    BACKGROUND_DETAILS = "Background Details"
    ALTERNATE_TERMS = "Alternate Terms"
    MODE_OF_INTRODUCTION = "Modes Of Introduction"
    EXPLOITABLE_FACTORS = "Exploitation Factors"
    LIKELIHOOD_OF_EXPLOIT = "Likelihood of Exploit"
    COMMON_CONSEQUENCES = "Common Consequences"
    DETECTION_METHODS = "Detection Methods"
    POTENTIAL_MITIGATIONS = "Potential Mitigations"
    EXAMPLES = "Observed Examples"
    FUNCTIONAL_AREAS = "Functional Areas"
    AFFECTED_RESOURCES = "Affected Resources"
    TAXONOMY_MAPPINGS = "Taxonomy Mappings"
    ATTACK_PATTERNS = "Related Attack Patterns"
    NOTE = "Notes"


def _build_entry(weakness: dict) -> CWEEntry:
    """Convert a cwe2 Weakness object into a CWEEntry."""
    return CWEEntry(
        cwe_id=int(weakness[_Col.CWE_ID]),
        name=weakness[_Col.NAME],
        abstraction=weakness[_Col.ABSTRACTION],
        status=weakness[_Col.STATUS],
        description=weakness[_Col.DESCRIPTION],
        extended_description=weakness[_Col.EXTENDED_DESCRIPTION] or "",
        relationships=parse_relationships(weakness[_Col.RELATED_WEAKNESSES] or ""),
        platforms=parse_platforms(weakness[_Col.PLATFORMS] or ""),
        consequences=parse_consequences(weakness[_Col.COMMON_CONSEQUENCES] or ""),
        mitigations=parse_mitigations(weakness[_Col.POTENTIAL_MITIGATIONS] or ""),
        alternate_terms=parse_alternate_terms(weakness[_Col.ALTERNATE_TERMS] or ""),
        observed_examples=parse_observed_examples(weakness[_Col.EXAMPLES] or ""),
    )


class CWEKnowledgeBase:
    """Queryable in-memory CWE knowledge base.

    Loads the full MITRE corpus at init via cwe2. All queries are dict lookups
    after construction.
    """

    def __init__(self) -> None:
        db = Database()
        self._entries: dict[int, CWEEntry] = {}
        self._build_cwe_db(db)
        self._build_taxonomy_dag()

    def _build_cwe_db(self, db: Database):
        for f in db.cwe_files:
            f.seek(0)
            reader = csv.DictReader(f)
            for csv_row in reader:
                if csv_row[_Col.STATUS] == "Deprecated":
                    continue

                entry = _build_entry(csv_row)
                self._entries[entry.cwe_id] = entry

    def _build_taxonomy_dag(self):
        self._dag = Graph()

        # first I add vertices to guarantee all nodes are present
        for cwe_id in self._entries:
            self._dag.add_vertex(cwe_id=cwe_id)

        # then I add edges (skip relation types the graph doesn't handle)
        for cwe_id, cwe_entry in self._entries.items():
            for rel in cwe_entry.relationships:
                if rel.nature in Graph._inverse:
                    self._dag.add_edge(
                        source_node_id=cwe_id,
                        relation_type=rel.nature,
                        target_node_id=rel.cwe_id,
                    )

    def __len__(self) -> int:
        return len(self._entries)

    def get_by_id(self, cwe_id: int) -> CWEEntry | None:
        return self._entries.get(cwe_id)

    def get_all(self) -> list[CWEEntry]:
        return list(self._entries.values())

    def get_c_cpp_relevant(self) -> list[CWEEntry]:
        """Return entries whose applicable platforms include C or C++."""
        return [
            entry
            for entry in self._entries.values()
            if any(p.name in ("C", "C++") for p in entry.platforms)
        ]

    def search_by_keyword(self, keyword: str) -> list[CWEEntry]:
        """Case-insensitive keyword search across name and description."""
        kw = keyword.lower()
        return [
            entry
            for entry in self._entries.values()
            if kw in entry.name.lower() or kw in entry.description.lower()
        ]

    def get_children(self, cwe_id: int) -> set[int]:
        return self._dag.neighbours(node_id=cwe_id, relation_type=LinkType.PARENTOF)

    def get_parents(self, cwe_id: int) -> set[int]:
        return self._dag.neighbours(node_id=cwe_id, relation_type=LinkType.CHILDOF)

    def get_ancestors(self, cwe_id: int) -> set[int]:
        return self._dag.search(
            backend="bfs", start_id=cwe_id, relation_type=LinkType.CHILDOF
        )

    def get_descendants(self, cwe_id: int) -> set[int]:
        return self._dag.search(
            backend="bfs", start_id=cwe_id, relation_type=LinkType.PARENTOF
        )
