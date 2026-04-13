from __future__ import annotations

import re
from collections import defaultdict

from .typedefs import (
    AlternateTerm,
    Consequence,
    LinkType,
    Mitigation,
    ObservedExample,
    Platform,
    Relationship,
)


def split_entries(raw: str, delimiter: str = "::") -> list[str]:
    """Split a cwe2 delimited string into non-empty chunks."""
    return [chunk for chunk in raw.split(delimiter) if chunk.strip()]


def _parse_keyed_fields(entry: str, keys: list[str]) -> dict[str, list[str]]:
    """Parse a KEY:VALUE:KEY:VALUE entry where values may contain colons.

    Only splits on :KEY: boundaries, so values like "DoS: Crash" are preserved.
    """
    key_pattern = "|".join(re.escape(k) for k in keys)
    pattern = rf"({key_pattern}):(.*?)(?=:(?:{key_pattern}):|$)"
    matches = re.findall(pattern, entry)
    result: dict[str, list[str]] = defaultdict(list)
    for key, value in matches:
        stripped = value.strip()
        if stripped:
            result[key].append(stripped)
    return dict(result)


def parse_platforms(raw: str) -> list[Platform]:
    platforms: list[Platform] = []
    for entry in split_entries(raw):
        fields = _parse_keyed_fields(
            entry,
            [
                "LANGUAGE NAME",
                "LANGUAGE CLASS",
                "LANGUAGE PREVALENCE",
                "TECHNOLOGY CLASS",
                "TECHNOLOGY PREVALENCE",
            ],
        )
        # Determine the category and name
        for category_key in ("LANGUAGE NAME", "LANGUAGE CLASS", "TECHNOLOGY CLASS"):
            if category_key in fields:
                prevalence_key = category_key.split()[0] + " PREVALENCE"
                platforms.append(
                    Platform(
                        category=category_key,
                        name=fields[category_key][0],
                        prevalence=fields.get(prevalence_key, ["Undetermined"])[0],
                    )
                )
    return platforms


def parse_relationships(raw: str) -> list[Relationship]:
    relationships: list[Relationship] = []
    for entry in split_entries(raw):
        fields = _parse_keyed_fields(entry, ["NATURE", "CWE ID", "VIEW ID", "ORDINAL"])
        if "NATURE" not in fields or "CWE ID" not in fields:
            continue

        view_id: list[str] | None = fields.get("VIEW ID")
        if not view_id or view_id[0] != "1000":
            continue

        nature = fields["NATURE"][0]
        if nature not in LinkType.__members__.values():
            continue

        relationships.append(
            Relationship(
                nature=LinkType(fields["NATURE"][0]),
                cwe_id=int(fields["CWE ID"][0]),
                view_id=int(fields.get("VIEW ID", ["0"])[0]),
                ordinal=fields.get("ORDINAL", [""])[0],
            )
        )
    return relationships


def parse_consequences(raw: str) -> list[Consequence]:
    consequences: list[Consequence] = []
    for entry in split_entries(raw):
        fields = _parse_keyed_fields(entry, ["SCOPE", "IMPACT", "NOTE"])
        consequences.append(
            Consequence(
                scopes=fields.get("SCOPE", []),
                impacts=fields.get("IMPACT", []),
                note=fields.get("NOTE", [""])[0] if "NOTE" in fields else "",
            )
        )
    return consequences


def parse_mitigations(raw: str) -> list[Mitigation]:
    mitigations: list[Mitigation] = []
    for entry in split_entries(raw):
        fields = _parse_keyed_fields(
            entry, ["PHASE", "STRATEGY", "DESCRIPTION", "EFFECTIVENESS"]
        )
        if "DESCRIPTION" not in fields:
            continue
        mitigations.append(
            Mitigation(
                phase=fields.get("PHASE", [""])[0],
                strategy=fields.get("STRATEGY", [""])[0],
                description=fields["DESCRIPTION"][0],
            )
        )
    return mitigations


def parse_alternate_terms(raw: str) -> list[AlternateTerm]:
    terms: list[AlternateTerm] = []
    for entry in split_entries(raw):
        fields = _parse_keyed_fields(entry, ["TERM", "DESCRIPTION"])
        if "TERM" not in fields:
            continue
        terms.append(
            AlternateTerm(
                term=fields["TERM"][0],
                description=fields.get("DESCRIPTION", [""])[0],
            )
        )
    return terms


def parse_observed_examples(raw: str) -> list[ObservedExample]:
    examples: list[ObservedExample] = []
    for entry in split_entries(raw):
        fields = _parse_keyed_fields(entry, ["REFERENCE", "DESCRIPTION", "LINK"])
        if "REFERENCE" not in fields:
            continue
        examples.append(
            ObservedExample(
                cve_id=fields["REFERENCE"][0],
                description=fields.get("DESCRIPTION", [""])[0],
            )
        )
    return examples
