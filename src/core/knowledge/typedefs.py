from enum import StrEnum

from pydantic import BaseModel


class LinkType(StrEnum):
    CHILDOF = "ChildOf"
    PARENTOF = "ParentOf"
    CANPRECEDE = "CanPrecede"
    CANFOLLOW = "CanFollow"
    PEEROF = "PeerOf"


class Platform(BaseModel):
    category: str  # "LANGUAGE NAME", "LANGUAGE CLASS", "TECHNOLOGY CLASS"
    name: str
    prevalence: str


class Relationship(BaseModel):
    nature: LinkType  # ChildOf, ParentOf, CanPrecede, PeerOf, …
    cwe_id: int
    view_id: int
    ordinal: str  # Primary, Secondary


class Consequence(BaseModel):
    scopes: list[str]  # e.g. ["Integrity", "Confidentiality"]
    impacts: list[str]  # e.g. ["Execute Unauthorized Code or Commands"]
    note: str


class Mitigation(BaseModel):
    phase: str
    strategy: str
    description: str


class AlternateTerm(BaseModel):
    term: str
    description: str


class ObservedExample(BaseModel):
    cve_id: str
    description: str
