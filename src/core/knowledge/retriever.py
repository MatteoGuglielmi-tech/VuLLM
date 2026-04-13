from .cwe_knowledge_base import CWEEntry


def should_index(entry: CWEEntry) -> bool:
    platforms: set[str] = {p.name for p in entry.platforms}
    if platforms & {"C", "C++", "Not Language-Specific"} or entry.abstraction in {
        "Pillar",
        "Class",
    }:
        return True

    return False
