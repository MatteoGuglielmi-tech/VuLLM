from .cwe_knowledge_base import CWEEntry

_MITIGATION_CHAR_CAP = 200


def _truncate_to_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    if window[-1] in ".!?":
        return window
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut != -1:
        return window[: cut + 1]
    for i in range(max_chars, len(text)):
        if text[i] in ".!?" and (i + 1 == len(text) or text[i + 1] == " "):
            return text[: i + 1]
    return text


def build_embedding_text(entry: CWEEntry) -> str:
    parts: list[str] = [
        f"CWE-{entry.cwe_id}: {entry.name}",
        "",
        f"Description: {entry.description}",
    ]
    if entry.extended_description:
        parts += ["", entry.extended_description]

    if entry.consequences:
        parts += ["", "Consequences:"]
        for c in entry.consequences:
            scopes = ", ".join(c.scopes) or "Unspecified"
            impacts = ", ".join(c.impacts) or "Unspecified"
            parts.append(f"- {scopes}: {impacts}")

    if entry.mitigations:
        parts += ["", "Mitigations:"]
        for m in entry.mitigations:
            label = " / ".join(x for x in (m.phase, m.strategy) if x)
            desc = _truncate_to_sentence(m.description, _MITIGATION_CHAR_CAP)
            parts.append(f"- {label}: {desc}" if label else f"- {desc}")

    return "\n".join(parts)
