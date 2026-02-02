import json
from pathlib import Path
from typing import Iterable


def write_json(
    record: Iterable, dst: str, encoding: str = "utf-8", ensure_ascii: bool = False
):
    dst_pth = Path(dst)
    dst_pth.parent.mkdir(parents=True, exist_ok=True)

    with open(file=dst_pth, mode="w", encoding=encoding) as f:
        for entry in record:
            f.write(json.dumps(entry, ensure_ascii=ensure_ascii) + "\n")


def filter_contaminated(input_path: str, output_path: str, contaminated_path: str):
    """
    Remove samples that reference 'known outcome'.
    Save contaminated samples separately for reference.
    """
    clean = []
    contaminated = []

    contamination_phrases = [
        "known outcome",
        "as per the label",
        "the label states",
        "the label indicates",
        "according to the label",
    ]

    with open(file=input_path, mode="r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            reasoning = entry.get("reasoning", "").lower()

            is_contaminated = any(
                phrase in reasoning for phrase in contamination_phrases
            )

            if is_contaminated:
                entry["_original_index"] = i
                contaminated.append(entry)
            else:
                clean.append(entry)

    # write data
    write_json(record=clean, dst=output_path)
    write_json(record=contaminated, dst=contaminated_path)

    # stats
    total = len(clean) + len(contaminated)
    print("=" * 60)
    print("CONTAMINATION REMOVAL RESULTS")
    print("=" * 60)
    print(f"Total samples:        {total}")
    print(f"Contaminated:         {len(contaminated)} ({len(contaminated)/total:.1%})")
    print(f"Clean:                {len(clean)} ({len(clean)/total:.1%})")
    print("=" * 60)

    clean_vuln = sum(1 for e in clean if e.get("target", 0) == 1)
    clean_safe = len(clean) - clean_vuln
    print(f"\nClean set balance:")
    print(f"  Vulnerable: {clean_vuln} ({clean_vuln/len(clean):.1%})")
    print(f"  Safe:       {clean_safe} ({clean_safe/len(clean):.1%})")

    return len(clean), len(contaminated)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print(
            "Usage: python main.py <input.jsonl> <output_clean.jsonl> <output_contaminated.jsonl>"
        )
        sys.exit(1)

    input, output, contaminated = sys.argv[1:]
    filter_contaminated(input, output, contaminated)
