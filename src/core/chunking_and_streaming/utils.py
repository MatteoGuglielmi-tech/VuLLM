import difflib
import json
import os
from typing import Any


def assert_file_extensions(
    filepaths: str | list[str],
    target_extensions: str | list[str],
    assertion_msg: str = "",
) -> None:

    # if string, wrap it in list
    filepaths = [filepaths] if isinstance(filepaths, str) else filepaths
    target_extensions = (
        [target_extensions] if isinstance(target_extensions, str) else target_extensions
    )

    # check for multi-files
    for fp, target_ext in zip(filepaths, target_extensions):
        _, ext = os.path.splitext(p=fp)

        assert ext.strip() == target_ext.strip(), (
            f"Wrong extension! Desired file extension for {fp} is {target_ext}"
            if not assertion_msg
            else assertion_msg
        )


def JSON_serialize(obj: list[dict[str, str]], pth: str) -> None:
    with open(file=pth, mode="w") as f:
        for entry in obj:
            f.write(json.dumps(entry) + "\n")


def JSON_read_in(fp: str) -> dict[str, dict[str, str]]:
    assert_file_extensions(filepaths=fp, target_extensions=".json")

    with open(file=fp, mode="r") as f:
        raw: dict[str, dict[str, str]] = json.load(fp=f)

    return raw


def diff_two_strings(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2, autojunk=False)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag != "equal":
            print(
                "{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}".format(
                    tag, i1, i2, j1, j2, s1[i1:i2], s2[j1:j2]
                )
            )


def UNUSED(var: Any) -> None:
    _ = var
    del _
