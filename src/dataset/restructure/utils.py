import json
import os
import re
import subprocess
import time

from log import logger

PATH2JSON: str = "../../../DiverseVul/small_diversevul.json"
FIELDS_IN_JSON = 9


def findall_regex(pattern: str | re.Pattern, target: str) -> str | list[str]:
    regex: str | re.Pattern = (
        re.compile(pattern=pattern) if isinstance(pattern, str) else pattern
    )
    content = re.findall(pattern=regex, string=target)
    return content if content else ""


def match_regex(pattern: str | re.Pattern, target: str) -> bool:
    regex: str | re.Pattern = (
        re.compile(pattern=pattern) if isinstance(pattern, str) else pattern
    )
    content = re.search(pattern=regex, string=target)
    return True if content else False


def remove_tabs(lineContent: str):
    # remove tabs in function body
    newlineRegEx: re.Pattern = re.compile(pattern=r"\\t")
    lineContent = re.sub(pattern=newlineRegEx, repl=" ", string=lineContent)

    return lineContent


def remove_newlines(lineContent: str) -> str:
    # remove "\n" char in function body
    if match_regex(pattern=r"\\n", target=lineContent):
        lineContent = re.sub(pattern=r"\\n", repl=" ", string=lineContent)
    if match_regex(pattern=r"\n", target=lineContent):
        lineContent = re.sub(pattern=r"\n", repl="", string=lineContent)

    # do not remove newline char if we encountered "message" since it indicates we are at the
    # end of the function metadata block. Let's keep division between function metadata
    # if (matchedMessageFlag and match_regex(pattern=r"}\\n$", target=lineContent)):
    #     matchedMessageFlag = False

    # elif (not matchedMessageFlag) or (matchedMessageFlag and not match_regex(pattern=r"}\\n$", target=lineContent)):
    # remove "newline" char in function body
    # newlineRegEx: re.Pattern = re.compile(pattern=r"\n")
    # lineContent = re.sub(pattern=newlineRegEx, repl='', string=lineContent)

    return lineContent  # , matchedMessageFlag


def remove_backslashes(lineContent: str) -> str:
    obj: str | list[str] = findall_regex(pattern=r"\\", target=lineContent)
    return lineContent if not obj else lineContent.strip().replace(obj[0], "")


def remove_multiplespaces(lineContent: str) -> str:
    # remove multiple consecutive spaces in function body
    multipleSpacesRegEx: re.Pattern = re.compile(pattern=r"\s+(?=.)")
    lineContent = re.sub(pattern=multipleSpacesRegEx, repl=" ", string=lineContent)

    return lineContent


def remove_comments(lineContent: str) -> str:

    # extract comment encapsulated in /**/
    # it is important to use the greedy search to stop the matching at first match
    commentAstRegEx: re.Pattern = re.compile(pattern=r"/\*.*?\*/")
    # check for comments starting with // and match until the eol
    commentRegEx = re.compile(pattern=r"//.*")

    loc: list[str] | str = (
        findall_regex(pattern=commentAstRegEx, target=lineContent)
        if match_regex(pattern=commentAstRegEx, target=lineContent)
        else findall_regex(pattern=commentRegEx, target=lineContent)
    )
    if loc:
        for item in loc:
            lineContent = lineContent.replace(item, "")

    return lineContent


def read_json() -> list[str]:
    fileContent: list[str] = []
    with open(PATH2JSON, "r") as json:
        fileContent = json.readlines()
    return fileContent


def create_func_metadatablock(
    content: list[str],
) -> dict[int, dict[str, str | list[str]]]:
    json_dict: dict[int, dict[str, str | list[str]]] = {}
    local_d: dict[str, str | list[str]] = {}
    el: str | list[str]

    # extract "func": ".... }"
    funcRegEx = re.compile(pattern=r"\"func\".*}\"")
    subFuncRegEx = re.compile(pattern=r"\"func\"\s*:\s*")
    # extract "target": \d* "
    targetRegEx = re.compile(pattern=r"\"target\"\s*:\s*\d*")
    subTargetRegEx = re.compile(pattern=r"\"target\"\s*:\s*")
    # extract "cwe": [whatever is inside] "
    cweRegEx = re.compile(pattern=r"\"cwe\"\s*:\s*\[.*?(?=\])")
    subCweRegEx = re.compile(pattern=r"\"cwe\"\s*:\s*\[")

    # extract "project": "prjname"
    projectRegEx = re.compile(pattern=r"\"project\"\s*:\s*\"\w*(?=\")")
    subPrjRegEx = re.compile(pattern=r"\"project\"\s*:\s*\"")
    # extract "commit_id": "alphanumeric"
    commitidRegEx = re.compile(pattern=r"\"commit_id\"\s*:\s*\"\w*(?=\")")
    subCommitRegEx = re.compile(pattern=r"\"commit_id\"\s*:\s*\"")
    # extract "hash": "numeric"
    hashRegEx = re.compile(pattern=r"\"hash\"\s*:\s*\d*")
    subHashRegEx = re.compile(pattern=r"\"hash\"\s*:\s*")
    # extract "size": "size"
    sizeRegEx = re.compile(pattern=r"\"size\"\s*:\s*\d*")
    subSizeRegEx = re.compile(pattern=r"\"size\"\s*:")
    # extract "message": "commit_message"
    messageRegEx = re.compile(pattern=r"\"message\"\s*:\s*\".*(?=\")")
    subMsgRegEx = re.compile(pattern=r"\"message\"\s*:\s*\"")

    regex_dict = {
        "func": funcRegEx,
        "target": targetRegEx,
        "cwe": cweRegEx,
        "project": projectRegEx,
        "commit_id": commitidRegEx,
        "hash": hashRegEx,
        "size": sizeRegEx,
        "message": messageRegEx,
    }
    sub_regex_dict = {
        "func": subFuncRegEx,
        "target": subTargetRegEx,
        "cwe": subCweRegEx,
        "project": subPrjRegEx,
        "commit_id": subCommitRegEx,
        "hash": subHashRegEx,
        "size": subSizeRegEx,
        "message": subMsgRegEx,
    }

    # merge blocks in between "{" and "}\n"
    # iterate in the list and start merging to form blocks
    for lineIdx in range(len(content)):
        local_d = {}
        for key, val in regex_dict.items():
            # apply regex to extract field content
            el = (
                findall_regex(pattern=val, target=content[lineIdx])[0]
                if match_regex(pattern=val, target=content[lineIdx])
                else ""
            )
            # remove unnecessary chars
            el = re.sub(pattern=sub_regex_dict[key], repl="", string=el) if el else ""

            el = el.strip()
            # in case of "func" field, remove leading and trailing dquotes
            el = el[1:-1] if (el and key == "func") else el
            # in case of "cwe" field, check if there are more codes
            if key == "cwe":
                el = [e[1:-1] for e in el.split(sep=",")] if ("," in el) else el[1:-1]

            local_d[key] = el

        json_dict[lineIdx] = local_d

    return json_dict


def remove_unused_fields(
    dic: dict[int, dict[str, str | list[str]]]
) -> dict[int, dict[str, str | list[str]]]:
    lop: list[str] = ["func", "target", "cwe"]  # list of keys to preserve

    # needed because during iteration it is not possible to delete elements
    shrinkedDict: dict[int, dict[str, str | list[str]]] = {}
    localD: dict[str, str | list[str]] = {}
    ##

    for k, v in dic.items():
        localD = {}
        for key in v.keys():
            if key in lop:
                localD[key] = v[key]
        shrinkedDict[k] = localD
    return shrinkedDict


def write_json(dic: dict, output: str) -> None:
    output, _ = os.path.splitext(p=output)
    with open(f"{output}.json", "w") as outfile:
        json.dump(obj=dic, fp=outfile, indent=2, sort_keys=True)


def create_empty_tmp_source():
    with open("tmp.c", "w") as _:
        pass

    logger.debug("Tmp empty file created successfully")


def populate_tmp_file(filepth: str, dic: dict[int, dict[str, str | list[str]]]):
    with open(file=filepth, mode="w+") as f:
        for v in dic.values():
            f.write(str(v["func"]))  # casting to avoid linting issues
            f.write("\n")
            f.write("/*" + "*" * 20 + "*/")


def spawn_refactor(filepath: str) -> None:
    exit_code_obj = subprocess.run(
        args=[
            "indent",
            "-brf",
            "-nbfda",
            "-nbfde",
            "-nut",
            "-linux",
            "-as",
            "-i4",
            "-nbad",
            "-nhnl",
            "-nbap",
            "-l1000",
            f"{filepath}",
        ],
        capture_output=True,
        text=True,
    )
    if exit_code_obj.returncode:
        logger.error(exit_code_obj.stderr)
    else:
        logger.info("Refactor successfully achieved")


def get_refactored_chunks(src_pth: str) -> list[str]:
    loFuncBody: list[str] = []
    with open(file=src_pth, mode="r") as f:
        los: str = f.read()

    # split based on function separator (comment)
    loFuncBody = los.split(sep=("/*" + "*" * 20 + "*/"))
    # filter out all final "newline" char added by formatter and empty strings
    loFuncBody = list(
        filter(
            None,
            [
                (
                    l[:-1]
                    if not match_regex(pattern=re.compile(pattern=r"^\s*"), target=l[0])
                    else re.sub(pattern=r"^\s*", repl="", string=l[:-1])
                )
                for l in loFuncBody
            ],
        )
    )

    return loFuncBody


def build_refactored_json(
    dic: dict[int, dict[str, str | list[str]]], src_pth: str
) -> dict[int, dict[str, str | list[str]]]:
    ref_chunks: list[str] = get_refactored_chunks(src_pth=src_pth)
    for idx, k in enumerate(dic.keys()):
        dic[k].update({"func": ref_chunks[idx]})

    return dic


def rm_tmp_file(filepath: str) -> None:
    exit_code_obj = subprocess.run(
        args=["rm", f"{filepath}", f"{filepath}~"], capture_output=True, text=True
    )
    if exit_code_obj.returncode:
        logger.error(exit_code_obj.stderr)
    else:
        logger.info("Tmp file removed successfully")


def add_desc_to_metadata(
    dic: dict[str, str | list[str]], llm
) -> dict[str, str | list[str]]:
    # add an initial time delay to avoid resource echausted error
    time.sleep(10)
    # input the function body and get description
    desc: str = llm.generate_description(dic["func"])
    dic.update({"fdesc": desc[1:-1]})

    return dic
