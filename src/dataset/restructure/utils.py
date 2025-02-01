import json
import os
import re
import subprocess
from ast import pattern

from alive_progress import alive_bar

import animate
from log import std_logger

# import time


# PATH2JSON: str = "../../../DiverseVul/small_diversevul.json"
PATH2JSON: str = "../../../DiverseVul/diversevul_20230702.json"
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


def split_lineContent(lineContent: str) -> dict[str, str]:
    d: dict[str, str] = {}
    regex_dict = {
        # extract "func": "...{ ... }"
        # .*? is necessary since some functions have comments at the end
        "func": re.compile(pattern=r"\"func\".*}.*?\""),
        # extract "target": \d* "
        "target": re.compile(pattern=r"\"target\"\s*:\s*\d*"),
        # extract "cwe": [whatever is inside] "
        "cwe": re.compile(pattern=r"\"cwe\"\s*:\s*\[.*\]"),
        # extract "project": "prjname"
        "project": re.compile(pattern=r"\"project\"\s*:\s*\"\w*\""),
        # extract "commit_id": "alphanumeric"
        "commit_id": re.compile(pattern=r"\"commit_id\"\s*:\s*\"\w*\""),
        # extract "hash": "numeric"
        "hash": re.compile(pattern=r"\"hash\"\s*:\s*\d*"),
        # extract "size": "size"
        "size": re.compile(pattern=r"\"size\"\s*:\s*\d*"),
        # extract "message": "commit_message"
        "message": re.compile(pattern=r"\"message\"\s*:\s*\".*\""),
    }

    for key, reg in regex_dict.items():
        # extract block of metadata ("<field>" : "<body>")
        content = findall_regex(pattern=reg, target=lineContent)
        content = content if isinstance(content, str) else content[0]

        d[key] = content

    return d


def remove_tabs(lineContent: str):
    # remove tabs in function body
    newlineRegEx: re.Pattern = re.compile(pattern=r"\\t")
    lineContent = re.sub(pattern=newlineRegEx, repl=" ", string=lineContent)

    return lineContent


def remove_multiple_newlines(lineContent: str) -> str:
    # remove "\\n" within string
    if match_regex(pattern=r"\\\\n", target=lineContent):
        lineContent = re.sub(pattern=r"\\\\n", repl=" ", string=lineContent)
    # remove "\n" within string
    if match_regex(pattern=r"\\n", target=lineContent):
        lineContent = re.sub(pattern=r"\\n", repl=" ", string=lineContent)

    return lineContent


def remove_escaping_quotes(lineContent: str) -> str:
    obj: str | list[str] = findall_regex(pattern=r"\\\"", target=lineContent)
    return lineContent if not obj else lineContent.strip().replace(obj[0], '"')


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
    commentRegEx = re.compile(pattern=r"//.*?(?=\\n)")

    # list of block comments
    lobc: list[str] | str = findall_regex(pattern=commentAstRegEx, target=lineContent)
    loc: list[str] | str = findall_regex(pattern=commentRegEx, target=lineContent)

    # findall_regex(pattern=commentRegEx, target=lineContent)
    if lobc:
        for item in lobc:
            lineContent = lineContent.replace(item, "")
    if loc:
        for item in loc:
            lineContent = lineContent.replace(item, "")

    return lineContent


def read_json() -> list[str]:
    fileContent: list[str] = []
    with animate.Loader(
        desc="Reading original dataset malformed Json", end="List of lines obtained"
    ):
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
    funcRegEx = re.compile(pattern=r"\"func\".*}.*?\"(?=.*\"target\")")
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
    content_len: int = len(content)
    with alive_bar(
        total=content_len,
        title="Creating valid datastructure (JSON)",
        length=60,
        bar="smooth",
    ) as bar:
        for lineIdx in range(content_len):
            local_d = {}
            for key, val in regex_dict.items():
                # apply regex to extract field content
                el = (
                    findall_regex(pattern=val, target=content[lineIdx])[0]
                    if match_regex(pattern=val, target=content[lineIdx])
                    else ""
                )
                # remove unnecessary chars
                el = (
                    re.sub(pattern=sub_regex_dict[key], repl="", string=el)
                    if el
                    else ""
                )

                el = el.strip()
                # in case of "func" field, remove leading and trailing dquotes
                el = el[1:-1] if (el and key == "func") else el
                # in case of "cwe" field, check if there are more codes
                if key == "cwe":
                    el = (
                        [e[1:-1] for e in el.split(sep=",")]
                        if ("," in el)
                        else el[1:-1]
                    )

                local_d[key] = el

            json_dict[lineIdx] = local_d
            bar()

    return json_dict


def remove_unused_fields(
    dic: dict[int, dict[str, str | list[str]]]
) -> dict[int, dict[str, str | list[str]]]:
    lop: list[str] = ["func", "target", "cwe"]  # list of keys to preserve

    # needed because during iteration it is not possible to delete elements
    shrinkedDict: dict[int, dict[str, str | list[str]]] = {}
    localD: dict[str, str | list[str]] = {}
    ##############################
    dic_len: int = len(dic.keys())
    with alive_bar(
        total=dic_len, title="Removing unused fields -> ", length=60, bar="smooth"
    ) as bar:
        for k, v in dic.items():
            localD = {}
            for key in v.keys():
                if key in lop:
                    localD[key] = v[key]
            shrinkedDict[k] = localD
            bar()

    return shrinkedDict


def write_json(dic: dict, output: str) -> None:
    output, _ = os.path.splitext(p=output)
    with open(f"{output}.json", "w") as outfile:
        json.dump(obj=dic, fp=outfile, indent=2, sort_keys=True)

    std_logger.debug(msg="Processed dictionary successfully saved as JSON file")


def create_empty_tmp_source():
    with open("tmp.c", "w") as _:
        pass

    std_logger.debug("Tmp empty file created successfully")


# def populate_tmp_file(filepth: str, dic: dict[int, dict[str, str | list[str]]]):
def populate_tmp_file(func_str_body: str) -> None:
    print(func_str_body)
    regex_functionName: re.Pattern = re.compile(pattern=r"^(.*?)(?=\{)")
    function_name: list[str] | str = re.findall(
        pattern=regex_functionName, string=func_str_body
    )
    try:
        function_name = (
            function_name if isinstance(function_name, str) else function_name[0]
        )
    except:
        if match_regex(pattern=r"^\\n", target=func_str_body):
            func_str_body = re.sub(pattern=r"^\\n", repl="", string=func_str_body)

    std_logger.debug(msg=f"Adding {function_name} to tmp.c file")

    # override content with current function
    with open(file="tmp.c", mode="w") as f:
        f.write(func_str_body)


def spawn_refactor(filepath: str) -> int:
    std_logger.debug("GNU Indent spawned")
    # with animate.Loader(desc="Refactoring "):
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
        std_logger.error(exit_code_obj.stderr)
    else:
        std_logger.info("Refactor successful")

    return exit_code_obj.returncode


def read_file_content_as_str(filepath: str) -> str:
    with open(file=filepath, mode="r") as f:
        file_content: str = f.read()

    return file_content


def build_refactored_json(
    dic: dict[int, dict[str, str | list[str]]], src_pth: str = "tmp.c"
) -> dict[int, dict[str, str | list[str]]]:
    refactored_chunk: str = ""
    # faulty_refactor: list[int] = []
    content_len: int = len(dic.keys())
    with alive_bar(total=content_len, title="", length=60, bar="smooth") as bar:
        for idx, k in enumerate(dic.keys()):
            # str() casting to avoid linting error
            # populate temporary file with current "func" field content
            populate_tmp_file(func_str_body=str(dic[idx]["func"]))

            # spawn refactor on just added function body
            if spawn_refactor(filepath=src_pth):
                std_logger.critical(
                    msg="Problem encountered in refactoring current function"
                )
                pause_exection()

            refactored_chunk = read_file_content_as_str(filepath=src_pth)
            try:
                refactored_chunk = (
                    refactored_chunk
                    if not refactored_chunk[-1] == "\n"
                    else refactored_chunk[:-1]
                )
            except:
                refactored_chunk = refactored_chunk
            dic[k].update({"func": refactored_chunk})
            bar()

    # std_logger.error(f"Refactor not worked in the following indexes: {faulty_refactor}")

    return dic


def pause_exection():
    input("Press enter to continue ...")


def rm_tmp_file(filepath: str) -> None:
    exit_code_obj = subprocess.run(
        args=["rm", f"{filepath}", f"{filepath}~"], capture_output=True, text=True
    )
    if exit_code_obj.returncode:
        std_logger.error(exit_code_obj.stderr)
    else:
        std_logger.info("Tmp file removed successfully")


def add_desc_to_metadata(
    dic: dict[str, str | list[str]], llm
) -> dict[str, str | list[str]]:
    # add an initial time delay to avoid resource exhausted error
    # time.sleep(10)
    # input the function body and get description
    desc: str = llm.generate_description(dic["func"])
    dic.update({"fdesc": desc[1:-1]})

    return dic
