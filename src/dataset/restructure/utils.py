import json
import os
import re
import subprocess

from alive_progress import alive_bar

import animate
from log import std_logger

# PATH2JSON: str = "../../../DiverseVul/diversevul_20230702.json"
PATH2JSON: str = "../../../DiverseVul/small_diversevul.json"
FIELDS_IN_JSON = 9


def findall_regex(pattern: str | re.Pattern, target: str) -> list[str]:
    regex: str | re.Pattern = (
        re.compile(pattern=pattern) if isinstance(pattern, str) else pattern
    )
    content = re.findall(pattern=regex, string=target)
    return content


def match_regex(pattern: str | re.Pattern, target: str) -> bool:
    regex: str | re.Pattern = (
        re.compile(pattern=pattern) if isinstance(pattern, str) else pattern
    )
    content = re.search(pattern=regex, string=target)
    return True if content else False


def replace(target: str, old: str, new: str) -> str:
    return target.replace(old, new)


def split_lineContent(lineContent: str) -> dict[str, str]:
    func_check_closing_bracket: re.Pattern = re.compile(
        pattern=r"\"func\".*\}\"(?=.*\"target\")"
        # pattern=r"\"func\".*\}\"" @)
    )
    func_check_syntax_block_comment: re.Pattern = re.compile(
        pattern=r"(?<=\")\s+.*?\*/"
    )

    d: dict[str, str] = {}
    regex_dict = {
        # extract "func": "...{ ... }"
        # .*? is necessary since some functions have comments at the end
        "func": re.compile(pattern=r"\"func\".*\"(?=.*\"target\")"),
        # extract "target": \d* "
        "target": re.compile(pattern=r"\"target\"\s*:\s*\d*"),
        # extract "cwe": [whatever is inside] "
        "cwe": re.compile(pattern=r"\"cwe\"\s*:\s*\[.*\](?=.*\"project\")"),
        # extract "project": "prjname"
        "project": re.compile(pattern=r"\"project\"\s*:\s*\".*?\""),
        # extract "commit_id": "alphanumeric"
        "commit_id": re.compile(pattern=r"\"commit_id\"\s*:\s*\".*?\""),
        # extract "hash": numeric
        "hash": re.compile(pattern=r"\"hash\"\s*:\s*\d*"),
        # extract "size": "size"
        "size": re.compile(pattern=r"\"size\"\s*:\s*\d*"),
        # extract "message": "commit_message"
        "message": re.compile(pattern=r"\"message\"\s*:\s*\".*\""),
    }

    for key, reg in regex_dict.items():
        # extract block of metadata ("<field>" : "<body>")
        content = findall_regex(pattern=reg, target=lineContent)  # list[str]
        content = content[0] if content else ""

        if key == "func":

            if not match_regex(pattern=func_check_closing_bracket, target=lineContent):
                # adding closing braket at the end of the function
                content = content[:-1] + "}" + '"'
            if match_regex(pattern=func_check_syntax_block_comment, target=content):
                comment_to_correct = findall_regex(
                    pattern=func_check_syntax_block_comment, target=content
                )[0]
                content = replace(
                    target=content,
                    old=comment_to_correct,
                    new="/*" + comment_to_correct,
                )

        d[key] = content

    return d


def remove_tabs(lineContent: str):
    # remove tabs in function body
    newlineRegEx: re.Pattern = re.compile(pattern=r"\\t")
    lineContent = re.sub(pattern=newlineRegEx, repl=" ", string=lineContent)

    return lineContent


def remove_multiple_newlines(lineContent: str) -> str:
    hashIfRegex: re.Pattern = re.compile(pattern=r"#if")
    hashDefineRegex: re.Pattern = re.compile(pattern=r"#define")
    dowhileMacroRegex: re.Pattern = re.compile(
        pattern=r"#define\s*.*?\(.*?\)\s*do(?:\{|\()\s*.*?(?:\}\)|\})\s*while\(0\)"
    )
    multilineMacroRegex: re.Pattern = re.compile(
        pattern=r"#define\s*.*?\(.*?\)\s*(?:\(\{|\{)\s*.*?(?:\}\)|\})\s*.*?\\\\n\}(?=\\n)"
    )
    hashEndIfDefineRegex: re.Pattern = re.compile(pattern=r"#endif")

    strBlocks: list[str]
    flag: bool = False
    # remove "\\n" within string
    if not (
        match_regex(pattern=multilineMacroRegex, target=lineContent)
        or match_regex(pattern=dowhileMacroRegex, target=lineContent)
    ):
        if match_regex(pattern=r"\\\\n", target=lineContent):
            lineContent = re.sub(
                pattern=r"\\\\n",
                repl="",
                string=lineContent,
            )
    # careful here, cannot simply get rid of \\\\n otherwise spurious \ remain
    else:
        flag = True

    # 1. parse if some pre-processor instructions are there
    if not (
        match_regex(pattern=hashDefineRegex, target=lineContent)
        or match_regex(pattern=hashIfRegex, target=lineContent)
        or match_regex(pattern=hashEndIfDefineRegex, target=lineContent)
    ):
        if match_regex(pattern=r"\\n", target=lineContent):
            lineContent = re.sub(pattern=r"\\n", repl="", string=lineContent)

        return lineContent

    # 2. split line based on "\n" if pre-processor macros are present
    if flag:

        multilineMacro: list[str] = findall_regex(
            pattern=multilineMacroRegex, target=lineContent
        )
        dowhileMacro: list[str] = findall_regex(
            pattern=dowhileMacroRegex, target=lineContent
        )

        tmp: str = ""

        if multilineMacro:
            for i, v in enumerate(multilineMacro):
                if match_regex(pattern=r"\\\\n", target=v):
                    tmp = re.sub(pattern=r"\\\\n", repl="", string=v)
                    tmp = re.sub(pattern=r"(?:\\|\\\\)", repl="", string=tmp)

                lineContent = lineContent.replace(multilineMacro[i], tmp)

        if dowhileMacro:
            tmp: str = ""
            for i, v in enumerate(dowhileMacro):
                if match_regex(pattern=r"\\\\n", target=v):
                    tmp = re.sub(pattern=r"\\\\n", repl="", string=v)
                    tmp = re.sub(pattern=r"(?:\\|\\\\)", repl="", string=tmp)

                lineContent = lineContent.replace(dowhileMacro[i], tmp)

    strBlocks = lineContent.split(sep="\\n")

    # 3. parse block of strings and remove "\n" if no pre-processor instruction is present
    strBlocks = [
        (
            s
            if not (
                match_regex(pattern=hashDefineRegex, target=s)
                or match_regex(pattern=hashIfRegex, target=s)
                or match_regex(pattern=hashEndIfDefineRegex, target=s)
            )
            else "\\n" + s + "\\n"
        )
        for s in strBlocks
    ]

    # 4. merge blocks together
    lineContent = " ".join(strBlocks)

    return lineContent


def remove_escaping_quotes(lineContent: str) -> str:
    opening_quotes: re.Pattern = re.compile(
        pattern=r"(?:(?<=,)|(?<=\()|(?<=\{)|(?:\\t)|(?:\\n))\s*\\\"\s*"
    )
    closing_quotes: re.Pattern = re.compile(pattern=r"\s*\\\"\s*(?=,|\)|})")

    # if no matches, lineContent is returned unchanged
    modified_line: str = re.sub(pattern=opening_quotes, repl='"', string=lineContent)
    modified_line = re.sub(pattern=closing_quotes, repl='"', string=modified_line)

    return modified_line


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
    commentRegEx = re.compile(
        pattern=r"(?:(?<=\\n)|(?<=\{)|(?<=\})|(?<=;)|(?<=\()|(?<=\)))\s*//.*?(?=\\n)"
    )

    lineContent = re.sub(pattern=commentAstRegEx, repl="", string=lineContent)
    lineContent = re.sub(pattern=commentRegEx, repl="", string=lineContent)

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
    content: list[dict[str, str]],
) -> dict[int, dict[str, str | list[str]]]:

    json_dict: dict[int, dict[str, str | list[str]]] = {}
    local_d: dict[str, str | list[str]] = {}
    el: str | list[str]

    # extract "func": ".... }"
    subFuncRegEx = re.compile(pattern=r"\"func\"\s*:\s*")
    # extract "target": \d* "
    subTargetRegEx = re.compile(pattern=r"\"target\"\s*:\s*")
    # extract "cwe": [whatever is inside] "
    subCweRegEx = re.compile(pattern=r"\"cwe\"\s*:\s*\[")
    # extract "project": "prjname"
    subPrjRegEx = re.compile(pattern=r"\"project\"\s*:\s*\"")
    # extract "commit_id": "alphanumeric"
    subCommitRegEx = re.compile(pattern=r"\"commit_id\"\s*:\s*\"")
    # extract "hash": "numeric"
    subHashRegEx = re.compile(pattern=r"\"hash\"\s*:\s*")
    # extract "size": "size"
    subSizeRegEx = re.compile(pattern=r"\"size\"\s*:")
    # extract "message": "commit_message"
    subMsgRegEx = re.compile(pattern=r"\"message\"\s*:\s*\"")

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
            for key in content[lineIdx].keys():
                # remove field name
                el = re.sub(
                    pattern=sub_regex_dict[key], repl="", string=content[lineIdx][key]
                )
                el = el.strip()
                # in case of "func" field, remove leading and trailing dquotes
                el = el[1:-1] if (el and key == "func") else el
                # in case of "cwe" field, check if there are more codes
                if key == "cwe":
                    # if multiple cwe
                    if "," in el:
                        it: list[str] = el.split(sep=",")
                        el = [
                            (
                                # remove spaces and " from 1st up to (n-1)th elements
                                # in case of last element, remove ] too
                                re.sub(pattern=r"\s*", repl="", string=e)[1:-1]
                                if idx != len(it)
                                else re.sub(pattern=r"\s*", repl="", string=e)[1:-2]
                            )
                            for idx, e in enumerate(iterable=it, start=1)
                        ]
                    else:
                        el = el[1:-2]
                if (key == "project") or (key == "commit_id"):
                    # remove final quotes
                    el = el[:-1]

                local_d[key] = el

            json_dict[lineIdx] = local_d
            bar()

    return json_dict


def remove_unused_fields(
    dic: dict[int, dict[str, str | list[str]]],
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


def create_empty_tmp_source(filename: str = "tmp.c") -> None:
    with open(file=filename, mode="w") as _:
        pass

    std_logger.debug("Tmp empty file created successfully")


def populate_tmp_file(func_str_body: str) -> None:
    regex_functionName: re.Pattern = re.compile(pattern=r"^(.*?)(?=\{)")
    function_name: list[str] | str = re.findall(
        pattern=regex_functionName, string=func_str_body
    )

    if function_name:
        function_name = (
            function_name if isinstance(function_name, str) else function_name[0]
        )

    if match_regex(pattern=r"^\\n", target=func_str_body):
        func_str_body = re.sub(pattern=r"^\\n", repl="", string=func_str_body)

    std_logger.debug(msg=f"Adding {function_name} to tmp.c file")

    # at this point, the char "\n" can only be found where pre-processor instructions are
    # split based on that character and enforce new line to avoid refactoring error
    # override content with current function
    with open(file="tmp.c", mode="w") as f:
        try:
            f.writelines("\n".join(func_str_body.split(sep="\\n")))
        except:
            f.write(func_str_body)


def spawn_refactor(filepath: str) -> int:
    # std_logger.debug("GNU Indent spawned")
    # clang-format provided by clangd.
    # Using nvim as editor, I've installed it via Mason
    # for some reason, subprocess cannot run clang-format
    exit_code = os.system(
        command=f"~/.local/share/nvim/mason/bin/clang-format -style=file -i {filepath}"
    )

    if exit_code != 0:
        std_logger.error(msg="Some error has occured")
    # else:
    #     std_logger.info(msg="Refactor successfully accomplished")

    return exit_code


def read_file_content_as_str(filepath: str) -> str:
    with open(file=filepath, mode="r") as f:
        file_content: str = f.read()

    return file_content


def build_refactored_json(
    dic: dict[int, dict[str, str | list[str]]], src_pth: str = "tmp.c"
) -> dict[int, dict[str, str | list[str]]]:
    refactored_chunk: str = ""
    content_len: int = len(dic.keys())
    with alive_bar(total=content_len, title="", length=60, bar="smooth") as bar:
        for idx, k in enumerate(dic.keys()):
            # str() casting to avoid linting error
            # populate temporary file with current "func" field content
            populate_tmp_file(func_str_body=str(dic[idx]["func"]))

            # spawn refactor on just added function body
            if spawn_refactor(filepath=src_pth):
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

    return dic


def pause_exection():
    input("Press enter to continue ...")


def rm_tmp_file(filepath: str) -> None:
    try:
        subprocess.run(
            args=["rm", f"{filepath}", f"{filepath}~"], capture_output=True, text=True
        )
    except:
        return


def add_desc_to_metadata(
    dic: dict[str, str | list[str]], llm
) -> dict[str, str | list[str]]:
    # add an initial time delay to avoid resource exhausted error
    # time.sleep(10)
    # input the function body and get description
    desc: str = llm.generate_description(dic["func"])
    dic.update({"fdesc": desc[1:-1]})

    return dic
