import json
import os
import pickle
import re
import subprocess

from alive_progress import alive_bar

import animate
import argparser
from log import std_logger
from treesitter import TreeSitter

FIELDS_IN_JSON: int = 9

ts: TreeSitter = TreeSitter()


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


def pop_regex(
    pattern: str | re.Pattern, target: str, repl: str = ""
) -> tuple[str, str | list[str]]:
    matches: list[str] = findall_regex(pattern=pattern, target=target)
    slice: str = re.sub(pattern=pattern, repl=repl, string=target)

    return slice, matches


def replace(target: str, old: str, new: str) -> str:
    return target.replace(old, new)


def split_lineContent(lineContent: str) -> dict[str, str]:
    d: dict[str, str] = {}
    el: str

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

    sub_regex_dict: dict[str, re.Pattern] = {
        # remove "func":
        "func": re.compile(pattern=r"\"func\"\s*:\s*"),
        # remove "target":
        "target": re.compile(pattern=r"\"target\"\s*:\s*"),
        # remove "cwe":
        "cwe": re.compile(pattern=r"\"cwe\"\s*:\s*\["),
        # remove "project":
        "project": re.compile(pattern=r"\"project\"\s*:\s*\""),
        # remove "commit_id":
        "commit_id": re.compile(pattern=r"\"commit_id\"\s*:\s*\""),
        # remove "hash":
        "hash": re.compile(pattern=r"\"hash\"\s*:\s*"),
        # remove "size":
        "size": re.compile(pattern=r"\"size\"\s*:"),
        # remove "message":
        "message": re.compile(pattern=r"\"message\"\s*:\s*\""),
    }

    for key, reg in regex_dict.items():
        # extract block of metadata ("<field>" : "<body>")
        content = findall_regex(pattern=reg, target=lineContent)  # list[str]
        if not content:
            raise ValueError('Cannot find "func" field')

        content = content[0]

        el = re.sub(pattern=sub_regex_dict[key], repl="", string=content).strip()
        # in case of "func" field, remove leading and trailing quotes
        el = el[1:-1] if (el and key == "func") else el

        if (key == "project") or (key == "commit_id"):
            # remove final quotes
            el = el[:-1]

        d[key] = el

    return d


def remove_tabs(lineContent: str):
    # remove tabs in function body
    newlineRegEx: re.Pattern = re.compile(pattern=r"\\t")
    lineContent = re.sub(pattern=newlineRegEx, repl=" ", string=lineContent)

    return lineContent


def fix_func_proto(lineContent: str) -> str:
    subFuncNameRegEx: re.Pattern = re.compile(pattern=r".*?(?=\{)")
    try:
        func_prototype: str = findall_regex(
            pattern=subFuncNameRegEx, target=lineContent
        )[0]
    except:
        std_logger.critical(msg=f"Strange function prototype !!")
        populate_tmp_file(func_str_body=lineContent)
        pause_exection(msg="Correct function issue(s) and press enter to continue ...")
        # read fixed line and proceed
        lineContent = read_file_content_as_str(filepath="tmp.c")
    else:
        if func_prototype:
            # remove old proto
            lineContent = lineContent.replace(func_prototype, "")
            # filter func_prototype
            func_prototype = parse_func_proto(decl=func_prototype)
            # re-add corrected function signature
            lineContent = func_prototype + lineContent

    return lineContent


def remove_multiple_newlines(lineContent: str) -> str:
    goto: bool = False

    # this can be achieved via tree-sitter as well
    # ===================== PATTERN DEFINITTIONS ======================================
    hashIfRegex: re.Pattern = re.compile(pattern=r"#\s*if")
    hashElseRegex: re.Pattern = re.compile(pattern=r"#\s*else")
    hashElifRegex: re.Pattern = re.compile(pattern=r"#\s*elif")
    hashEndIfRegex: re.Pattern = re.compile(pattern=r"#\s*endif")
    hashUndefRegex: re.Pattern = re.compile(pattern=r"#\s*undef")
    hashDefineRegex: re.Pattern = re.compile(pattern=r"#\s*define")
    hashIncludeRegex: re.Pattern = re.compile(pattern=r"#\s*include")
    # 305 steps required
    dowhileMacroRegex: re.Pattern = re.compile(pattern=r"#define.*?do\{.*?while\(0\)")
    # 345 steps required
    multilineMacroRegex: re.Pattern = re.compile(pattern=r"#define.*?\{.*?\\\\n\}")
    # =================================================================================

    # fix prototype immediately
    lineContent = fix_func_proto(lineContent=lineContent)

    # check for correctly matched curly braces this comparison may not be valid
    # although, in those cases a mismatched is very likely to be detected
    ts.parse_input(code_snippet=lineContent)  # update internal tree
    if ts.is_closing_curvy_needed() or lineContent[-1] != "}":
        # adding closing braket at the end of the function
        lineContent = lineContent + "}"

    # ===================== PATTERNS APPLICATION ======================================
    list_defines: list[str] = findall_regex(pattern=hashDefineRegex, target=lineContent)
    list_ifs: list[str] = findall_regex(pattern=hashIfRegex, target=lineContent)
    list_elses: list[str] = findall_regex(pattern=hashElseRegex, target=lineContent)
    list_elifs: list[str] = findall_regex(pattern=hashElifRegex, target=lineContent)
    list_endifs: list[str] = findall_regex(pattern=hashEndIfRegex, target=lineContent)
    list_undefs: list[str] = findall_regex(pattern=hashUndefRegex, target=lineContent)
    list_includes: list[str] = findall_regex(
        pattern=hashIncludeRegex, target=lineContent
    )
    # =================================================================================

    # 1. parse if some pre-processor instructions are there
    if not (
        list_defines
        or list_ifs
        or list_elses
        or list_elifs
        or list_endifs
        or list_undefs
        or list_includes
    ):
        lineContent = re.sub(pattern=r"\\n", repl=" ", string=lineContent)
        # this pattern should be found only inside strings
        lineContent = re.sub(pattern=r"\\\\n", repl="", string=lineContent)

        # return lineContent
        goto = True

    if not goto:
        multilineMacros: list[str] = findall_regex(
            pattern=multilineMacroRegex, target=lineContent
        )
        dowhileMacros: list[str] = findall_regex(
            pattern=dowhileMacroRegex, target=lineContent
        )

        # If the pattern isn’t found, string is returned unchanged.
        lineContent = re.sub(pattern=r"\\\\n", repl=" ", string=lineContent)

        # careful here, cannot simply get rid of \\\\n otherwise spurious \ remain
        if multilineMacros or dowhileMacros:
            tmp: str = ""

            # hashDefineRegex matches, also multilineMacroRegex and dowhileMacroRegex do
            for i, multiline in enumerate(multilineMacros):
                if match_regex(pattern=r"do\{", target=multiline):
                    continue
                # remove every \ to go to the next line an let the refactor
                # do all the work
                # tmp = re.sub(pattern=r"\\\\n", repl="", string=multiline)
                tmp = re.sub(pattern=r"(?:\\|\\\\)", repl="", string=multiline)

                lineContent = lineContent.replace(multilineMacros[i], tmp)

            for i, dowhile in enumerate(dowhileMacros):
                # tmp = re.sub(pattern=r"\\\\n", repl="", string=dowhile)
                tmp = re.sub(pattern=r"(?:\\|\\\\)", repl="", string=dowhile)

                lineContent = lineContent.replace(dowhileMacros[i], tmp)

        # 2. split line based on "\n" if pre-processor macros are present
        strBlocks: list[str] = lineContent.split(sep="\\n")
        # print(strBlocks)

        # 3. parse block of strings and remove "\n" if no pre-processor instruction is present
        strBlocks = [
            (
                s
                if not (
                    match_regex(pattern=hashDefineRegex, target=s)
                    or match_regex(pattern=hashIfRegex, target=s)
                    or match_regex(pattern=hashElseRegex, target=s)
                    or match_regex(pattern=hashElifRegex, target=s)
                    or match_regex(pattern=hashEndIfRegex, target=s)
                    or match_regex(pattern=hashUndefRegex, target=s)
                    or match_regex(pattern=hashIncludeRegex, target=s)
                )
                # in #include instructions, the quotes are not matched by precvious regex
                # idx = 0 will always contain "func and the name of it"
                else (
                    (re.sub(pattern=r"\\\"", repl='"', string=s) + "\\n")
                    if idx == 0
                    else ("\\n" + re.sub(pattern=r"\\\"", repl='"', string=s) + "\\n")
                )
            )
            for idx, s in enumerate(strBlocks)
        ]

        # 4. merge blocks together
        lineContent = " ".join(strBlocks)

        # here I want to check if some pre-processor instructions
        # ain't properly matched
        if len(list_ifs) != len(list_endifs):
            # manually check the error
            std_logger.critical(
                msg=f"Mismatched pre-processor instruction:"
                f"#if: {len(list_ifs)}, #endif: {len(list_endifs)}"
            )
            populate_tmp_file(func_str_body=lineContent)
            pause_exection(
                msg="Correct function issue(s) and press enter to continue ..."
            )
            # read fixed line and proceed
            lineContent = read_file_content_as_str(filepath="tmp.c")

    return lineContent


def parse_func_proto(decl: str) -> str:
    # mistakes emprically verified
    func_prototype = remove_tabs(lineContent=decl)
    func_prototype = remove_comments(lineContent=func_prototype)
    func_prototype = re.sub(pattern=r"\\n", repl=" ", string=func_prototype)
    func_prototype = re.sub(pattern=r"\*/\s*", repl="", string=func_prototype)
    func_prototype = re.sub(pattern=r"/\*\s*", repl="", string=func_prototype)

    ts.parse_input(code_snippet=func_prototype)
    func_prototype = ts.replace_error_nodes(src=func_prototype)

    return func_prototype


def remove_escaping_quotes(lineContent: str) -> str:
    # remove the escape character for all opening double quotes
    rm_esc_quotes: re.Pattern = re.compile(pattern=r"(?<!\\)\\\"")
    # replace \\\" with \"
    rm_multiple_esc: re.Pattern = re.compile(pattern=r"\\\\\"")

    # if no matches, lineContent is returned unchanged
    modified_line: str = re.sub(pattern=rm_esc_quotes, repl='"', string=lineContent)
    modified_line = re.sub(pattern=rm_multiple_esc, repl='"', string=modified_line)

    return modified_line


def remove_multiplespaces(lineContent: str) -> str:
    # remove multiple consecutive spaces in function body
    multipleSpacesRegEx: re.Pattern = re.compile(pattern=r"\s+(?=.)")
    lineContent = re.sub(pattern=multipleSpacesRegEx, repl=" ", string=lineContent)

    return lineContent


def remove_comments(lineContent: str) -> str:
    ts.parse_input(code_snippet=lineContent)
    comments: list[bytes] = ts.extract_comments()

    if comments:
        for comment in comments:
            str_cmnt = comment.decode(encoding="utf-8").__repr__()[1:-1]
            str_cmnt = re.sub(pattern=r"\\\\", repl=r"\\", string=str_cmnt)
            lineContent = lineContent.replace(str_cmnt, "")

    return lineContent


def dif_two_strings(s1, s2):
    import difflib

    s = difflib.SequenceMatcher(None, s1, s2, autojunk=False)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag != "equal":
            print(
                "{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}".format(
                    tag, i1, i2, j1, j2, s1[i1:i2], s2[j1:j2]
                )
            )


def read_json() -> list[str]:
    fileContent: list[str] = []

    with animate.Loader(
        desc="Reading original dataset malformed Json", end="List of lines obtained"
    ):
        with open(file=argparser.args.path, mode="r") as json:
            fileContent = json.readlines()

    start_idx = (argparser.args.start_idx) if argparser.args.start_idx > 0 else 0
    return fileContent[start_idx:]


def create_func_metadatablock(
    content: list[dict],  # list[dict[str, str]]
) -> dict[int, dict[str, str | list[str]]]:

    json_dict: dict[int, dict[str, str | list[str]]] = {}

    if os.path.isfile(path=argparser.args.file_name) and not argparser.args.clear_json:
        existing_content: dict[int, dict[str, str | list[str]]] = _load_backup()
        json_dict = existing_content.copy()
        del existing_content

    content_len: int = len(content)
    with alive_bar(
        total=content_len,
        title="Creating valid JSON",
        length=60,
        bar="smooth",
    ) as bar:
        for lineIdx in range(content_len):
            el: list[str] | str
            # for "cwe" field, check if there are more codes
            if "," in content[lineIdx]["cwe"]:
                it: list[str] = content[lineIdx]["cwe"].split(sep=",")
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
                el = content[lineIdx]["cwe"][1:-2]

            content[lineIdx].update({"cwe": el})
            json_dict[(lineIdx + argparser.args.start_idx)] = content[lineIdx]
            bar()

    return json_dict


def remove_unused_fields(
    dic: dict[int, dict[str, str | list[str]]],
) -> dict[int, dict[str, str | list[str]]]:
    lop: list[str] = ["func", "target", "cwe", "project"]  # list of keys to preserve

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


def write_json(dic: dict) -> None:
    with open(file=argparser.args.file_name, mode="w", encoding="utf-8") as outfile:
        json.dump(obj=dic, fp=outfile, indent=2, sort_keys=False)


def _clear_file_content(filename: str) -> None:
    open(filename, mode="w").close()


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
        function_name = re.sub(pattern=r"\\n", repl=" ", string=function_name)

    if match_regex(pattern=r"^\\n", target=func_str_body):
        func_str_body = re.sub(pattern=r"^\\n", repl="", string=func_str_body)

    if argparser.args.debug:
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

    return exit_code


def read_file_content_as_str(filepath: str) -> str:
    with open(file=filepath, mode="rb") as f:
        file_content: bytes = f.read()

    return file_content.decode(encoding="utf-8")


def build_refactored_json(
    dic: dict[int, dict[str, str | list[str]]], src_pth: str = "tmp.c"
) -> None:
    refactored_chunk: str = ""
    content_len: int = len(dic.keys())
    running_d: dict = {}

    if argparser.args.clear_json:
        _clear_file_content(filename=argparser.args.file_name)
        _delete_backup()

    with alive_bar(total=content_len, title="", length=60, bar="smooth") as bar:
        for idx, k in enumerate(dic.keys()):
            assert isinstance(
                dic[idx]["func"], str
            ), "For some reason, the `func` field is not a string"

            populate_tmp_file(func_str_body=str(dic[idx]["func"]))

            if spawn_refactor(filepath=src_pth):
                pause_exection()

            refactored_chunk = _remove_spurious_escape(
                refactored_chunk=read_file_content_as_str(filepath=src_pth)
            )

            try:
                refactored_chunk = (
                    refactored_chunk
                    if not refactored_chunk[-1] == "\n"
                    else refactored_chunk[:-1]
                )
            except:
                refactored_chunk = refactored_chunk

            dic[k].update({"func": refactored_chunk})
            running_d[k] = dic[k]
            _save_backup(obj=running_d)
            bar()

    write_json(dic)


def _remove_spurious_escape(refactored_chunk: str) -> str:
    refactored_chunk = re.sub(pattern=r"\\(?!n)\s*", repl=r"", string=refactored_chunk)

    return refactored_chunk


def _save_backup(obj: dict) -> None:
    with open(file="intrmd_bkup.pkl", mode="wb") as fp:
        pickle.dump(obj=obj, file=fp)

    if argparser.args.debug:
        std_logger.info(msg="Pickling successful")


def _load_backup() -> dict[int, dict[str, str | list[str]]]:
    with open(file="intrmd_bkup.pkl", mode="rb") as fp:
        obj = pickle.load(file=fp)
    return obj


def _delete_backup() -> None:
    if os.path.exists(path="intrmd_bkup.pkl"):
        os.remove(path="intrmd_bkup.pkl")


def pause_exection(msg: str = "Press enter to continue ..."):
    input(msg)


def rm_tmp_file(filepath: str = "tmp.c") -> None:
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
