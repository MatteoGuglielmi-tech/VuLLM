import json
import re

PATH2JSON: str = "../../DiverseVul/small_diversevul.json"


def remove_tabs(lineContent: str):
    # remove tabs in function body
    newlineRegEx: re.Pattern = re.compile(pattern=r"\\t")
    lineContent = re.sub(pattern=newlineRegEx,
                         repl=" ", string=lineContent)

    return lineContent


def findall_regex(pattern: str, target: str) -> str:
    regex: re.Pattern = re.compile(pattern=pattern)
    content = re.findall(pattern=regex, string=target)
    return content[0] if content else ""


def match_regex(pattern: str, target: str) -> bool:
    regex: re.Pattern = re.compile(pattern=pattern)
    content = re.search(pattern=regex, string=target)
    return True if content else False


def remove_newlines(lineContent: str, matchedMessageFlag: bool) -> tuple[str, bool]:
    # do not remove newline char if we encountered "message" since it indicates we are at the
    # end of the function metadata block. Let's keep division between function metadata
    if (not matchedMessageFlag) or (matchedMessageFlag and not match_regex(pattern=r"^}\\n$", target=lineContent)):
        # remove "\n" char in function body
        newlineRegEx: re.Pattern = re.compile(
            pattern=r"\\n")
        lineContent = re.sub(pattern=newlineRegEx,
                             repl='', string=lineContent)
        # remove "newline" char in function body
        # newlineRegEx: re.Pattern = re.compile(
        # pattern=r"\n")
        # lineContent = re.sub(pattern=newlineRegEx,
        # repl='', string=lineContent)

    elif (matchedMessageFlag and match_regex(pattern=r"^}\\n$", target=lineContent)):
        matchedMessageFlag = False

    return lineContent, matchedMessageFlag


def remove_backslashes(lineContent: str) -> str:
    obj: str = findall_regex(pattern=r"\\", target=lineContent)
    return lineContent if not obj else lineContent.strip().replace(obj, "")


def remove_multiplespaces(lineContent: str) -> str:
    # remove multiple consecutive spaces in function body
    multipleSpacesRegEx: re.Pattern = re.compile(pattern=r"\s+")
    lineContent = re.sub(pattern=multipleSpacesRegEx,
                         repl=" ", string=lineContent)

    return lineContent


def remove_comments(lineContent: str) -> str:

    def __commentMatching_handler(comment: str, line: str) -> str:
        tmpLine: str = ""
        if comment != line.strip():  # not prefect match, substitute comment with space
            tmpLine = line.strip().replace(comment[0], " ")
        else:
            # in post-processing, empty elements will be popped (filter(lambda x: x != ""), list)
            tmpLine = ""
        return tmpLine

    # extract comment encapsulated in /**/
    commentRegEx: re.Pattern = re.compile(pattern=r"/\*.*\*/")
    comment = re.findall(pattern=commentRegEx, string=lineContent.strip())
    if comment:
        lineContent = __commentMatching_handler(
            comment=comment[0], line=lineContent)
    else:
        # check for comments starting with // and match until the eol
        commentRegEx: re.Pattern = re.compile(pattern=r"//.*")
        comment = re.findall(pattern=commentRegEx,
                             string=lineContent.strip())
        if comment:
            lineContent = __commentMatching_handler(
                comment=comment[0], line=lineContent)

    return lineContent


def read_json() -> list[str]:
    fileContent: list[str] = []
    with open(PATH2JSON, "r") as json:
        fileContent = json.readlines()
    return fileContent
