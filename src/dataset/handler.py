import json
from dataclasses import dataclass
from pprint import pprint as pp

import utils


@dataclass
class Builder():
    def __post_init__(self):
        self.__fileContent: list[str] = utils.read_json()
        self.matchedMessageFlag: bool = False

    def __filter_metadata_line(self, lineContent: str) -> str:
        # remove tabs
        lineContent = utils.remove_tabs(lineContent=lineContent)
        # remove newline char
        lineContent = utils.remove_newlines(lineContent=lineContent)
        # substitute mutliple spaces with single space
        lineContent = utils.remove_multiplespaces(lineContent=lineContent)
        # remove "\"
        lineContent = utils.remove_backslashes(lineContent=lineContent)
        # # remove comments
        lineContent = utils.remove_comments(lineContent=lineContent)

        return lineContent

    def filter_file(self) -> list[str]:
        contentCpy: list[str] = [self.__filter_metadata_line(
            line) for line in self.__fileContent]
        return contentCpy

    def run(self):
        lol = self.filter_file()
        # print(lol)
        lol = utils.create_func_metadatablock(lol)
        # pp(lol)
        lol = utils.remove_unused_fiels(dic=lol)
        # pp(lol)
        utils.create_empty_tmp_source()
        # pp(lol)
        utils.populate_tmp_file(filepth="tmp.c", dic=lol)


if __name__ == "__main__":
    Builder().run()
