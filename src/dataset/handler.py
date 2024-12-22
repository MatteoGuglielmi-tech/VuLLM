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

    def __assemble_metadata(self) -> dict[int, dict[str, str | list[str]]]:
        lol: list[str] = self.filter_file()  # list of lines
        # string organized as dictionary
        meta_d: dict[int, dict[str, str | list[str]]
                     ] = utils.create_func_metadatablock(content=lol)
        # filtering unnecessary fields
        meta_d = utils.remove_unused_fields(dic=meta_d)

        return meta_d

    def run(self):
        # translate string into valid json structure
        dic: dict[int, dict[str, str | list[str]]] = self.__assemble_metadata()
        # create empty temporary file to call "gnu indent refactor" upon
        utils.create_empty_tmp_source()
        # populate temporary file with dictionary "func" fields content
        utils.populate_tmp_file(filepth="tmp.c", dic=dic)
        # call refactor on temporary file
        utils.spawn_refactor(filepath="tmp.c")
        # substitute in "func" field refactored string version
        dic = utils.build_refactored_json(dic=dic, src_pth="tmp.c")
        utils.rm_tmp_file(filepath="tmp.c")
        pp(dic)


if __name__ == "__main__":
    Builder().run()
