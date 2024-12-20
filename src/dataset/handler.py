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
        # if message flag not set, check if current line matches the "message" field
        if not self.matchedMessageFlag:
            self.matchedMessageFlag = True if utils.match_regex(
                pattern=r'"message"', target=lineContent) else False

        # remove tabs
        lineContent = utils.remove_tabs(lineContent=lineContent)
        # remove newline char
        lineContent, self.matchedMessageFlag = utils.remove_newlines(
            lineContent=lineContent, matchedMessageFlag=self.matchedMessageFlag)
        # substitute mutliple spaces with single space
        lineContent = utils.remove_multiplespaces(lineContent=lineContent)
        # remove comments
        lineContent = utils.remove_comments(lineContent=lineContent)
        # remove "\"
        lineContent = utils.remove_backslashes(lineContent=lineContent)

        return lineContent

    def filter_file(self):
        contentCpy: list[str] = []
        for idx, line in enumerate(self.__fileContent):
            contentCpy.append(self.__filter_metadata_line(line))

        return contentCpy

    def run(self):
        pp(self.filter_file())

        # pp(self.__fileContent)
        # pp(self.__filter_metadata_line(self.__fileContent[20]))
        # pp(self.__filter_metadata_line(self.__fileContent[21]))


if __name__ == "__main__":
    Builder().run()
