from dataclasses import dataclass

from alive_progress import alive_bar

import utils
from log import std_logger

# from gemini import Gemini

# from pprint import pprint as pp


@dataclass
class Builder:
    def __post_init__(self):
        self.__fileContent: list[str] = utils.read_json()
        # self.gemini = Gemini(model_name="gemini-1.5-pro")
        # self.gemini = Gemini(model_name="gemini-2.0-flash-exp")

    def __filter_metadata_line(self, lineContent: str) -> dict[str, str]:
        # line content is a string representing a line in the Diversevul.json file with all metadata information
        # dictionary of "field_name" : "corpus" pairs
        dof: dict[str, str] = utils.split_lineContent(lineContent=lineContent)
        # remove tabs from func body only: removing tabs should be pretty safe
        dof.update({"func": utils.remove_tabs(lineContent=dof["func"])})
        # remove '\"' chars
        dof.update({"func": utils.remove_escaping_quotes(lineContent=dof["func"])})
        # substitute mutliple spaces with single space
        dof.update({"func": utils.remove_multiplespaces(lineContent=dof["func"])})
        # remove block and inline comments
        # WARNING: order is important, first clear comments and then newline chars
        dof.update({"func": utils.remove_comments(lineContent=dof["func"])})
        # substitute multiple newlines with single newline
        dof.update({"func": utils.remove_multiple_newlines(lineContent=dof["func"])})
        std_logger.info("removing multiple lines")
        dof.update(
            {"message": utils.remove_multiple_newlines(lineContent=dof["message"])}
        )

        return dof

    def __filter_file(self) -> list[dict[str, str]]:
        contentCpy: list[dict[str, str]] = []
        with alive_bar(
            total=len(self.__fileContent),
            title="Fixing lines syntax",
            length=60,
            bar="smooth",
        ) as bar:
            for line in self.__fileContent:
                contentCpy.append(self.__filter_metadata_line(lineContent=line))
                bar()

        return contentCpy

    def __assemble_metadata(self) -> dict[int, dict[str, str | list[str]]]:
        std_logger.debug(msg="Assembling metadata")
        lol: list[dict[str, str]] = self.__filter_file()  # list of lines
        # string organized as dictionary
        std_logger.debug(msg="Creating dictionary dataset")
        meta_d: dict[int, dict[str, str | list[str]]] = utils.create_func_metadatablock(
            content=lol
        )
        # filtering unnecessary fields
        meta_d = utils.remove_unused_fields(dic=meta_d)

        return meta_d

    def __update_json_with_funcdesc(
        self, dic: dict[int, dict[str, str | list[str]]]
    ) -> dict[int, dict[str, str | list[str]]]:
        for k in dic.keys():
            dic.update({k: utils.add_desc_to_metadata(dic=dic[k], llm=self.gemini)})
        return dic

    def run(self):
        # translate string into valid json structure
        dic: dict[int, dict[str, str | list[str]]] = self.__assemble_metadata()
        # create empty temporary file to call "gnu indent refactor" upon
        utils.create_empty_tmp_source()
        # substitute in "func" field refactored string version
        dic = utils.build_refactored_json(dic=dic)
        # remove temp file and copy
        utils.rm_tmp_file(filepath="tmp.c")
        # add description information to metadata
        # dic = self.__update_json_with_funcdesc(dic=dic)
        # save processed dataset as json
        utils.write_json(dic=dic, output="DiverseVul_fixed.json")


if __name__ == "__main__":
    Builder().run()
