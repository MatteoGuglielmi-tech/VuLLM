import csv
import json
import logging
import pandas as pd
from dataclasses import dataclass
from datasets import Dataset
from tqdm import tqdm


logger = logging.getLogger(name=__name__)


@dataclass
class DatasetHandler:
    JSON_source_dataset: str = "../../../Dataset/prepro_diversvul.json"
    CSV_output_dataset: str = "../../../Dataset/preproc_diversvul.csv"

    def JSON_read_file_content(self) -> dict[str, dict[str, str | list[str]]]:
        with open(file=self.JSON_source_dataset, mode="r") as jf:
            json_content = json.load(fp=jf)

        return json_content

    def CSV_write_entries(self) -> None:
        json_content: dict[str, dict[str, str | list[str]]] = (
            self.JSON_read_file_content()
        )
        column_names: list[str] = list(json_content["0"].keys())
        with open(
            file=self.CSV_output_dataset, mode="w", newline="", encoding="utf-8-sig"
        ) as cf:
            # initialize csv writer object + removing ^M in lineterminator ("\r")
            writer_obj = csv.writer(cf, lineterminator="\n")
            # write headers
            writer_obj.writerow(column_names)
            # write line by line content
            for item in tqdm(
                iterable=json_content.values(),
                desc="Writing Rows",
                total=len(json_content.values()),
                unit=" element",
                ncols=200,
            ):
                writer_obj.writerow(item.values())

    def CSV_load_in_RAM(self) -> None:
        self.df: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self.CSV_output_dataset,
            # index_col="Unnamed: 0",
            delimiter=",",
            skip_blank_lines=True,
        ).fillna("")

    def CSV_shuffle_dataframe(self) -> None:
        self.df = self.df.sample(frac=1, random_state=85).reset_index(drop=True)

    def CSV_split_dataframe(
        self, train_size: float = 0.8, eval_size: float = 0.1
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Calculate sizes
        train_end = int(train_size * len(self.df))
        eval_end = train_end + int(eval_size * len(self.df))

        # Split the data
        X_train: pd.DataFrame = pd.DataFrame(self.df[:train_end])
        X_eval: pd.DataFrame = pd.DataFrame(self.df[train_end:eval_end])
        X_test: pd.DataFrame = pd.DataFrame(self.df[eval_end:])

        return X_train, X_eval, X_test

    # Define the prompt generation functions
    def formatting_prompts_func(
        self, data_point: pd.Series, inference_mode: bool = False
    ):
        if not inference_mode:
            prompt: str = (
                f"""Below an instruction that describes the task, an input sample followed by the desired response.\n\n# Instruction:\nYou are a binary text classifier. Classify the input text, representing a C/C++ function, into vulnerable [1] or non-vulnerable [0] and return the answer as the corresponding binary label.\n\n# Input:\n{data_point["func"]}\n\n# Response:\n{data_point["target"]}"""
            )
        else:
            prompt: str = (
                f"""Below an instruction that describes the task, an input sample followed by the desired response.\n\n# Instruction:\nYou are a binary text classifier. Classify the input text, representing a C/C++ function, into vulnerable [1] or non-vulnerable [0] and return the answer as the corresponding binary label.\n\n# Input:\n{data_point["func"]}\n\n# Response:\n"""
            )

        return prompt

    def CSV_add_prompt_column(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.CSV_shuffle_dataframe()
        X_train, X_eval, X_test = self.CSV_split_dataframe()

        # Generate prompts for training data
        tqdm.pandas(desc="Generating training prompts", unit=" entry", ncols=200)
        # note: column "text" cannot be renamed
        X_train.loc[:, "text"] = X_train.progress_apply(
            self.formatting_prompts_func, axis=1
        )

        # Generate prompts for evaluation data
        tqdm.pandas(desc="Generating evaluation prompts", unit=" entry", ncols=200)
        X_eval.loc[:, "text"] = X_eval.progress_apply(
            self.formatting_prompts_func, axis=1
        )
        # Generate test prompts and extract true labels
        y_true = X_test.loc[:, "target"]
        X_test = pd.DataFrame(
            X_test.apply(self.formatting_prompts_func, axis=1, args=(True,)),
            columns=["text"],
        )

        X_test["y_true"] = y_true

        del y_true

        return X_train, X_eval, X_test

    def DATAFRAME_check_label_distribution(self):
        train, eval, test = self.CSV_add_prompt_column()
        # check labels distribution
        print(f"Label distribution : \n {dh.df.loc[:, "target"].value_counts()}")
        print(
            f"Training label distribution : \n {train.loc[:, "target"].value_counts()}"
        )
        print(f"Eval label distribution : \n {eval.loc[:, "target"].value_counts()}")
        print(f"Test label distribution : \n {test.loc[:, "y_true"].value_counts()}")

    def DATAFRAME_convert_to_hf_dataset(self, df: pd.DataFrame) -> Dataset:
        # Convert pd df to hf datasets
        return Dataset.from_pandas(pd.DataFrame(df[["text"]]))

    def HF_DATASET_run_pipeline(self) -> tuple[Dataset, Dataset, pd.DataFrame]:
        # serialize JSON content as CSV
        self.CSV_write_entries()
        # load serialized file to RAM
        self.CSV_load_in_RAM()
        # add prompt and split dataset
        train, eval, test = self.CSV_add_prompt_column()
        # convert train and evaluation sets to hf format
        # compatible with SFTTrainer()
        logger.info("Converting training data to huggingface format")
        hf_train_data: Dataset = self.DATAFRAME_convert_to_hf_dataset(df=train)
        logger.info("Converting eval data to huggingface format")
        hf_eval_data: Dataset = self.DATAFRAME_convert_to_hf_dataset(df=eval)

        del train, eval

        return hf_train_data, hf_eval_data, test


if __name__ == "__main__":
    dh: DatasetHandler = DatasetHandler()
    dh.CSV_write_entries()
    dh.CSV_load_in_RAM()

    train, eval, test = dh.CSV_add_prompt_column()

    dh.DATAFRAME_convert_to_hf_dataset(df=train)
    dh.DATAFRAME_convert_to_hf_dataset(df=eval)

    # print(train["text"][3])
