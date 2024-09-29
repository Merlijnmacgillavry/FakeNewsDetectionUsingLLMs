import os
import pandas as pd
import utils
from readcalc import readcalc
import multiprocessing as mp
from tqdm import tqdm
import numpy as np


class Merger:
    input_path_option = {
        "dest": "input",
        "type": str,
        "nargs": 1,
        "metavar": "<INPUT PATH>",
        "help": "The path to the folder of the CSVs that need to be combined",
    }
    output_path_option = {
        "dest": "output",
        "type": str,
        "nargs": 1,
        "metavar": "<OUTPUT PATH>",
        "help": "The path to the output CSV file",
    }

    def __init__(self, logger) -> None:
        self.logger = logger

    def add_parser(self, sub_parsers):
        merger_parse = sub_parsers.add_parser(
            "merge", help="combine CSVs to one big CSV"
        )
        merger_parse.add_argument(**self.input_path_option)
        merger_parse.add_argument(**self.output_path_option)
        merger_parse.set_defaults(
            func=lambda args: self.merge(args.input[0], args.output[0])
        )

    def merge(self, input_path, output_path):
        self.logger.info(
            f"Merging all csv files from directory: {input_path} into {output_path}"
        )
        utils.create_file_with_directories(output_path, self.logger)
        csv_files = list_csv_files(input_path)
        self.logger.info(f"Found {len(csv_files)} files to be merged...")
        dfs = []
        for file in csv_files:
            print(file)
            df = pd.read_csv(file, index_col=False)
            dfs.append(df)
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=["content"], keep="first")
        merged_df = compute_readability(merged_df, "content")
        self.logger.info(
            f"Sucessfully merged {len(merged_df)} entries into {output_path}"
        )
        merged_df.to_csv(output_path, mode="a", index=False)


def list_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def calculate_reading_scores(df):
    pass


def compute_readability_for_content(content, index):
    """Computes readability measures for a chunk of the dataframe."""
    results = []
    # for ind, row in chunk.iterrows():
    calc = readcalc.ReadCalc(content, preprocesshtml=None)

    results.append(
        {
            "index": index,
            "smog_index": calc.get_smog_index(),
            "flesch_reading_ease": calc.get_flesch_reading_ease(),
            "flesch_kincaid_grade_level": calc.get_flesch_kincaid_grade_level(),
            "coleman_liau_index": calc.get_coleman_liau_index(),
            "gunning_fog_index": calc.get_gunning_fog_index(),
            "ari_index": calc.get_ari_index(),
            "lix_index": calc.get_lix_index(),
            "dale_chall_score": calc.get_dale_chall_score(),
            "dale_chall_known_fraction": calc.get_dale_chall_known_fraction(),
        }
    )
    return results


def compute_readability(df, col):
    """Computes the readability measures of text in parallel."""
    num_workers = mp.cpu_count()
    df_split = np.array_split(df, 1)
    results = []
    entries = [(row["content"], i) for i, row in df.iterrows()]
    with mp.Pool(num_workers) as pool:
        results += pool.starmap(
            compute_readability_for_content,
            tqdm(entries, total=len(df)),
        )

    # Combine results into the original dataframe
    for result in results:
        for res in result:
            ind = res.pop("index")
            for key, value in res.items():
                df.at[ind, key] = value

    return df
