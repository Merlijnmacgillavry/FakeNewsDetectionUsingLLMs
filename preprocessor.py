from argparse import _SubParsersAction, ArgumentParser
import os
import json
import pandas as pd
import utils
from tqdm import tqdm
from nltk import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import multiprocessing as mp
import nltk

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("wordnet")

columns = ["topic", "label", "length", "sentiment", "content"]

analyzer = SentimentIntensityAnalyzer()


class Preprocessor:
    path_option = {
        "dest": "path",
        "type": str,
        "nargs": 1,
        "metavar": "<PATH>",
        "help": "The input path to the dataset",
    }

    def __init__(self, logger) -> None:
        self.logger = logger
        dataset_options = self.get_preprocess_functions()
        self.dataset_option = {
            "dest": "dataset",
            "type": str,
            "nargs": 1,
            "metavar": "<DATASET>",
            "help": "The type of the dataset (Either 'FakeNewsNet' or 'FakeHealth' or 'MOCHEG)",
            "choices": dataset_options,
        }

    def get_preprocess_functions(self):
        methods = [method for method in dir(self) if not method.startswith("_")]
        available_options = [
            method for method in methods if method.startswith("preprocess_")
        ]
        return map(lambda x: x.split("preprocess_")[1], available_options)

    def add_parser(self, sub_parsers):
        preprocess_parse = sub_parsers.add_parser(
            "preprocess", help="Preprocess datasets for prompting and data analysis"
        )
        preprocess_parse.add_argument(**self.dataset_option)
        preprocess_parse.add_argument(**self.path_option)
        try:
            preprocess_parse.set_defaults(
                func=lambda args: self._preprocess(args.dataset[0], args.path[0])
            )

        except ValueError as error:
            self.logger.error(f"Value error in preprocessing command: {error}")

    # main preprocess function:
    def _preprocess(self, type: str, path: str):
        method_name = f"preprocess_{type}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(path)
        else:
            raise ValueError(f"Invalid type: {type}")

    def preprocess_MOCHEG(self, path: str):
        output_path = "./data/out/preprocess/MOCHEG.csv"
        self.logger.info(f"Preprocessing MOCHEG")
        utils.create_file_with_directories(output_path, self.logger)
        df = pd.DataFrame(columns=columns)
        entries_path = os.path.join(path, "Corpus2.csv")
        entries = pd.read_csv(entries_path)
        total_entries = len(entries)
        entries = entries.drop_duplicates(subset=["claim_id"], keep="first")
        unique_entries = len(entries)
        self.logger.warning(
            f"Removed {total_entries - unique_entries} duplicate entries from FakeHealth"
        )
        df = _normalize_MOCHEG(path, df, entries)
        self.logger.info(f"Performing sentiment analysis:")
        df = _sentiment_analysis(df)
        df.to_csv(output_path, mode="a", index_label="index")
        self.logger.info(f"Sucessfully preprocessed MOCHEG")

    def preprocess_FakeHealth(self, path: str):
        output_path = "./data/out/preprocess/FakeHealth.csv"
        self.logger.info(f"Preprocessing FakeHealth...")
        utils.create_file_with_directories(output_path, self.logger)
        health_release_path = os.path.join(path, "HealthRelease")
        health_story_path = os.path.join(path, "HealthStory")
        df = pd.DataFrame(columns=columns)
        self.logger.info(f"Preprocessing HealthRelease:")
        df = _normalize_FakeHealth(path, "HealthRelease", df)
        self.logger.info(f"Preprocessing HealthStory:")
        df = _normalize_FakeHealth(path, "HealthStory", df)
        total_entries = len(df)
        df = df.drop_duplicates(subset=["content"], keep=False)
        unique_entries = len(df)
        self.logger.warning(
            f"Removed {total_entries - unique_entries} duplicate entries from FakeHealth"
        )
        self.logger.info(f"Performing sentiment analysis:")
        df = _sentiment_analysis(df)
        df.to_csv(output_path, mode="a", index_label="index")
        self.logger.info(f"Sucessfully preprocessed FakeHealth")

    def preprocess_FakeNewsNet(self, path: str):
        output_path = "./data/out/preprocess/FakeNewsNet.csv"
        self.logger.info(f"Preprocessing FakeNewsNet...")
        utils.create_file_with_directories(output_path, self.logger)
        politifact_fake_path = os.path.join(path, "politifact", "fake")
        politifact_real_path = os.path.join(path, "politifact", "real")
        gossipcop_fake_path = os.path.join(path, "gossipcop", "fake")
        gossipcop_real_path = os.path.join(path, "gossipcop", "real")
        df = pd.DataFrame(columns=columns)
        self.logger.info(f"Preprocessing politifact fake news:")
        df = _normalize_FakeNewsNet_new(politifact_fake_path, "politics", 1, df)
        self.logger.info(f"Preprocessing politifact real news:")
        df = _normalize_FakeNewsNet_new(politifact_real_path, "politics", 0, df)
        self.logger.info(f"Preprocessing gossipcop fake news:")
        df = _normalize_FakeNewsNet_new(gossipcop_fake_path, "entertainment", 1, df)
        self.logger.info(f"Preprocessing gossipcop real news:")
        df = _normalize_FakeNewsNet_new(gossipcop_real_path, "entertainment", 0, df)
        total_entries = len(df)
        df = df.drop_duplicates(subset=["content"], keep=False)
        unique_entries = len(df)
        self.logger.warning(
            f"Removed {total_entries - unique_entries} duplicate entries from FakeNewsNet"
        )
        self.logger.info(f"Performing sentiment analysis:")
        df = _sentiment_analysis(df)
        df.to_csv(output_path, mode="a", index_label="index")
        self.logger.info(f"Sucessfully preprocessed FakeNewsNet")


def _normalize_FakeHealth(path: str, partition: str, df: pd.DataFrame):
    reviews_path = os.path.join(path, "reviews", f"{partition}.json")
    data_path = os.path.join(path, partition)
    reviews = pd.read_json(reviews_path, orient="records")[["news_id", "rating"]]
    reviews["news_id"] = reviews["news_id"].apply(_remove_prefix)
    entries = os.listdir(data_path)
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results += pool.starmap(
            _normalize_FakeHealth_entry,
            tqdm(
                [(path, partition, entry, reviews) for entry in entries],
                total=len(entries),
            ),
        )
    filtered_results = [result for result in results if result is not None]
    df = df._append(filtered_results, ignore_index=True)
    return df


def _normalize_FakeHealth_entry(path: str, partition: str, entry: str, reviews):
    entry_path = os.path.join(path, partition, entry)
    if os.path.exists(entry_path):
        with open(entry_path, "r") as f:
            json_data = f.read()
            data = json.loads(json_data)
            text = data["text"]
            text = text.replace("\n", " ")
            text = text.replace('"', "'")
            news_id = _remove_prefix(entry).split(".json")[0]
            rating = reviews[reviews["news_id"] == news_id]["rating"].iloc[0]
            rating = 0 if rating > 2 else 1
            row = {
                "topic": partition,
                "label": rating,
                "content": str(text),
                "length": len(text),
                "sentiment": "neutral",
            }
            return row
    else:
        return None


def _normalize_FakeNewsNet_new(path: str, topic: str, label: str, df: pd.DataFrame):
    entries = os.listdir(path)
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results += pool.starmap(
            _normalize_FakeNewsNet_entry,
            tqdm(
                [(path, topic, label, entry) for entry in entries], total=len(entries)
            ),
        )
    filtered_results = [result for result in results if result is not None]
    df = df._append(filtered_results, ignore_index=True)
    return df


def _normalize_FakeNewsNet_entry(path, topic, label, entry):
    entry_path = os.path.join(path, entry, "news content.json")
    if os.path.exists(entry_path):
        with open(entry_path, "r") as f:
            json_data = f.read()
            data = json.loads(json_data)
            text = data["text"]
            text = text.replace("\n", " ")
            text = text.replace('"', "'")
            row = {
                "topic": topic,
                "label": label,
                "content": str(text),
                "length": len(text),
                "sentiment": "neutral",
            }
            return row
    else:
        return None


def _sentiment_analysis(df):
    sentiments_list = []
    contents = [str(row.content) for x, row in df.iterrows()]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        sentiments_list = list(
            tqdm(pool.imap(_sentiment_analysis_entry, contents), total=len(contents))
        )
    df["sentiment"] = sentiments_list
    return df


def _sentiment_analysis_entry(content: str):
    stop_words = set(stopwords.words("english"))
    sentence_list = sent_tokenize(content)
    word_tokens = word_tokenize(content)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    sentence_list = filtered_sentence
    sentiments = {"compound": 0.0}
    for sentence in sentence_list:
        vs = analyzer.polarity_scores(str(sentence))
        sentiments["compound"] += vs["compound"]
    sentiments["compound"] = sentiments["compound"] / len(sentence_list)
    compound = sentiments["compound"]
    label = "neutral"
    if compound > 0.001:
        label = "positive"
    if compound < -0.001:
        label = "negative"
    return label


def _remove_prefix(value):
    return "_".join(value.split("_")[1:])


def _normalize_MOCHEG(path: str, df: pd.DataFrame, entries: pd.DataFrame):
    for index, row in tqdm(entries.iterrows(), total=len(entries)):
        modified_row = _normalize_MOCHEG_entry(row)
        df.loc[-1] = modified_row
        df.index = df.index + 1
        df = df.sort_index()
    return df


def _normalize_MOCHEG_entry(entry):
    text = entry["Claim"]
    text = text.replace("\n", " ")
    text = text.replace('"', "'")
    rating = entry["cleaned_truthfulness"]
    rating = 0 if rating != "refuted" else 1
    row = {
        "topic": "Undefined",
        "label": rating,
        "content": str(text),
        "length": len(text),
        "sentiment": "neutral",
    }
    return row
