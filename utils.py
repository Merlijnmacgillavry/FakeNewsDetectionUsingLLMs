import os
import string
import pandas as pd
from nltk.corpus import stopwords

model_options = ["gemma-2b", "gemma-2-9b", "llama-3.1-8b-it", "mistral-0.2-7b-it"]
method_options = ["discrete", "cot", "binary", "finetuned", "percentage"]


def create_file_with_directories(file_path, logger, content=""):
    """
    Creates a file at the specified path, including any missing directories.

    Args:
        file_path (str): The path to the file to create.
        content (str, optional): The content to write to the file. Defaults to "".
    """

    # Get directory path from the file path
    directory_path = os.path.dirname(file_path)

    # Create directories if they don't exist
    try:
        os.makedirs(directory_path)
    except OSError as error:
        logger.error(f"Error creating directories: {error}")
        pass

    # Open the file in write mode (creates it if it doesn't exist)
    try:
        with open(file_path, "w") as f:
            f.write(content)
        logger.info(f"File created at: {file_path}")
    except OSError as error:
        logger.error(f"Error creating file: {error}")


def get_repo(model):
    return f"Lord-Papillon/{model}-finetuned-FND"


def get_real_news(df):
    return df[df["label"] == 0]


def get_fake_news(df):
    return df[df["label"] == 1]


def get_real_amount(df):
    return len(get_real_news(df))


def get_fake_amount(df):
    return len(get_fake_news(df))


def get_total_amount(df):
    return len(df)


def get_entertainment_news(df):
    return df[df["topic"] == "entertainment"]


def get_not_entertainment_news(df):
    return df[df["topic"] != "entertainment"]


def get_politics_news(df):
    return df[df["topic"] == "politics"]


def get_not_politics_news(df):
    return df[df["topic"] != "politics"]


def get_health_news(df):
    return pd.concat(
        [df[df["topic"] == "HealthStory"], df[df["topic"] == "HealthRelease"]]
    )


def get_not_health_news(df):
    return df[(df["topic"] != "HealthStory") & (df["topic"] != "HealthRelease")]


def get_undefined_news(df):
    return df[df["topic"] == "Undefined"]


def get_defined_news(df):
    return df[df["topic"] != "Undefined"]


def get_positive_sentiment(df):
    return df[df["sentiment"] == "positive"]


def get_negative_sentiment(df):
    return df[df["sentiment"] == "negative"]


def get_neutral_sentiment(df):
    return df[df["sentiment"] == "neutral"]


def text_length_bins(df):
    percentiles = [i for i in range(0, 76, 5)]
    length_percentiles = df["length"].quantile([p / 100.0 for p in percentiles])
    df["length_bin"] = pd.cut(
        df["length"],
        bins=length_percentiles,
        include_lowest=True,
        labels=percentiles[:-1],
    )
    return df


def col_bins(df, col):
    if col == "gunning_fog_index":
        return col_bins_gfi(df, col)
    if col == "dale_chall_score":
        return col_bins_dale_chall(df, col)
    if col == "lix_index":
        return col_bins_lix_index(df, col)
    percentiles = [i for i in range(0, 101, 5)]
    length_percentiles = df[col].quantile([p / 100.0 for p in percentiles])
    unique_percentiles = length_percentiles.unique()
    df[f"{col}_bin"] = pd.cut(
        df[col],
        bins=unique_percentiles,
        include_lowest=True,
        labels=percentiles[: len(unique_percentiles) - 1],
    )
    return df


def col_bins_dale_chall(df, col):
    # Define the bin edges between 4 and 10 (inclusive)
    bin_edges = [4, 5, 6, 7, 8, 9, 10]

    # Create labels corresponding to the Dale-Chall range
    labels = [f"{i}-{i+1}" for i in range(4, 10)]

    # Bin the column based on the specified bin edges
    df[f"{col}_bin"] = pd.cut(
        df[col], bins=bin_edges, include_lowest=True, labels=labels
    )

    return df


def col_bins_lix_index(df, col):
    bin_edges = [20, 25, 30, 35, 40, 45, 50, 55, 60]
    labels = [f"{i}-{i+5}" for i in range(20, 60, 5)]
    df[f"{col}_bin"] = pd.cut(
        df[col], bins=bin_edges, include_lowest=True, labels=labels
    )
    return df


def col_bins_gfi(df, col):
    # Define the bin edges between 6 and 17 (inclusive)
    bin_edges = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    # Create labels corresponding to the Gunning Fog range
    labels = [f"{i}-{i+1}" for i in range(6, 17)]

    # Bin the column based on the specified bin edges
    df[f"{col}_bin"] = pd.cut(
        df[col], bins=bin_edges, include_lowest=True, labels=labels
    )

    return df


def get_text_length_quantile(df, quantile):
    return df["length"].quantile(quantile / 100)


def get_col_quantile(df, quantile, col):
    return df[col].quantile(quantile / 100)


def get_name_by_file(file):
    return file.split(".csv")[0].split("/")[-1].replace("predictions_", "")


def get_model_size_by_file(file):
    name = get_name_by_file(file)
    return name.replace("-it", "").split("-")[-1].split("_")[0]


def process_content_for_wordcloud(content):
    stop_words = set(stopwords.words("english"))
    custom_stopwords = set(["winter", "sweater", "christmas", "ugly", "sweaterugly"])
    stop_words.update(custom_stopwords)
    text = content.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def get_false_positives(df):
    return df[(df["label"] == 0) & (df["prediction"] == 1)]


def get_false_negatives(df):
    return df[(df["label"] == 1) & (df["prediction"] == 0)]


def fp_binary(df):
    return len(df[(df["prediction"] == 1) & (df["label"] == 0)])


def fn_binary(df):
    return len(df[(df["prediction"] == 0) & (df["label"] == 1)])


def tp_binary(df):
    return len(df[(df["prediction"] == 1) & (df["label"] == 1)])


def tn_binary(df):
    return len(df[(df["prediction"] == 0) & (df["label"] == 0)])


def acc_binary(df):
    return len(df[df["prediction"] == df["label"]]) / len(df)


def change_discrete_above_mostly_true_to_binary(df):
    df = df.apply(change_discrete_above_mostly_true_to_binary_row, axis=1)
    return df


def change_percentage_above_50_to_binary(df):
    df = df.apply(change_percentage_above_50_to_binary_row, axis=1)
    return df


def change_percentage_above_75_to_binary(df):
    df = df.apply(change_percentage_above_75_to_binary_row, axis=1)
    return df


def change_percentage_above_50_to_binary_row(row):
    return change_percentage_above_cutoff_to_binary_row(row, 50)


def change_percentage_above_75_to_binary_row(row):
    return change_percentage_above_cutoff_to_binary_row(row, 75)


def change_percentage_above_cutoff_to_binary_row(row, cutoff):
    prediction = str(row["prediction"])
    if not prediction.isdigit():
        row["prediction"] = 0
        return row
    prediction = float(prediction)
    if prediction > cutoff:
        row["prediction"] = 1
    else:
        row["prediction"] = 0
    return row


def change_discrete_includes_true_to_binary(df):
    df = df.apply(change_discrete_includes_true_to_binary_row, axis=1)
    return df


def change_discrete_includes_true_to_binary_row(row):
    prediction = row["prediction"]
    if "true" in str(prediction).lower():
        row["prediction"] = 0
    else:
        row["prediction"] = 1
    return row


def change_discrete_above_mostly_true_to_binary_row(row):
    prediction = row["prediction"]
    if "true" in str(prediction).lower() and not "half true" in str(prediction).lower():
        row["prediction"] = 0
    else:
        row["prediction"] = 1
    return row


def create_dir(path, logger):
    try:
        os.makedirs(path)
    except OSError as error:
        logger.error(f"Error creating directories: {error}")
        pass


colors = [
    "#0C2340",
    "#00B8C8",  # cyan
    "#6F1D77",
    "#FFB81C",
    "#E03C31",  # orange
    "#6CC24A",
    "#EF60A3",
    "#009B77",
    "#A50034",
    "#F89D2A",
    "#D0006F",
    "#FFD662",
    "#F092B0",
    "#00A6ED",
    "#FF7F32",
    "#FF3C38",
    "#FFD662",
    "#FFB81C",
]
