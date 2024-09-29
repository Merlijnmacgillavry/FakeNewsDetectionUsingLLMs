import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utils
import os


def split_length_bin(df):
    # Define the percentiles you want to use for binning
    percentiles = [i for i in range(0, 76, 5)]

    # Calculate the quantile values for these percentiles
    length_percentiles = (
        df["length"].quantile([p / 100.0 for p in percentiles]).unique()
    )

    # Create bins based on these quantiles
    df["length_bin"] = pd.cut(
        df["length"],
        bins=length_percentiles,
        include_lowest=True,  # Include the lowest bin edge
        labels=percentiles[:-1],  # Labels should match the number of bins - 1
    )

    return df


def plot_column_distribution(train_df, val_df, test_df, column_name):
    def calculate_percentage(df, column):
        return df[column].value_counts(normalize=True) * 100

    # Calculate percentages
    train_dist = calculate_percentage(train_df, column_name)
    val_dist = calculate_percentage(val_df, column_name)
    test_dist = calculate_percentage(test_df, column_name)

    # Create a DataFrame for plotting
    dist_df = pd.DataFrame(
        {"Train": train_dist, "Validation": val_dist, "Test": test_dist}
    ).fillna(
        0
    )  # Fill NaNs with 0 for missing categories

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    dist_df["Train"].plot(kind="bar", ax=axes[0], color="skyblue", title="Training Set")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Percentage")

    dist_df["Validation"].plot(
        kind="bar", ax=axes[1], color="lightgreen", title="Validation Set"
    )
    axes[1].set_xlabel("Value")

    dist_df["Test"].plot(kind="bar", ax=axes[2], color="lightcoral", title="Test Set")
    axes[2].set_xlabel("Value")

    fig.suptitle(f"Distribution of {column_name} across different splits", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the suptitle
    plt.show()


class Splitter:
    input_path_option = {
        "dest": "input",
        "type": str,
        "nargs": 1,
        "metavar": "<INPUT PATH>",
        "help": "The path to the combined dataset that needs to be split",
    }

    output_path_option = {
        "dest": "output",
        "type": str,
        "nargs": 1,
        "metavar": "<OUTPUT PATH>",
        "help": "The path to the folder containing the split datasets",
    }

    def __init__(self, logger) -> None:
        self.logger = logger

    def add_parser(self, sub_parsers):
        splitter_parse = sub_parsers.add_parser(
            "split", help="split the dataset into train, test, and validation sets"
        )
        splitter_parse.add_argument(**self.input_path_option)
        splitter_parse.add_argument(**self.output_path_option)
        splitter_parse.set_defaults(
            func=lambda args: self.split(args.input[0], args.output[0])
        )

    def split(self, input_path, output_path):
        self.logger.info(f"Splitting dataset at: {input_path}")
        utils.create_file_with_directories(output_path, self.logger)
        dataset = pd.read_csv(input_path)
        split_length_bin(dataset)

        # We combined HealthRelease and HealthStory into a single category
        dataset["combined_topics"] = dataset["topic"].replace(
            {"HealthRelease": "health", "HealthStory": "health"}
        )
        dataset["stratify_group"] = (
            dataset["label"].astype(str)
            + "_"
            + dataset["combined_topics"].astype(str)
            + "_"
            + dataset["sentiment"].astype(str)
        )
        stratify_counts = dataset["stratify_group"].value_counts()

        # Identify classes with only 1 member
        low_count_classes = stratify_counts[stratify_counts < 2]
        self.logger.info(f"Classes with only 1 member: {low_count_classes}")

        train_data, test_data = train_test_split(
            dataset,
            test_size=0.2,
            stratify=dataset["stratify_group"],
            random_state=42,
        )
        train_data, val_data = train_test_split(
            train_data,
            test_size=0.2,
            stratify=train_data["stratify_group"],
            random_state=42,
        )

        files = ["train", "test", "val"]
        for f in files:
            output_file = f"{os.path.join(output_path, f)}.csv"
            data = (
                train_data if f == "train" else test_data if f == "test" else val_data
            )
            data = data.drop(
                columns=["length_bin", "combined_topics", "stratify_group"]
            )
            data.to_csv(output_file)


# if __name__ == "__main__":
#     dataset = pd.read_csv("./data/out/preprocess_all.csv")
#     split_length_bin(dataset)
#     dataset["combined_topics"] = dataset["topic"].replace(
#         {"HealthRelease": "health", "HealthStory": "health"}
#     )
#     dataset["stratify_group"] = (
#         dataset["label"].astype(str)
#         + "_"
#         + dataset["combined_topics"].astype(str)
#         + "_"
#         + dataset["sentiment"].astype(str)
#     )
#     stratify_counts = dataset["stratify_group"].value_counts()

#     # Identify classes with only 1 member
#     low_count_classes = stratify_counts[stratify_counts < 2]
#     print(f"Classes with only 1 member: {low_count_classes}")

#     train_data, test_data = train_test_split(
#         dataset, test_size=0.2, stratify=dataset["stratify_group"], random_state=42
#     )
#     train_data, val_data = train_test_split(
#         train_data,
#         test_size=0.2,
#         stratify=train_data["stratify_group"],
#         random_state=42,
#     )
#     sentiment_counts_test = test_data["label"].value_counts()
#     sentiment_counts_val = val_data["label"].value_counts()

#     # Plot distributions
#     plot_column_distribution(train_data, val_data, test_data, "topic")
#     plot_column_distribution(train_data, val_data, test_data, "label")
#     plot_column_distribution(train_data, val_data, test_data, "sentiment")
#     dir = "./data/finetuning"
#     files = ["train", "test", "val"]
#     for f in files:
#         output_file = f"{os.path.join(dir, f)}.csv"
#         data = train_data if f == "train" else test_data if f == "test" else val_data
#         data = data.drop(columns=["length_bin", "combined_topics", "stratify_group"])
#         data.to_csv(output_file)
