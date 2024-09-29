from collections import Counter
import os
from matplotlib import pyplot as plt
import pandas as pd
import utils
from wordcloud import WordCloud

# Initialization of pandas columns
raw_data_columns = ["Slice", "Amount Real", "Amount Fake", "Amount Total", "Percentage"]


class AnalyzeData:
    input_data_option = {
        "dest": "input",
        "type": str,
        "nargs": 1,
        "metavar": "<INPUT DATA>",
        "help": "Input data file path to analyze",
    }

    result_dir_option = {
        "dest": "result_dir",
        "type": str,
        "nargs": 1,
        "metavar": "<RESULT DIR OPTION>",
        "help": "Result directory to save the analyzed data",
    }

    def __init__(self, logger):
        self.logger = logger

    def add_parser(self, sub_parsers):
        analyze_data_parser = sub_parsers.add_parser(
            "analyze_data",
            help="Analyze data based on domain, sentiment, and text-length",
        )
        analyze_data_parser.add_argument(**self.input_data_option)
        analyze_data_parser.add_argument(**self.result_dir_option)
        try:
            analyze_data_parser.set_defaults(
                func=lambda args: self.main(args.input[0], args.result_dir[0])
            )
        except ValueError as error:
            self.logger.error(f"Value error in analyze data command: {error}")

    def main(self, input_path, result_dir):
        if not os.path.exists(input_path):
            self.logger.error(f"Input data file path does not exist: {input_path}")
            return

        result_dir = os.path.join(
            result_dir, "analyze_data", input_path.split(".csv")[0].split("/")[-1]
        )

        visualize_data_path = os.path.join(result_dir, "visualize_data")
        # utils.create_file_with_directories(visualize_data_path, self.logger)
        raw_csv_data_path = os.path.join(result_dir, "raw_csv_data.csv")

        self.logger.info(f"Analyzing data from {raw_csv_data_path}")
        utils.create_file_with_directories(raw_csv_data_path, self.logger)

        self.logger.info(f"Analyzing data from {result_dir}")
        # Add your code here to analyze data
        df = pd.read_csv(input_path)
        # Create raw data csv file
        # Amount Real, Amount Fake, Amount Total
        self.analyze_raw_data(df, raw_csv_data_path)
        self.visualize_data(df, visualize_data_path)

        self.logger.info("Data analysis completed")

    def analyze_raw_data(
        self,
        df,
        raw_csv_data_path,
    ):
        result = pd.DataFrame([], columns=raw_data_columns)
        # domains
        health = utils.get_health_news(df)
        entertainment = utils.get_entertainment_news(df)
        politics = utils.get_politics_news(df)
        undefined = utils.get_undefined_news(df)

        result.loc[len(result.index)] = [
            "Domains",
            "-",
            "-",
            "-",
            "-",
        ]

        result.loc[len(result.index)] = [
            "Total",
            utils.get_real_amount(df),
            utils.get_fake_amount(df),
            utils.get_total_amount(df),
            "100",
        ]
        result.loc[len(result.index)] = [
            "Health",
            utils.get_real_amount(health),
            utils.get_fake_amount(health),
            utils.get_total_amount(health),
            len(health) / len(df) * 100,
        ]
        result.loc[len(result.index)] = [
            "Entertainment",
            utils.get_real_amount(entertainment),
            utils.get_fake_amount(entertainment),
            utils.get_total_amount(entertainment),
            len(entertainment) / len(df) * 100,
        ]
        result.loc[len(result.index)] = [
            "Politics",
            utils.get_real_amount(politics),
            utils.get_fake_amount(politics),
            utils.get_total_amount(politics),
            len(politics) / len(df) * 100,
        ]
        result.loc[len(result.index)] = [
            "Undefined",
            utils.get_real_amount(undefined),
            utils.get_fake_amount(undefined),
            utils.get_total_amount(undefined),
            len(undefined) / len(df) * 100,
        ]

        # Sentiments
        positive = utils.get_positive_sentiment(df)
        negative = utils.get_negative_sentiment(df)
        neutral = utils.get_neutral_sentiment(df)

        result.loc[len(result.index)] = [
            "Sentiments",
            "-",
            "-",
            "-",
            "-",
        ]

        result.loc[len(result.index)] = [
            "Positive",
            utils.get_real_amount(positive),
            utils.get_fake_amount(positive),
            utils.get_total_amount(positive),
            len(positive) / len(df) * 100,
        ]
        result.loc[len(result.index)] = [
            "Negative",
            utils.get_real_amount(negative),
            utils.get_fake_amount(negative),
            utils.get_total_amount(negative),
            len(negative) / len(df) * 100,
        ]
        result.loc[len(result.index)] = [
            "Neutral",
            utils.get_real_amount(neutral),
            utils.get_fake_amount(neutral),
            utils.get_total_amount(neutral),
            len(neutral) / len(df) * 100,
        ]
        result.to_csv(raw_csv_data_path, index=False, mode="w")

        # Reduce to two decimals and add percentage of Percentage column
        result["Percentage"] = result["Percentage"].apply(
            lambda x: str(round(float(x), 2)) + "\%" if x != "-" else x
        )
        result.to_latex(raw_csv_data_path.replace(".csv", ".txt"), index=False)

    def visualize_data(self, df, visualize_data_path):
        # Add your code here to visualize data
        health = utils.get_health_news(df)
        entertainment = utils.get_entertainment_news(df)
        politics = utils.get_politics_news(df)
        undefined = utils.get_undefined_news(df)

        positive = utils.get_positive_sentiment(df)
        negative = utils.get_negative_sentiment(df)
        neutral = utils.get_neutral_sentiment(df)

        amount_health = len(health)
        amount_entertainment = len(entertainment)
        amount_politics = len(politics)
        amount_undefined = len(undefined)

        amount_positive = len(positive)
        amount_negative = len(negative)
        amount_neutral = len(neutral)

        # Create a pie charts

        # Domain distribution
        fig, ax = plt.subplots()
        ax.pie(
            [amount_health, amount_entertainment, amount_politics, amount_undefined],
            labels=["Health", "Entertainment", "Politics", "Undefined"],
            autopct="%1.1f%%",
        )
        ax.axis("equal")
        plt.title("News distribution by domain")
        news_distribution_by_domain_path = os.path.join(
            visualize_data_path, "news_distribution_by_domain.png"
        )
        utils.create_file_with_directories(
            news_distribution_by_domain_path, self.logger
        )
        plt.savefig(news_distribution_by_domain_path)

        # Sentiment distribution
        fig, ax = plt.subplots()
        ax.pie(
            [amount_positive, amount_negative, amount_neutral],
            labels=["Positive", "Negative", "Neutral"],
            autopct="%1.1f%%",
        )
        ax.axis("equal")
        plt.title("News distribution by sentiment")
        news_distribution_by_sentiment_path = os.path.join(
            visualize_data_path, "news_distribution_by_sentiment.png"
        )
        utils.create_file_with_directories(
            news_distribution_by_sentiment_path, self.logger
        )
        plt.savefig(news_distribution_by_sentiment_path)

        # Text Characterisstics:
        # Stacked bar graph where every bar is a text length bin containing the amount of real and fake news

        self.visualize_bins_per_col(df, "length", visualize_data_path)
        # self.visualize_bins_per_col(df, "smog_index", visualize_data_path)
        # self.visualize_bins_per_col(df, "flesch_reading_ease", visualize_data_path)
        # self.visualize_bins_per_col(
        #     df, "flesch_kincaid_grade_level", visualize_data_path
        # )
        # self.visualize_bins_per_col(df, "coleman_liau_index", visualize_data_path)
        # self.visualize_bins_per_col(df, "gunning_fog_index", visualize_data_path)
        # self.visualize_bins_per_col(df, "ari_index", visualize_data_path)
        # self.visualize_bins_per_col(df, "lix_index", visualize_data_path)
        # self.visualize_bins_per_col(df, "dale_chall_score", visualize_data_path)
        # self.visualize_bins_per_col(
        #     df, "dale_chall_known_fraction", visualize_data_path
        # )

        # Word clouds for fake and real data

        real_news = utils.get_real_news(df)
        fake_news = utils.get_fake_news(df)
        self.wordcloud_veracity(real_news, fake_news, visualize_data_path, "All")

        health_real = utils.get_real_news(health)
        health_fake = utils.get_fake_news(health)
        self.wordcloud_veracity(health_real, health_fake, visualize_data_path, "Health")

        entertainment_real = utils.get_real_news(entertainment)
        entertainment_fake = utils.get_fake_news(entertainment)
        self.wordcloud_veracity(
            entertainment_real, entertainment_fake, visualize_data_path, "Entertainment"
        )

        politics_real = utils.get_real_news(politics)
        politics_fake = utils.get_fake_news(politics)
        self.wordcloud_veracity(
            politics_real, politics_fake, visualize_data_path, "Politics"
        )

    def visualize_bins_per_col(self, df, col, path):
        df = utils.col_bins(df, col)
        bin_name = f"{col}_bin"
        bins = df[bin_name].unique().sort_values()
        bin_ranges = {}
        old = 0
        for bin in bins:
            if bin == bin:
                col_val = utils.get_col_quantile(df, bin, col).round(0)
                bin_ranges[bin] = f"{old}-{col_val}"
                old = col_val
        count_df = df.groupby([bin_name, "label"]).size().unstack(fill_value=0)
        percentage_df = count_df.div(count_df.sum(axis=1), axis=0) * 100
        print(percentage_df)
        fig, ax = plt.subplots(figsize=(10, 6))
        percentage_df.plot(
            kind="bar", stacked=True, color=["#1f77b4", "#ff7f0e"], ax=ax
        )
        plt.title(f"Percentage of Labels per {col}")
        plt.xlabel(f"{col} bin")
        plt.ylabel("Percentage")
        plt.legend(
            ["Real", "Fake"], title="Label", loc="upper left", bbox_to_anchor=(1, 1)
        )
        plt.tight_layout()
        plt.title(f"News distribution by {col}")
        news_distribution_by_col_path = os.path.join(
            path, f"news_distribution_by_{col}.png"
        )
        utils.create_file_with_directories(news_distribution_by_col_path, self.logger)
        plt.savefig(news_distribution_by_col_path)

    def wordcloud_veracity(self, df_real, df_fake, path, label):
        text_real = " ".join(df_real["content"])
        text_fake = " ".join(df_fake["content"])
        # text = " ".join(df["content"])
        # self.logger.info(
        #     f"Creating wordcloud for {label} news, Found {len(text)} words"
        # )
        wordcloud_real = WordCloud(width=800, height=400).generate(text_real)
        wordcloud_fake = WordCloud(width=800, height=400).generate(text_fake)

        fake_words = Counter(wordcloud_fake.words_)
        real_words = Counter(wordcloud_real.words_)

        unique_fake_words = {
            word: freq for word, freq in fake_words.items() if word not in real_words
        }
        unique_real_words = {
            word: freq for word, freq in real_words.items() if word not in fake_words
        }

        unique_fake_wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(unique_fake_words)
        unique_real_wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(unique_real_words)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_real, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Wordcloud of Real news")
        word_cloud_path = os.path.join(path, f"wordcloud_{label}_Real.png")
        utils.create_file_with_directories(word_cloud_path, self.logger)
        plt.savefig(word_cloud_path)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fake, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Wordcloud of Fake news")
        word_cloud_path = os.path.join(path, f"wordcloud_{label}_Fake.png")
        utils.create_file_with_directories(word_cloud_path, self.logger)
        plt.savefig(word_cloud_path)

        plt.figure(figsize=(10, 5))
        plt.imshow(unique_real_wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Wordcloud of Unique Real news")
        word_cloud_path = os.path.join(path, f"wordcloud_{label}_Real_unique.png")
        utils.create_file_with_directories(word_cloud_path, self.logger)
        plt.savefig(word_cloud_path)

        plt.figure(figsize=(10, 5))
        plt.imshow(unique_fake_wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Wordcloud of Unique Fake news")
        word_cloud_path = os.path.join(path, f"wordcloud_{label}_Fake_unique.png")
        utils.create_file_with_directories(word_cloud_path, self.logger)
        plt.savefig(word_cloud_path)
