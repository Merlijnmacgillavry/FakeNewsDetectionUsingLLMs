from collections import Counter
import os
from matplotlib import pyplot as plt
import pandas as pd
import utils
from wordcloud import WordCloud
import numpy as np
from analyze_predictions_module import performance, textual, domain, psychological

raw_results_columns = [
    "model",
    "total_predictions",
    "usable_predictions",
    "accuracy_all",
    "accuracy_usable",
    "false_positive_usable",
    "false_negative_usable",
    "true_positive_usable",
    "true_negative_usable",
    "accuracy_entertainment",
    "false_positive_entertainment",
    "false_negative_entertainment",
    "true_positive_entertainment",
    "true_negative_entertainment",
    "accuracy_health",
    "false_positive_health",
    "false_negative_health",
    "true_positive_health",
    "true_negative_health",
    "accuracy_politics",
    "false_positive_politics",
    "false_negative_politics",
    "true_positive_politics",
    "true_negative_politics",
    "accuracy_undefined",
    "false_positive_undefined",
    "false_negative_undefined",
    "true_positive_undefined",
    "true_negative_undefined",
    "accuracy_usable_entertainment",
    "false_positive_usable_entertainment",
    "false_negative_usable_entertainment",
    "true_positive_usable_entertainment",
    "true_negative_usable_entertainment",
    "accuracy_usable_health",
    "false_positive_usable_health",
    "false_negative_usable_health",
    "true_positive_usable_health",
    "true_negative_usable_health",
    "accuracy_usable_politics",
    "false_positive_usable_politics",
    "false_negative_usable_politics",
    "true_positive_usable_politics",
    "true_negative_usable_politics",
    "accuracy_usable_undefined",
    "false_positive_usable_undefined",
    "false_negative_usable_undefined",
    "true_positive_usable_undefined",
    "true_negative_usable_undefined",
    "accuracy_positive_sentiment",
    "false_positive_positive_sentiment",
    "false_negative_positive_sentiment",
    "true_positive_positive_sentiment",
    "true_negative_positive_sentiment",
    "accuracy_negative_sentiment",
    "false_positive_negative_sentiment",
    "false_negative_negative_sentiment",
    "true_positive_negative_sentiment",
    "true_negative_negative_sentiment",
    "accuracy_neutral_sentiment",
    "false_positive_neutral_sentiment",
    "false_negative_neutral_sentiment",
    "true_positive_neutral_sentiment",
    "true_negative_neutral_sentiment",
    "accuracy_usable_positive_sentiment",
    "false_positive_usable_positive_sentiment",
    "false_negative_usable_positive_sentiment",
    "true_positive_usable_positive_sentiment",
    "true_negative_usable_positive_sentiment",
    "accuracy_usable_negative_sentiment",
    "false_positive_usable_negative_sentiment",
    "false_negative_usable_negative_sentiment",
    "true_positive_usable_negative_sentiment",
    "true_negative_usable_negative_sentiment",
    "accuracy_usable_neutral_sentiment",
    "false_positive_usable_neutral_sentiment",
    "false_negative_usable_neutral_sentiment",
    "true_positive_usable_neutral_sentiment",
    "true_negative_usable_neutral_sentiment",
]

performance_columns = [
    "Model",
    "Method",
    "Model Size",
    "Dataset Size",
    "Usable Size",
    "Usable Predictions",
    "Accuracy",
    "Accuracy Usable",
    "False Positive",
    "False Negative",
    "True Positive",
    "True Negative",
]

domain_columns = [
    "Model",
    "Method",
    "Domain",
    "Model Size",
    "Usable Predictions",
    "Accuracy",
    "Accuracy Usable",
    "False Positive",
    "False Negative",
    "True Positive",
    "True Negative",
]


class AnalyzePredictions:
    input_dir_option = {
        "dest": "input",
        "type": str,
        "nargs": 1,
        "metavar": "<INPUT DIR>",
        "help": "Input directory containing the prediction files",
    }
    result_dir_option = {
        "dest": "result",
        "type": str,
        "nargs": 1,
        "metavar": "<RESULT DIR>",
        "help": "Output directory to store the analysis results",
    }

    def __init__(self, logger):
        self.logger = logger

    def add_parser(self, subparsers):
        parser = subparsers.add_parser(
            "analyze_predictions", help="Analyze the prediction files"
        )
        parser.add_argument(**self.input_dir_option)
        parser.add_argument(**self.result_dir_option)
        try:
            parser.set_defaults(
                func=lambda args: self.main(args.input[0], args.result[0])
            )
        except ValueError as error:
            self.logger.error(f"Value error in analyze data command: {error}")

    def main(self, input_dir, result_dir):
        self.logger.info(f"Analyzing prediction files in {input_dir}")
        result_dir = os.path.join(
            result_dir,
            "analyze_predictions",
        )
        self.logger.info(f"Storing the results in {result_dir}")
        # Add your code here
        # self.analyze_raw_results(input_dir, result_dir)
        # self.visualize_results(input_dir, result_dir)
        self.performance_metrics(input_dir, result_dir)
        self.textual_metrics(input_dir, result_dir)
        self.domain_metrics(input_dir, result_dir)
        self.psychological_metrics(input_dir, result_dir)
        self.logger.info("Analysis completed")

    def textual_metrics(self, input_dir, result_dir):
        self.logger.info("Calculating textual metrics")
        textual_dir = os.path.join(result_dir, "textual")
        # textual.text_charcteristics_metrics_model_avg(
        #     input_dir, textual_dir, "length", self.logger
        # )
        textual.text_characteristics_metrics_model_avg(
            input_dir, textual_dir, "length", self.logger
        )
        textual.text_characteristics_metrics_model_avg(
            input_dir, textual_dir, "dale_chall_score", self.logger
        )
        # textual.text_characteristics_metrics_model(
        #     input_dir, textual_dir, "lix_index", self.logger
        # )
        # textual.text_length_metrics_method(
        #     input_dir, textual_dir, "length", self.logger
        # )
        textual.text_chracteristics_metrics_method(
            input_dir, textual_dir, "gunning_fog_index", self.logger
        )
        textual.text_chracteristics_metrics_method_avg(
            input_dir, textual_dir, "length", self.logger
        )
        textual.text_chracteristics_metrics_method_avg(
            input_dir, textual_dir, "dale_chall_score", self.logger
        )
        textual.text_chracteristics_metrics_method(
            input_dir, textual_dir, "lix_index", self.logger
        )
        self.logger.info("Textual metrics calculated")

    def visualize_textual_metrics(self, input_dir, textual_dir):
        prediction_files = os.listdir(input_dir)
        for file in prediction_files:
            file = os.path.join(input_dir, file)
            if "binary" in file or "cot" in file:
                self.visualize_bins(file, textual_dir)

    def domain_metrics(self, input_dir, result_dir):
        self.logger.info("Calculating domain metrics")

        domain_dir = os.path.join(result_dir, "domain")
        if os.path.exists(os.path.join(domain_dir, "domain_metrics_table.csv")):
            raw_result = pd.read_csv(
                os.path.join(domain_dir, "domain_metrics_table.csv")
            )
        else:
            raw_result = domain.domain_metrics_table(input_dir, domain_dir, self.logger)

        # # Bar graph for false positive and false negative for each domain and method
        domain.bar_graph_domain_metrics_fp_fn_methods(
            input_dir, domain_dir, raw_result, self.logger
        )
        # # Bar graph for false positive and false negative for each domain and model
        domain.bar_graph_domain_metrics_fp_fn_models(
            input_dir, domain_dir, raw_result, self.logger
        )

        # # Box plot per metric for each domain and metric
        domain.box_plot_domain_metrics(input_dir, domain_dir, raw_result, self.logger)

        # # Wordclouds
        # self.word_cloud(input_dir, domain_dir, raw_result)

        self.logger.info("Domain metrics calculated")

    def psychological_metrics(self, input_dir, result_dir):
        self.logger.info("Calculating psychological metrics")
        psychological_dir = os.path.join(result_dir, "psychological")
        if os.path.exists(
            os.path.join(psychological_dir, "sentiment_metrics_table.csv")
        ):
            raw_result = pd.read_csv(
                os.path.join(psychological_dir, "sentiment_metrics_table.csv")
            )
        else:
            raw_result = psychological.psychological_metrics_table(
                input_dir, psychological_dir, self.logger
            )

        # # Bar graph per method for each psychological metric
        # psychological.bar_graph_psychological_metrics_accuracy(
        #     input_dir, psychological_dir, raw_result, self.logger
        # )
        # # # Bar graph for false positive and false negative for each psychological and method
        # psychological.bar_graph_psychological_metrics_fp_fn_methods_sentimental(
        #     input_dir, psychological_dir, raw_result, self.logger
        # )
        # psychological.bar_graph_psychological_metrics_fp_fn_methods_sentimental_models(
        #     input_dir, psychological_dir, raw_result, self.logger
        # )
        # # # Bar graph for false positive and false negative for each psychological and model
        # psychological.bar_graph_psychological_metrics_fp_fn_models(
        #     input_dir, psychological_dir, raw_result, self.logger
        # )
        # # Box plot per metric for each psychological and metric
        # psychological.box_plot_psychological_metrics_models(
        #     input_dir, psychological_dir, raw_result, self.logger
        # )
        # psychological.box_plot_psychological_sentimental_metrics(
        #     input_dir, psychological_dir, raw_result, self.logger
        # )
        psychological.box_plot_psychological_metrics_models(
            input_dir, psychological_dir, raw_result, self.logger
        )
        psychological.box_plot_psychological_sentimental_metrics_models(
            input_dir, psychological_dir, raw_result, self.logger
        )
        # self.bar_graph_psychological_metrics(input_dir, psychological_dir, raw_result)
        self.logger.info("Psychological metrics calculated")

    def word_cloud(self, input_dir, result_dir, raw_result):
        self.logger.info("Creating word clouds")
        prediction_files = os.listdir(input_dir)
        false_positive_text_entertainment = ""
        false_positive_text_politics = ""
        false_positive_text_health = ""
        false_positive_text_undefined = ""
        false_positive_text_defined = ""

        false_negative_text_entertainment = ""
        false_negative_text_politics = ""
        false_negative_text_health = ""
        false_negative_text_undefined = ""
        false_negative_text_defined = ""

        not_entertainment_text = ""
        not_politics_text = ""
        not_health_text = ""
        not_undefined_text = ""
        not_defined_text = ""

        for file in prediction_files:
            file = os.path.join(input_dir, file)
            df = pd.read_csv(file)
            if "discrete" in file:
                continue
            fp = utils.get_false_positives(df)
            fn = utils.get_false_negatives(df)

            # not_entertainment = utils.get_not_entertainment_news(df)
            # not_entertainment["content"] = not_entertainment["content"].apply(
            #     utils.process_content_for_wordcloud
            # )
            # not_entertainment_text += " ".join(not_entertainment["content"].to_list())

            # not_politics = utils.get_not_politics_news(df)
            # not_politics["content"] = not_politics["content"].apply(
            #     utils.process_content_for_wordcloud
            # )
            # not_politics_text += " ".join(not_politics["content"].to_list())

            # not_health = utils.get_not_health_news(df)
            # not_health["content"] = not_health["content"].apply(
            #     utils.process_content_for_wordcloud
            # )
            # not_health_text += " ".join(not_health["content"].to_list())

            # not_undefined = utils.get_defined_news(df)
            # not_undefined["content"] = not_undefined["content"].apply(
            #     utils.process_content_for_wordcloud
            # )
            # not_undefined_text += " ".join(not_undefined["content"].to_list())

            # not_defined = utils.get_undefined_news(df)
            # not_defined["content"] = not_defined["content"].apply(
            #     utils.process_content_for_wordcloud
            # )
            # not_defined_text += " ".join(not_defined["content"].to_list())

            fn_entertainment = utils.get_entertainment_news(fn)
            fn_entertainment["content"] = fn_entertainment["content"].apply(
                utils.process_content_for_wordcloud
            )
            false_negative_text_entertainment += " ".join(
                fn_entertainment["content"].to_list()
            )
            fn_politics = utils.get_politics_news(fn)
            fn_politics["content"] = fn_politics["content"].apply(
                utils.process_content_for_wordcloud
            )
            false_negative_text_politics += " ".join(fn_politics["content"].to_list())
            fn_health = utils.get_health_news(fn)
            fn_health["content"] = fn_health["content"].apply(
                utils.process_content_for_wordcloud
            )
            false_negative_text_health += " ".join(fn_health["content"].to_list())

            fn_undefined = utils.get_undefined_news(fn)
            fn_undefined["content"] = fn_undefined["content"].apply(
                utils.process_content_for_wordcloud
            )
            false_negative_text_undefined += " ".join(fn_undefined["content"].to_list())

            fn_defined = utils.get_defined_news(fn)
            fn_defined["content"] = fn_defined["content"].apply(
                utils.process_content_for_wordcloud
            )
            false_negative_text_defined += " ".join(fn_defined["content"].to_list())

            fp_entertainment = utils.get_entertainment_news(fp)
            fp_entertainment["content"] = fp_entertainment["content"].apply(
                utils.process_content_for_wordcloud
            )
            false_positive_text_entertainment += " ".join(
                fp_entertainment["content"].to_list()
            )

            fp_politics = utils.get_politics_news(fp)
            fp_politics["content"] = fp_politics["content"].apply(
                utils.process_content_for_wordcloud
            )
            false_positive_text_politics += " ".join(fp_politics["content"].to_list())

            fp_health = utils.get_health_news(fp)
            fp_health["content"] = fp_health["content"].apply(
                utils.process_content_for_wordcloud
            )
            false_positive_text_health += " ".join(fp_health["content"].to_list())

            fp_undefined = utils.get_undefined_news(fp)
            fp_undefined["content"] = fp_undefined["content"].apply(
                utils.process_content_for_wordcloud
            )
            false_positive_text_undefined += " ".join(fp_undefined["content"].to_list())

            fp_defined = utils.get_defined_news(fp)
            fp_defined["content"] = fp_defined["content"].apply(
                utils.process_content_for_wordcloud
            )
            false_positive_text_defined += " ".join(fp_defined["content"].to_list())

        # wordcloud_not_entertainment = WordCloud(width=800, height=400).generate(
        #     not_entertainment_text
        # )
        # not_entertainment_words = Counter(wordcloud_not_entertainment.words_)
        # wordcloud_not_politics = WordCloud(width=800, height=400).generate(
        #     not_politics_text
        # )
        # not_politics_words = Counter(wordcloud_not_politics.words_)
        # wordcloud_not_health = WordCloud(width=800, height=400).generate(
        #     not_health_text
        # )
        # not_health_words = Counter(wordcloud_not_health.words_)
        # wordcloud_not_undefined = WordCloud(width=800, height=400).generate(
        #     not_undefined_text
        # )
        # not_undefined_words = Counter(wordcloud_not_undefined.words_)
        # wordcloud_not_defined = WordCloud(width=800, height=400).generate(
        #     not_defined_text
        # )
        # not_defined_words = Counter(wordcloud_not_defined.words_)

        wordcloud_fn_entertainment = WordCloud(width=800, height=400).generate(
            false_negative_text_entertainment
        )
        # fn_entertainment_words = Counter(wordcloud_fn_entertainment.words_)
        # unique_fn_entertainment = {
        #     word: freq
        #     for word, freq in fn_entertainment_words.items()
        #     if word not in not_entertainment_words
        # }
        # wordcloud_fn_entertainment = WordCloud(
        #     width=800, height=400
        # ).generate_from_frequencies(unique_fn_entertainment)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fn_entertainment, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Topic: Entertainment")
        plt.savefig(os.path.join(result_dir, f"wordcloud_fn_entertainment.png"))
        plt.close()

        wordcloud_fn_politics = WordCloud(width=800, height=400).generate(
            false_negative_text_politics
        )
        # fn_politics_words = Counter(wordcloud_fn_politics.words_)
        # unique_fn_politics = {
        #     word: freq
        #     for word, freq in fn_politics_words.items()
        #     if word not in not_politics_words
        # }
        # wordcloud_fn_politics = WordCloud(
        #     width=800, height=400
        # ).generate_from_frequencies(unique_fn_politics)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fn_politics, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Topic: Politics")
        plt.savefig(os.path.join(result_dir, f"wordcloud_fn_politics.png"))
        plt.close()

        wordcloud_fn_health = WordCloud(width=800, height=400).generate(
            false_negative_text_health
        )
        # fn_health_words = Counter(wordcloud_fn_health.words_)
        # unique_fn_health = {
        #     word: freq
        #     for word, freq in fn_health_words.items()
        #     if word not in not_health_words
        # }
        # wordcloud_fn_health = WordCloud(
        #     width=800, height=400
        # ).generate_from_frequencies(unique_fn_health)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fn_health, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Topic: Health")
        plt.savefig(os.path.join(result_dir, f"wordcloud_fn_health.png"))
        plt.close()

        wordcloud_fn_undefined = WordCloud(width=800, height=400).generate(
            false_negative_text_undefined
        )
        # fn_undefined_words = Counter(wordcloud_fn_undefined.words_)
        # unique_fn_undefined = {
        #     word: freq
        #     for word, freq in fn_undefined_words.items()
        #     if word not in not_undefined_words
        # }
        # wordcloud_fn_undefined = WordCloud(
        #     width=800, height=400
        # ).generate_from_frequencies(unique_fn_undefined)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fn_undefined, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Topic: Undefined")
        plt.savefig(os.path.join(result_dir, f"wordcloud_fn_undefined.png"))
        plt.close()

        wordcloud_fn_defined = WordCloud(width=800, height=400).generate(
            false_negative_text_defined
        )
        # fn_defined_words = Counter(wordcloud_fn_defined.words_)
        # unique_fn_defined = {
        #     word: freq
        #     for word, freq in fn_defined_words.items()
        #     if word not in not_defined_words
        # }
        # wordcloud_fn_defined = WordCloud(
        #     width=800, height=400
        # ).generate_from_frequencies(unique_fn_defined)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fn_defined, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Topic: Defined")
        plt.savefig(os.path.join(result_dir, f"wordcloud_fn_defined.png"))
        plt.close()

        wordcloud_fp_entertainment = WordCloud(width=800, height=400).generate(
            false_positive_text_entertainment
        )
        # fp_entertainment_words = Counter(wordcloud_fp_entertainment.words_)
        # unique_fp_entertainment = {
        #     word: freq
        #     for word, freq in fp_entertainment_words.items()
        #     if word not in not_entertainment_words
        # }
        # wordcloud_fp_entertainment = WordCloud(
        #     width=800, height=400
        # ).generate_from_frequencies(unique_fp_entertainment)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fp_entertainment, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Topic: Entertainment")
        plt.savefig(os.path.join(result_dir, f"wordcloud_fp_entertainment.png"))
        plt.close()

        wordcloud_fp_politics = WordCloud(width=800, height=400).generate(
            false_positive_text_politics
        )
        # fp_politics_words = Counter(wordcloud_fp_politics.words_)
        # unique_fp_politics = {
        #     word: freq
        #     for word, freq in fp_politics_words.items()
        #     if word not in not_politics_words
        # }
        # wordcloud_fp_politics = WordCloud(
        #     width=800, height=400
        # ).generate_from_frequencies(unique_fp_politics)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fp_politics, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Topic: Politics")
        plt.savefig(os.path.join(result_dir, f"wordcloud_fp_politics.png"))
        plt.close()

        wordcloud_fp_health = WordCloud(width=800, height=400).generate(
            false_positive_text_health
        )
        # fp_health_words = Counter(wordcloud_fp_health.words_)
        # unique_fp_health = {
        #     word: freq
        #     for word, freq in fp_health_words.items()
        #     if word not in not_health_words
        # }
        # wordcloud_fp_health = WordCloud(
        #     width=800, height=400
        # ).generate_from_frequencies(unique_fp_health)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fp_health, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Topic: Health")
        plt.savefig(os.path.join(result_dir, f"wordcloud_fp_health.png"))
        plt.close()

        wordcloud_fp_undefined = WordCloud(width=800, height=400).generate(
            false_positive_text_undefined
        )
        # fp_undefined_words = Counter(wordcloud_fp_undefined.words_)
        # unique_fp_undefined = {
        #     word: freq
        #     for word, freq in fp_undefined_words.items()
        #     if word not in not_undefined_words
        # }
        # wordcloud_fp_undefined = WordCloud(
        #     width=800, height=400
        # ).generate_from_frequencies(unique_fp_undefined)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fp_undefined, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Topic: Undefined")
        plt.savefig(os.path.join(result_dir, f"wordcloud_fp_undefined.png"))
        plt.close()

        wordcloud_fp_defined = WordCloud(width=800, height=400).generate(
            false_positive_text_defined
        )
        # fp_defined_words = Counter(wordcloud_fp_defined.words_)
        # unique_fp_defined = {
        #     word: freq
        #     for word, freq in fp_defined_words.items()
        #     if word not in not_defined_words
        # }
        # wordcloud_fp_defined = WordCloud(
        #     width=800, height=400
        # ).generate_from_frequencies(unique_fp_defined)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fp_defined, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Topic: Defined")
        plt.savefig(os.path.join(result_dir, f"wordcloud_fp_defined.png"))
        plt.close()

    def performance_metrics(self, input_dir, result_dir):
        self.logger.info("Calculating Performance metrics")
        # Overall performance Table
        performance_dir = os.path.join(result_dir, "performance")
        if os.path.exists(os.path.join(performance_dir, "raw_performance_table.csv")):
            raw_result = pd.read_csv(
                os.path.join(performance_dir, "raw_performance_table.csv")
            )
        else:
            raw_result = performance.raw_performance_table(
                input_dir, performance_dir, self.logger
            )
        # Bar graphs for performance metrics
        dir = os.path.join(performance_dir, "bar_graphs")
        utils.create_dir(dir, self.logger)
        performance.plot_performance_data(
            raw_result, dir, self.logger, "Usable Predictions"
        )
        performance.plot_performance_data(
            raw_result, dir, self.logger, "Accuracy Usable"
        )
        performance.plot_performance_false_predictions(raw_result, dir, self.logger)
        dir = os.path.join(performance_dir, "box_plots")
        utils.create_dir(dir, self.logger)
        performance.box_plot_accuracy_models(input_dir, dir, raw_result, self.logger)
        performance.box_plot_accuracy_methods(input_dir, dir, raw_result, self.logger)

        # Linegraph based on model size
        # self.line_graph_model_size(input_dir, performance_dir, raw_result)
        # # Box plots for accuracy
        # # Stacked bargraphs for fp and fn
        # self.stacked_bar_graph_fp_fn_models(input_dir, performance_dir, raw_result)
        performance.stacked_bar_graph_fp_fn_methods(
            input_dir, performance_dir, raw_result, self.logger
        )

        self.logger.info("Performance metrics calculated")

    def box_plot_accuracy(self, input_dir, result_dir, raw_result):
        self.logger.info("Creating box plot for accuracy")
        unique_methods = raw_result["Model"].unique()
        unique_methods = [model for model in unique_methods if "test" not in model]
        plt.figure(figsize=(10, 5))
        data_to_plot = [
            raw_result[raw_result["Model"] == model]["Accuracy Usable"]
            for model in unique_methods
        ]
        plt.boxplot(data_to_plot, labels=unique_methods, patch_artist=True)
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.1)
        plt.title(f"Accuracy for all models")
        path = os.path.join(result_dir, f"box_plot_models.png")
        utils.create_file_with_directories(path, self.logger)
        plt.savefig(path)
        plt.close()

        unique_methods = raw_result["Method"].unique()
        plt.figure(figsize=(10, 5))
        data_to_plot = [
            raw_result[raw_result["Method"] == method]["Accuracy Usable"]
            for method in unique_methods
        ]
        plt.boxplot(data_to_plot, labels=unique_methods, patch_artist=True)
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.1)
        plt.title(f"Accuracy for all methods")
        path = os.path.join(result_dir, f"box_plot_methods.png")
        utils.create_file_with_directories(path, self.logger)
        plt.savefig(path)
        plt.close()

    def stacked_bar_graph_fp_fn_models(self, input_dir, result_dir, raw_result):
        self.logger.info(
            "Creating stacked bar graph for false positive and false negative for each model and method"
        )
        unique_models = raw_result["Model"].unique()
        unique_models = [model for model in unique_models if "test" not in model]
        for model in unique_models:
            result = raw_result[
                model == raw_result["Model"].apply(lambda x: x.split("_")[0])
            ]
            plt.figure(figsize=(10, 5))
            plt.bar(
                result["Method"],
                result["False Positive"],
                label="False Positive",
                color="#1f77b4",
            )
            plt.bar(
                result["Method"],
                result["False Negative"],
                bottom=result["False Positive"],
                label="False Negative",
                color="#ff7f0e",
            )
            plt.xlabel("Method")
            plt.ylabel("Fraction")
            plt.ylim(0, 1.1)
            plt.title(f"False Positive and False Negative for {model}")
            plt.legend()
            path = os.path.join(result_dir, f"stacked_bar_graph_{model}.png")
            utils.create_file_with_directories(path, self.logger)
            plt.savefig(path)
            plt.close()

    def calculate_raw_performance_row(self, file, df):
        model = utils.get_name_by_file(file).split("_")[0]
        method = "binary"
        if "discrete" in file:
            method = "discrete"
        if "percentage" in file:
            method = "percentage"
        if "cot" in file:

            method = "cot"
        model_size = utils.get_model_size_by_file(file)
        usable_predictions = df[df["prediction"] != -1]
        dataset_size = len(df)
        usable_size = len(usable_predictions)
        usable = len(usable_predictions) / dataset_size
        accuracy = self.acc_binary(df)
        accuracy_usable = self.acc_binary(usable_predictions)
        false_positive = self.fp_binary(usable_predictions)
        false_negative = self.fn_binary(usable_predictions)
        true_positive = self.tp_binary(usable_predictions)
        true_negative = self.tn_binary(usable_predictions)
        return {
            "Model": model,
            "Method": method,
            "Model Size": model_size,
            "Dataset Size": dataset_size,
            "Usable Size": usable_size,
            "Usable Predictions": usable,
            "Accuracy": accuracy,
            "Accuracy Usable": accuracy_usable,
            "False Positive": false_positive / usable_size,
            "False Negative": false_negative / usable_size,
            "False Predictions": (true_positive + false_positive) / usable_size,
            "True Predictions": (true_negative + false_negative) / usable_size,
        }

    def visualize_results(self, input_dir, result_dir):
        self.logger.info("Visualizing results")
        prediction_files = os.listdir(input_dir)
        for file in prediction_files:
            file = os.path.join(input_dir, file)
            if "binary" in file or "cot" in file:
                # df = pd.read_csv(file)
                # model = utils.get_name_by_file(file)
                # self.plot_word_cloud(df, model, result_dir)
                # self.plot_confusion_matrix(df, model, result_dir)
                self.visualize_bins(file, result_dir)
                self.visualize_domain_and_sentiment(file, result_dir)
                pass
            if "discrete" in file:
                pass
                # df = pd.read_csv(file)
                # model = utils.get_name_by_file(file)
                # self.plot_word_cloud(df, model, result_dir)
                # self.plot_confusion_matrix(df, model, result_dir)
        self.logger.info("Results visualization completed")

    def analyze_raw_results(self, input_dir, result_dir):
        self.logger.info("Analyzing raw results")
        prediction_files = os.listdir(input_dir)
        result = pd.DataFrame([], columns=raw_results_columns)
        for file in prediction_files:
            file = os.path.join(input_dir, file)
            if "binary" in file or "cot" in file:
                metrics = self.analyze_binary_or_cot(file)
                # set_metrics = set(metrics)
                # set_raw_results_columns = set(raw_results_columns)

                # diff_metrics = set_metrics - set_raw_results_columns
                # diff_raw_results_columns = set_raw_results_columns - set_metrics

                # print("Items in metrics but not in raw_results_columns:", diff_metrics)
                # print(
                #     "Items in raw_results_columns but not in metrics:",
                #     diff_raw_results_columns,
                # )
                result.loc[len(result.index)] = metrics
            if "discrete" in file:
                metrics_includes_true = self.analyze_discrete_includes_true(file)
                result.loc[len(result.index)] = metrics_includes_true
                metrics_above_mostly_true = self.analyze_discrete_above_mostly_true(
                    file
                )
                result.loc[len(result.index)] = metrics_above_mostly_true
            # if "percentage" in file:
            #     try:

            #         # print(df['prediction'].unique())
            #         # df = df.sample(frac=0.01)
            #         metrics = calculate_metrics_percentage_50(df)
            #         # metrics_2 = calculate_metrics_discrete_above_mostly_true(df)
            #         result.loc[len(result.index)] = metrics
            #     except:
            #         pass

        column_subsets = {
            "Basic": [
                "model",
                "total_predictions",
                "usable_predictions",
                "accuracy_all",
                "accuracy_usable",
            ],
            "Usable": [
                "model",
                "accuracy_usable",
                "false_positive_usable",
                "false_negative_usable",
                "true_positive_usable",
                "true_negative_usable",
            ],
            "Entertainment": [
                "model",
                "accuracy_entertainment",
                "false_positive_entertainment",
                "false_negative_entertainment",
                "true_positive_entertainment",
                "true_negative_entertainment",
            ],
            "Health": [
                "model",
                "accuracy_health",
                "false_positive_health",
                "false_negative_health",
                "true_positive_health",
                "true_negative_health",
            ],
            "Politics": [
                "model",
                "accuracy_politics",
                "false_positive_politics",
                "false_negative_politics",
                "true_positive_politics",
                "true_negative_politics",
            ],
            "Undefined": [
                "model",
                "accuracy_undefined",
                "false_positive_undefined",
                "false_negative_undefined",
                "true_positive_undefined",
                "true_negative_undefined",
            ],
            "Usable Entertainment": [
                "model",
                "accuracy_usable_entertainment",
                "false_positive_usable_entertainment",
                "false_negative_usable_entertainment",
                "true_positive_usable_entertainment",
                "true_negative_usable_entertainment",
            ],
            "Usable Health": [
                "model",
                "accuracy_usable_health",
                "false_positive_usable_health",
                "false_negative_usable_health",
                "true_positive_usable_health",
                "true_negative_usable_health",
            ],
            "Usable Politics": [
                "model",
                "accuracy_usable_politics",
                "false_positive_usable_politics",
                "false_negative_usable_politics",
                "true_positive_usable_politics",
                "true_negative_usable_politics",
            ],
            "Usable Undefined": [
                "model",
                "accuracy_usable_undefined",
                "false_positive_usable_undefined",
                "false_negative_usable_undefined",
                "true_positive_usable_undefined",
                "true_negative_usable_undefined",
            ],
            "Positive Sentiment": [
                "model",
                "accuracy_positive_sentiment",
                "false_positive_positive_sentiment",
                "false_negative_positive_sentiment",
                "true_positive_positive_sentiment",
                "true_negative_positive_sentiment",
            ],
            "Negative Sentiment": [
                "model",
                "accuracy_negative_sentiment",
                "false_positive_negative_sentiment",
                "false_negative_negative_sentiment",
                "true_positive_negative_sentiment",
                "true_negative_negative_sentiment",
            ],
            "Neutral Sentiment": [
                "model",
                "accuracy_neutral_sentiment",
                "false_positive_neutral_sentiment",
                "false_negative_neutral_sentiment",
                "true_positive_neutral_sentiment",
                "true_negative_neutral_sentiment",
            ],
            "Usable Positive Sentiment": [
                "model",
                "accuracy_usable_positive_sentiment",
                "false_positive_usable_positive_sentiment",
                "false_negative_usable_positive_sentiment",
                "true_positive_usable_positive_sentiment",
                "true_negative_usable_positive_sentiment",
            ],
            "Usable Negative Sentiment": [
                "model",
                "accuracy_usable_negative_sentiment",
                "false_positive_usable_negative_sentiment",
                "false_negative_usable_negative_sentiment",
                "true_positive_usable_negative_sentiment",
                "true_negative_usable_negative_sentiment",
            ],
            "Usable Neutral Sentiment": [
                "model",
                "accuracy_usable_neutral_sentiment",
                "false_positive_usable_neutral_sentiment",
                "false_negative_usable_neutral_sentiment",
                "true_positive_usable_neutral_sentiment",
                "true_negative_usable_neutral_sentiment",
            ],
        }
        model_options = [
            "gemma-2b",
            "gemma-2-9b",
            "llama-3.1-8b-it",
            "mistral-0.2-7b-it",
        ]
        result_dir = os.path.join(result_dir, "raw_results")

        for option in model_options:
            for name, columns in column_subsets.items():
                df = result[columns]
                df = df[df["model"].str.contains(option)]
                df["model"] = df["model"].str.replace("predictions_", "")
                df = df.applymap(lambda x: round(x, 2) if isinstance(x, float) else x)
                path_csv = os.path.join(result_dir, f"csv_{option}_{name}.csv")
                path_latex = os.path.join(result_dir, f"latex_{option}_{name}.latex")
                utils.create_file_with_directories(path_csv, self.logger)
                utils.create_file_with_directories(path_latex, self.logger)
                df.to_csv(path_csv, index=False)
                df.to_latex(path_latex, index=False, float_format="%.2f")

        self.logger.info("Raw results analysis completed")

    def analyze_binary_or_cot(self, file):
        df = pd.read_csv(file)
        usable_predictions = df[df["prediction"] != -1]

        model = utils.get_name_by_file(file)
        total_predictions = len(df)
        len_usable_predictions = len(usable_predictions)
        accuracy_all = self.acc_binary(df)
        accuracy_usable = self.acc_binary(usable_predictions)

        false_positive_usable = (
            self.fp_binary(usable_predictions) / len_usable_predictions
        )
        false_negative_usable = (
            self.fn_binary(usable_predictions) / len_usable_predictions
        )
        true_positive_usable = (
            self.tp_binary(usable_predictions) / len_usable_predictions
        )
        true_negative_usable = (
            self.tn_binary(usable_predictions) / len_usable_predictions
        )

        entertainment = utils.get_entertainment_news(df)
        politics = utils.get_politics_news(df)
        health = utils.get_health_news(df)
        undefined = utils.get_undefined_news(df)

        accuracy_entertainment = self.acc_binary(entertainment)
        false_positive_entertainment = self.fp_binary(entertainment) / len(
            entertainment
        )
        false_negative_entertainment = self.fn_binary(entertainment) / len(
            entertainment
        )
        true_positive_entertainment = self.tp_binary(entertainment) / len(entertainment)
        true_negative_entertainment = self.tn_binary(entertainment) / len(entertainment)

        accuracy_health = self.acc_binary(health)
        false_positive_health = self.fp_binary(health) / len(health)
        false_negative_health = self.fn_binary(health) / len(health)
        true_positive_health = self.tp_binary(health) / len(health)
        true_negative_health = self.tn_binary(health) / len(health)

        accuracy_politics = self.acc_binary(politics)
        false_positive_politics = self.fp_binary(politics) / len(politics)
        false_negative_politics = self.fn_binary(politics) / len(politics)
        true_positive_politics = self.tp_binary(politics) / len(politics)
        true_negative_politics = self.tn_binary(politics) / len(politics)

        accuracy_undefined = self.acc_binary(undefined)
        false_positive_undefined = self.fp_binary(undefined) / len(undefined)
        false_negative_undefined = self.fn_binary(undefined) / len(undefined)
        true_positive_undefined = self.tp_binary(undefined) / len(undefined)
        true_negative_undefined = self.tn_binary(undefined) / len(undefined)

        entertainment_usable = utils.get_entertainment_news(usable_predictions)
        politics_usable = utils.get_politics_news(usable_predictions)
        health_usable = utils.get_health_news(usable_predictions)
        undefined_usable = utils.get_undefined_news(usable_predictions)

        accuracy_usable_entertainment = self.acc_binary(entertainment_usable)
        false_positive_usable_entertainment = self.fp_binary(
            entertainment_usable
        ) / len(entertainment_usable)
        false_negative_usable_entertainment = self.fn_binary(
            entertainment_usable
        ) / len(entertainment_usable)
        true_positive_usable_entertainment = self.tp_binary(entertainment_usable) / len(
            entertainment_usable
        )
        true_negative_usable_entertainment = self.tn_binary(entertainment_usable) / len(
            entertainment_usable
        )

        accuracy_usable_health = self.acc_binary(health_usable)
        false_positive_usable_health = self.fp_binary(health_usable) / len(
            health_usable
        )
        false_negative_usable_health = self.fn_binary(health_usable) / len(
            health_usable
        )
        true_positive_usable_health = self.tp_binary(health_usable) / len(health_usable)
        true_negative_usable_health = self.tn_binary(health_usable) / len(health_usable)

        accuracy_usable_politics = self.acc_binary(politics_usable)
        false_positive_usable_politics = self.fp_binary(politics_usable) / len(
            politics_usable
        )
        false_negative_usable_politics = self.fn_binary(politics_usable) / len(
            politics_usable
        )
        true_positive_usable_politics = self.tp_binary(politics_usable) / len(
            politics_usable
        )
        true_negative_usable_politics = self.tn_binary(politics_usable) / len(
            politics_usable
        )

        accuracy_usable_undefined = self.acc_binary(undefined_usable)
        false_positive_usable_undefined = self.fp_binary(undefined_usable) / len(
            undefined_usable
        )
        false_negative_usable_undefined = self.fn_binary(undefined_usable) / len(
            undefined_usable
        )
        true_positive_usable_undefined = self.tp_binary(undefined_usable) / len(
            undefined_usable
        )
        true_negative_usable_undefined = self.tn_binary(undefined_usable) / len(
            undefined_usable
        )

        positive_sentiment = utils.get_positive_sentiment(df)
        negative_sentiment = utils.get_negative_sentiment(df)
        neutral_sentiment = utils.get_neutral_sentiment(df)

        accuracy_positive_sentiment = self.acc_binary(positive_sentiment)
        false_positive_positive_sentiment = self.fp_binary(positive_sentiment) / len(
            positive_sentiment
        )
        false_negative_positive_sentiment = self.fn_binary(positive_sentiment) / len(
            positive_sentiment
        )
        true_positive_positive_sentiment = self.tp_binary(positive_sentiment) / len(
            positive_sentiment
        )
        true_negative_positive_sentiment = self.tn_binary(positive_sentiment) / len(
            positive_sentiment
        )

        accuracy_negative_sentiment = self.acc_binary(negative_sentiment)
        false_positive_negative_sentiment = self.fp_binary(negative_sentiment) / len(
            negative_sentiment
        )
        false_negative_negative_sentiment = self.fn_binary(negative_sentiment) / len(
            negative_sentiment
        )
        true_positive_negative_sentiment = self.tp_binary(negative_sentiment) / len(
            negative_sentiment
        )
        true_negative_negative_sentiment = self.tn_binary(negative_sentiment) / len(
            negative_sentiment
        )

        accuracy_neutral_sentiment = self.acc_binary(neutral_sentiment)
        false_positive_neutral_sentiment = self.fp_binary(neutral_sentiment) / len(
            neutral_sentiment
        )
        false_negative_neutral_sentiment = self.fn_binary(neutral_sentiment) / len(
            neutral_sentiment
        )
        true_positive_neutral_sentiment = self.tp_binary(neutral_sentiment) / len(
            neutral_sentiment
        )
        true_negative_neutral_sentiment = self.tn_binary(neutral_sentiment) / len(
            neutral_sentiment
        )

        usable_positive_sentiment = utils.get_positive_sentiment(usable_predictions)
        usable_negative_sentiment = utils.get_negative_sentiment(usable_predictions)
        usable_neutral_sentiment = utils.get_neutral_sentiment(usable_predictions)

        accuracy_usable_positive_sentiment = self.acc_binary(usable_positive_sentiment)
        false_positive_usable_positive_sentiment = self.fp_binary(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)
        false_negative_usable_positive_sentiment = self.fn_binary(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)
        true_positive_usable_positive_sentiment = self.tp_binary(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)
        true_negative_usable_positive_sentiment = self.tn_binary(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)

        accuracy_usable_negative_sentiment = self.acc_binary(usable_negative_sentiment)
        false_positive_usable_negative_sentiment = self.fp_binary(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)
        false_negative_usable_negative_sentiment = self.fn_binary(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)
        true_positive_usable_negative_sentiment = self.tp_binary(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)
        true_negative_usable_negative_sentiment = self.tn_binary(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)

        accuracy_usable_neutral_sentiment = self.acc_binary(usable_neutral_sentiment)
        false_positive_usable_neutral_sentiment = self.fp_binary(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)
        false_negative_usable_neutral_sentiment = self.fn_binary(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)
        true_positive_usable_neutral_sentiment = self.tp_binary(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)
        true_negative_usable_neutral_sentiment = self.tn_binary(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)

        return {
            "model": model,
            "total_predictions": total_predictions,
            "usable_predictions": len_usable_predictions,
            "accuracy_all": accuracy_all,
            "accuracy_usable": accuracy_usable,
            "false_positive_usable": false_positive_usable,
            "false_negative_usable": false_negative_usable,
            "true_positive_usable": true_positive_usable,
            "true_negative_usable": true_negative_usable,
            "accuracy_entertainment": accuracy_entertainment,
            "false_positive_entertainment": false_positive_entertainment,
            "false_negative_entertainment": false_negative_entertainment,
            "true_positive_entertainment": true_positive_entertainment,
            "true_negative_entertainment": true_negative_entertainment,
            "accuracy_health": accuracy_health,
            "false_positive_health": false_positive_health,
            "false_negative_health": false_negative_health,
            "true_positive_health": true_positive_health,
            "true_negative_health": true_negative_health,
            "accuracy_politics": accuracy_politics,
            "false_positive_politics": false_positive_politics,
            "false_negative_politics": false_negative_politics,
            "true_positive_politics": true_positive_politics,
            "true_negative_politics": true_negative_politics,
            "accuracy_undefined": accuracy_undefined,
            "false_positive_undefined": false_positive_undefined,
            "false_negative_undefined": false_negative_undefined,
            "true_positive_undefined": true_positive_undefined,
            "true_negative_undefined": true_negative_undefined,
            "accuracy_usable_entertainment": accuracy_usable_entertainment,
            "false_positive_usable_entertainment": false_positive_usable_entertainment,
            "false_negative_usable_entertainment": false_negative_usable_entertainment,
            "true_positive_usable_entertainment": true_positive_usable_entertainment,
            "true_negative_usable_entertainment": true_negative_usable_entertainment,
            "accuracy_usable_health": accuracy_usable_health,
            "false_positive_usable_health": false_positive_usable_health,
            "false_negative_usable_health": false_negative_usable_health,
            "true_positive_usable_health": true_positive_usable_health,
            "true_negative_usable_health": true_negative_usable_health,
            "accuracy_usable_politics": accuracy_usable_politics,
            "false_positive_usable_politics": false_positive_usable_politics,
            "false_negative_usable_politics": false_negative_usable_politics,
            "true_positive_usable_politics": true_positive_usable_politics,
            "true_negative_usable_politics": true_negative_usable_politics,
            "accuracy_usable_undefined": accuracy_usable_undefined,
            "false_positive_usable_undefined": false_positive_usable_undefined,
            "false_negative_usable_undefined": false_negative_usable_undefined,
            "true_positive_usable_undefined": true_positive_usable_undefined,
            "true_negative_usable_undefined": true_negative_usable_undefined,
            "accuracy_positive_sentiment": accuracy_positive_sentiment,
            "false_positive_positive_sentiment": false_positive_positive_sentiment,
            "false_negative_positive_sentiment": false_negative_positive_sentiment,
            "true_positive_positive_sentiment": true_positive_positive_sentiment,
            "true_negative_positive_sentiment": true_negative_positive_sentiment,
            "accuracy_negative_sentiment": accuracy_negative_sentiment,
            "false_positive_negative_sentiment": false_positive_negative_sentiment,
            "false_negative_negative_sentiment": false_negative_negative_sentiment,
            "true_positive_negative_sentiment": true_positive_negative_sentiment,
            "true_negative_negative_sentiment": true_negative_negative_sentiment,
            "accuracy_neutral_sentiment": accuracy_neutral_sentiment,
            "false_positive_neutral_sentiment": false_positive_neutral_sentiment,
            "false_negative_neutral_sentiment": false_negative_neutral_sentiment,
            "true_positive_neutral_sentiment": true_positive_neutral_sentiment,
            "true_negative_neutral_sentiment": true_negative_neutral_sentiment,
            "accuracy_usable_positive_sentiment": accuracy_usable_positive_sentiment,
            "false_positive_usable_positive_sentiment": false_positive_usable_positive_sentiment,
            "false_negative_usable_positive_sentiment": false_negative_usable_positive_sentiment,
            "true_positive_usable_positive_sentiment": true_positive_usable_positive_sentiment,
            "true_negative_usable_positive_sentiment": true_negative_usable_positive_sentiment,
            "accuracy_usable_negative_sentiment": accuracy_usable_negative_sentiment,
            "false_positive_usable_negative_sentiment": false_positive_usable_negative_sentiment,
            "false_negative_usable_negative_sentiment": false_negative_usable_negative_sentiment,
            "true_positive_usable_negative_sentiment": true_positive_usable_negative_sentiment,
            "true_negative_usable_negative_sentiment": true_negative_usable_negative_sentiment,
            "accuracy_usable_neutral_sentiment": accuracy_usable_neutral_sentiment,
            "false_positive_usable_neutral_sentiment": false_positive_usable_neutral_sentiment,
            "false_negative_usable_neutral_sentiment": false_negative_usable_neutral_sentiment,
            "true_positive_usable_neutral_sentiment": true_positive_usable_neutral_sentiment,
            "true_negative_usable_neutral_sentiment": true_negative_usable_neutral_sentiment,
        }

    def analyze_discrete_includes_true(self, file):
        df = pd.read_csv(file)
        usable_predictions = df[df["prediction"] != -1]

        model = f"{utils.get_name_by_file(file)}_includes_true"
        total_predictions = len(df)
        len_usable_predictions = len(usable_predictions)
        accuracy_all = self.acc_discrete_includes_true(df)
        accuracy_usable = self.acc_discrete_includes_true(usable_predictions)

        false_positive_usable = (
            self.fp_discrete_includes_true(usable_predictions) / len_usable_predictions
        )
        false_negative_usable = (
            self.fn_discrete_includes_true(usable_predictions) / len_usable_predictions
        )
        true_positive_usable = (
            self.tp_discrete_includes_true(usable_predictions) / len_usable_predictions
        )
        true_negative_usable = (
            self.tn_discrete_includes_true(usable_predictions) / len_usable_predictions
        )

        entertainment = utils.get_entertainment_news(df)
        politics = utils.get_politics_news(df)
        health = utils.get_health_news(df)
        undefined = utils.get_undefined_news(df)

        accuracy_entertainment = self.acc_discrete_includes_true(entertainment)
        false_positive_entertainment = self.fp_discrete_includes_true(
            entertainment
        ) / len(entertainment)
        false_negative_entertainment = self.fn_discrete_includes_true(
            entertainment
        ) / len(entertainment)
        true_positive_entertainment = self.tp_discrete_includes_true(
            entertainment
        ) / len(entertainment)
        true_negative_entertainment = self.tn_discrete_includes_true(
            entertainment
        ) / len(entertainment)

        accuracy_health = self.acc_discrete_includes_true(health)
        false_positive_health = self.fp_discrete_includes_true(health) / len(health)
        false_negative_health = self.fn_discrete_includes_true(health) / len(health)
        true_positive_health = self.tp_discrete_includes_true(health) / len(health)
        true_negative_health = self.tn_discrete_includes_true(health) / len(health)

        accuracy_politics = self.acc_discrete_includes_true(politics)
        false_positive_politics = self.fp_discrete_includes_true(politics) / len(
            politics
        )
        false_negative_politics = self.fn_discrete_includes_true(politics) / len(
            politics
        )
        true_positive_politics = self.tp_discrete_includes_true(politics) / len(
            politics
        )
        true_negative_politics = self.tn_discrete_includes_true(politics) / len(
            politics
        )

        accuracy_undefined = self.acc_discrete_includes_true(undefined)
        false_positive_undefined = self.fp_discrete_includes_true(undefined) / len(
            undefined
        )
        false_negative_undefined = self.fn_discrete_includes_true(undefined) / len(
            undefined
        )
        true_positive_undefined = self.tp_discrete_includes_true(undefined) / len(
            undefined
        )
        true_negative_undefined = self.tn_discrete_includes_true(undefined) / len(
            undefined
        )

        entertainment_usable = utils.get_entertainment_news(usable_predictions)
        politics_usable = utils.get_politics_news(usable_predictions)
        health_usable = utils.get_health_news(usable_predictions)
        undefined_usable = utils.get_undefined_news(usable_predictions)

        accuracy_usable_entertainment = self.acc_discrete_includes_true(
            entertainment_usable
        )
        false_positive_usable_entertainment = self.fp_discrete_includes_true(
            entertainment_usable
        ) / len(entertainment_usable)
        false_negative_usable_entertainment = self.fn_discrete_includes_true(
            entertainment_usable
        ) / len(entertainment_usable)
        true_positive_usable_entertainment = self.tp_discrete_includes_true(
            entertainment_usable
        ) / len(entertainment_usable)
        true_negative_usable_entertainment = self.tn_discrete_includes_true(
            entertainment_usable
        ) / len(entertainment_usable)

        accuracy_usable_health = self.acc_discrete_includes_true(health_usable)
        false_positive_usable_health = self.fp_discrete_includes_true(
            health_usable
        ) / len(health_usable)
        false_negative_usable_health = self.fn_discrete_includes_true(
            health_usable
        ) / len(health_usable)
        true_positive_usable_health = self.tp_discrete_includes_true(
            health_usable
        ) / len(health_usable)
        true_negative_usable_health = self.tn_discrete_includes_true(
            health_usable
        ) / len(health_usable)

        accuracy_usable_politics = self.acc_discrete_includes_true(politics_usable)
        false_positive_usable_politics = self.fp_discrete_includes_true(
            politics_usable
        ) / len(politics_usable)
        false_negative_usable_politics = self.fn_discrete_includes_true(
            politics_usable
        ) / len(politics_usable)
        true_positive_usable_politics = self.tp_discrete_includes_true(
            politics_usable
        ) / len(politics_usable)
        true_negative_usable_politics = self.tn_discrete_includes_true(
            politics_usable
        ) / len(politics_usable)

        accuracy_usable_undefined = self.acc_discrete_includes_true(undefined_usable)
        false_positive_usable_undefined = self.fp_discrete_includes_true(
            undefined_usable
        ) / len(undefined_usable)
        false_negative_usable_undefined = self.fn_discrete_includes_true(
            undefined_usable
        ) / len(undefined_usable)
        true_positive_usable_undefined = self.tp_discrete_includes_true(
            undefined_usable
        ) / len(undefined_usable)
        true_negative_usable_undefined = self.tn_discrete_includes_true(
            undefined_usable
        ) / len(undefined_usable)

        positive_sentiment = utils.get_positive_sentiment(df)
        negative_sentiment = utils.get_negative_sentiment(df)
        neutral_sentiment = utils.get_neutral_sentiment(df)

        accuracy_positive_sentiment = self.acc_discrete_includes_true(
            positive_sentiment
        )
        false_positive_positive_sentiment = self.fp_discrete_includes_true(
            positive_sentiment
        ) / len(positive_sentiment)
        false_negative_positive_sentiment = self.fn_discrete_includes_true(
            positive_sentiment
        ) / len(positive_sentiment)
        true_positive_positive_sentiment = self.tp_discrete_includes_true(
            positive_sentiment
        ) / len(positive_sentiment)
        true_negative_positive_sentiment = self.tn_discrete_includes_true(
            positive_sentiment
        ) / len(positive_sentiment)

        accuracy_negative_sentiment = self.acc_discrete_includes_true(
            negative_sentiment
        )
        false_positive_negative_sentiment = self.fp_discrete_includes_true(
            negative_sentiment
        ) / len(negative_sentiment)
        false_negative_negative_sentiment = self.fn_discrete_includes_true(
            negative_sentiment
        ) / len(negative_sentiment)
        true_positive_negative_sentiment = self.tp_discrete_includes_true(
            negative_sentiment
        ) / len(negative_sentiment)
        true_negative_negative_sentiment = self.tn_discrete_includes_true(
            negative_sentiment
        ) / len(negative_sentiment)

        accuracy_neutral_sentiment = self.acc_discrete_includes_true(neutral_sentiment)
        false_positive_neutral_sentiment = self.fp_discrete_includes_true(
            neutral_sentiment
        ) / len(neutral_sentiment)
        false_negative_neutral_sentiment = self.fn_discrete_includes_true(
            neutral_sentiment
        ) / len(neutral_sentiment)
        true_positive_neutral_sentiment = self.tp_discrete_includes_true(
            neutral_sentiment
        ) / len(neutral_sentiment)
        true_negative_neutral_sentiment = self.tn_discrete_includes_true(
            neutral_sentiment
        ) / len(neutral_sentiment)

        usable_positive_sentiment = utils.get_positive_sentiment(usable_predictions)
        usable_negative_sentiment = utils.get_negative_sentiment(usable_predictions)
        usable_neutral_sentiment = utils.get_neutral_sentiment(usable_predictions)

        accuracy_usable_positive_sentiment = self.acc_discrete_includes_true(
            usable_positive_sentiment
        )
        false_positive_usable_positive_sentiment = self.fp_discrete_includes_true(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)
        false_negative_usable_positive_sentiment = self.fn_discrete_includes_true(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)
        true_positive_usable_positive_sentiment = self.tp_discrete_includes_true(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)
        true_negative_usable_positive_sentiment = self.tn_discrete_includes_true(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)

        accuracy_usable_negative_sentiment = self.acc_discrete_includes_true(
            usable_negative_sentiment
        )
        false_positive_usable_negative_sentiment = self.fp_discrete_includes_true(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)
        false_negative_usable_negative_sentiment = self.fn_discrete_includes_true(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)
        true_positive_usable_negative_sentiment = self.tp_discrete_includes_true(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)
        true_negative_usable_negative_sentiment = self.tn_discrete_includes_true(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)

        accuracy_usable_neutral_sentiment = self.acc_discrete_includes_true(
            usable_neutral_sentiment
        )
        false_positive_usable_neutral_sentiment = self.fp_discrete_includes_true(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)
        false_negative_usable_neutral_sentiment = self.fn_discrete_includes_true(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)
        true_positive_usable_neutral_sentiment = self.tp_discrete_includes_true(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)
        true_negative_usable_neutral_sentiment = self.tn_discrete_includes_true(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)

        return {
            "model": model,
            "total_predictions": total_predictions,
            "usable_predictions": len_usable_predictions,
            "accuracy_all": accuracy_all,
            "accuracy_usable": accuracy_usable,
            "false_positive_usable": false_positive_usable,
            "false_negative_usable": false_negative_usable,
            "true_positive_usable": true_positive_usable,
            "true_negative_usable": true_negative_usable,
            "accuracy_entertainment": accuracy_entertainment,
            "false_positive_entertainment": false_positive_entertainment,
            "false_negative_entertainment": false_negative_entertainment,
            "true_positive_entertainment": true_positive_entertainment,
            "true_negative_entertainment": true_negative_entertainment,
            "accuracy_health": accuracy_health,
            "false_positive_health": false_positive_health,
            "false_negative_health": false_negative_health,
            "true_positive_health": true_positive_health,
            "true_negative_health": true_negative_health,
            "accuracy_politics": accuracy_politics,
            "false_positive_politics": false_positive_politics,
            "false_negative_politics": false_negative_politics,
            "true_positive_politics": true_positive_politics,
            "true_negative_politics": true_negative_politics,
            "accuracy_undefined": accuracy_undefined,
            "false_positive_undefined": false_positive_undefined,
            "false_negative_undefined": false_negative_undefined,
            "true_positive_undefined": true_positive_undefined,
            "true_negative_undefined": true_negative_undefined,
            "accuracy_usable_entertainment": accuracy_usable_entertainment,
            "false_positive_usable_entertainment": false_positive_usable_entertainment,
            "false_negative_usable_entertainment": false_negative_usable_entertainment,
            "true_positive_usable_entertainment": true_positive_usable_entertainment,
            "true_negative_usable_entertainment": true_negative_usable_entertainment,
            "accuracy_usable_health": accuracy_usable_health,
            "false_positive_usable_health": false_positive_usable_health,
            "false_negative_usable_health": false_negative_usable_health,
            "true_positive_usable_health": true_positive_usable_health,
            "true_negative_usable_health": true_negative_usable_health,
            "accuracy_usable_politics": accuracy_usable_politics,
            "false_positive_usable_politics": false_positive_usable_politics,
            "false_negative_usable_politics": false_negative_usable_politics,
            "true_positive_usable_politics": true_positive_usable_politics,
            "true_negative_usable_politics": true_negative_usable_politics,
            "accuracy_usable_undefined": accuracy_usable_undefined,
            "false_positive_usable_undefined": false_positive_usable_undefined,
            "false_negative_usable_undefined": false_negative_usable_undefined,
            "true_positive_usable_undefined": true_positive_usable_undefined,
            "true_negative_usable_undefined": true_negative_usable_undefined,
            "accuracy_positive_sentiment": accuracy_positive_sentiment,
            "false_positive_positive_sentiment": false_positive_positive_sentiment,
            "false_negative_positive_sentiment": false_negative_positive_sentiment,
            "true_positive_positive_sentiment": true_positive_positive_sentiment,
            "true_negative_positive_sentiment": true_negative_positive_sentiment,
            "accuracy_negative_sentiment": accuracy_negative_sentiment,
            "false_positive_negative_sentiment": false_positive_negative_sentiment,
            "false_negative_negative_sentiment": false_negative_negative_sentiment,
            "true_positive_negative_sentiment": true_positive_negative_sentiment,
            "true_negative_negative_sentiment": true_negative_negative_sentiment,
            "accuracy_neutral_sentiment": accuracy_neutral_sentiment,
            "false_positive_neutral_sentiment": false_positive_neutral_sentiment,
            "false_negative_neutral_sentiment": false_negative_neutral_sentiment,
            "true_positive_neutral_sentiment": true_positive_neutral_sentiment,
            "true_negative_neutral_sentiment": true_negative_neutral_sentiment,
            "accuracy_usable_positive_sentiment": accuracy_usable_positive_sentiment,
            "false_positive_usable_positive_sentiment": false_positive_usable_positive_sentiment,
            "false_negative_usable_positive_sentiment": false_negative_usable_positive_sentiment,
            "true_positive_usable_positive_sentiment": true_positive_usable_positive_sentiment,
            "true_negative_usable_positive_sentiment": true_negative_usable_positive_sentiment,
            "accuracy_usable_negative_sentiment": accuracy_usable_negative_sentiment,
            "false_positive_usable_negative_sentiment": false_positive_usable_negative_sentiment,
            "false_negative_usable_negative_sentiment": false_negative_usable_negative_sentiment,
            "true_positive_usable_negative_sentiment": true_positive_usable_negative_sentiment,
            "true_negative_usable_negative_sentiment": true_negative_usable_negative_sentiment,
            "accuracy_usable_neutral_sentiment": accuracy_usable_neutral_sentiment,
            "false_positive_usable_neutral_sentiment": false_positive_usable_neutral_sentiment,
            "false_negative_usable_neutral_sentiment": false_negative_usable_neutral_sentiment,
            "true_positive_usable_neutral_sentiment": true_positive_usable_neutral_sentiment,
            "true_negative_usable_neutral_sentiment": true_negative_usable_neutral_sentiment,
        }

    def analyze_discrete_above_mostly_true(self, file):
        df = pd.read_csv(file)
        usable_predictions = df[df["prediction"] != -1]

        model = f"{utils.get_name_by_file(file)}_above_mostly_true"
        total_predictions = len(df)
        len_usable_predictions = len(usable_predictions)
        accuracy_all = self.acc_discrete_above_mostly_true(df)
        accuracy_usable = self.acc_discrete_above_mostly_true(usable_predictions)

        false_positive_usable = (
            self.fp_discrete_above_mostly_true(usable_predictions)
            / len_usable_predictions
        )
        false_negative_usable = (
            self.fn_discrete_above_mostly_true(usable_predictions)
            / len_usable_predictions
        )
        true_positive_usable = (
            self.tp_discrete_above_mostly_true(usable_predictions)
            / len_usable_predictions
        )
        true_negative_usable = (
            self.tn_discrete_above_mostly_true(usable_predictions)
            / len_usable_predictions
        )

        entertainment = utils.get_entertainment_news(df)
        politics = utils.get_politics_news(df)
        health = utils.get_health_news(df)
        undefined = utils.get_undefined_news(df)

        accuracy_entertainment = self.acc_discrete_above_mostly_true(entertainment)
        false_positive_entertainment = self.fp_discrete_above_mostly_true(
            entertainment
        ) / len(entertainment)
        false_negative_entertainment = self.fn_discrete_above_mostly_true(
            entertainment
        ) / len(entertainment)
        true_positive_entertainment = self.tp_discrete_above_mostly_true(
            entertainment
        ) / len(entertainment)
        true_negative_entertainment = self.tn_discrete_above_mostly_true(
            entertainment
        ) / len(entertainment)

        accuracy_health = self.acc_discrete_above_mostly_true(health)
        false_positive_health = self.fp_discrete_above_mostly_true(health) / len(health)
        false_negative_health = self.fn_discrete_above_mostly_true(health) / len(health)
        true_positive_health = self.tp_discrete_above_mostly_true(health) / len(health)
        true_negative_health = self.tn_discrete_above_mostly_true(health) / len(health)

        accuracy_politics = self.acc_discrete_above_mostly_true(politics)
        false_positive_politics = self.fp_discrete_above_mostly_true(politics) / len(
            politics
        )
        false_negative_politics = self.fn_discrete_above_mostly_true(politics) / len(
            politics
        )
        true_positive_politics = self.tp_discrete_above_mostly_true(politics) / len(
            politics
        )
        true_negative_politics = self.tn_discrete_above_mostly_true(politics) / len(
            politics
        )

        accuracy_undefined = self.acc_discrete_above_mostly_true(undefined)
        false_positive_undefined = self.fp_discrete_above_mostly_true(undefined) / len(
            undefined
        )
        false_negative_undefined = self.fn_discrete_above_mostly_true(undefined) / len(
            undefined
        )
        true_positive_undefined = self.tp_discrete_above_mostly_true(undefined) / len(
            undefined
        )
        true_negative_undefined = self.tn_discrete_above_mostly_true(undefined) / len(
            undefined
        )

        entertainment_usable = utils.get_entertainment_news(usable_predictions)
        politics_usable = utils.get_politics_news(usable_predictions)
        health_usable = utils.get_health_news(usable_predictions)
        undefined_usable = utils.get_undefined_news(usable_predictions)

        accuracy_usable_entertainment = self.acc_discrete_above_mostly_true(
            entertainment_usable
        )
        false_positive_usable_entertainment = self.fp_discrete_above_mostly_true(
            entertainment_usable
        ) / len(entertainment_usable)
        false_negative_usable_entertainment = self.fn_discrete_above_mostly_true(
            entertainment_usable
        ) / len(entertainment_usable)
        true_positive_usable_entertainment = self.tp_discrete_above_mostly_true(
            entertainment_usable
        ) / len(entertainment_usable)
        true_negative_usable_entertainment = self.tn_discrete_above_mostly_true(
            entertainment_usable
        ) / len(entertainment_usable)

        accuracy_usable_health = self.acc_discrete_above_mostly_true(health_usable)
        false_positive_usable_health = self.fp_discrete_above_mostly_true(
            health_usable
        ) / len(health_usable)
        false_negative_usable_health = self.fn_discrete_above_mostly_true(
            health_usable
        ) / len(health_usable)
        true_positive_usable_health = self.tp_discrete_above_mostly_true(
            health_usable
        ) / len(health_usable)
        true_negative_usable_health = self.tn_discrete_above_mostly_true(
            health_usable
        ) / len(health_usable)

        accuracy_usable_politics = self.acc_discrete_above_mostly_true(politics_usable)
        false_positive_usable_politics = self.fp_discrete_above_mostly_true(
            politics_usable
        ) / len(politics_usable)
        false_negative_usable_politics = self.fn_discrete_above_mostly_true(
            politics_usable
        ) / len(politics_usable)
        true_positive_usable_politics = self.tp_discrete_above_mostly_true(
            politics_usable
        ) / len(politics_usable)
        true_negative_usable_politics = self.tn_discrete_above_mostly_true(
            politics_usable
        ) / len(politics_usable)

        accuracy_usable_undefined = self.acc_discrete_above_mostly_true(
            undefined_usable
        )
        false_positive_usable_undefined = self.fp_discrete_above_mostly_true(
            undefined_usable
        ) / len(undefined_usable)
        false_negative_usable_undefined = self.fn_discrete_above_mostly_true(
            undefined_usable
        ) / len(undefined_usable)
        true_positive_usable_undefined = self.tp_discrete_above_mostly_true(
            undefined_usable
        ) / len(undefined_usable)
        true_negative_usable_undefined = self.tn_discrete_above_mostly_true(
            undefined_usable
        ) / len(undefined_usable)

        positive_sentiment = utils.get_positive_sentiment(df)
        negative_sentiment = utils.get_negative_sentiment(df)
        neutral_sentiment = utils.get_neutral_sentiment(df)

        accuracy_positive_sentiment = self.acc_discrete_above_mostly_true(
            positive_sentiment
        )
        false_positive_positive_sentiment = self.fp_discrete_above_mostly_true(
            positive_sentiment
        ) / len(positive_sentiment)
        false_negative_positive_sentiment = self.fn_discrete_above_mostly_true(
            positive_sentiment
        ) / len(positive_sentiment)
        true_positive_positive_sentiment = self.tp_discrete_above_mostly_true(
            positive_sentiment
        ) / len(positive_sentiment)
        true_negative_positive_sentiment = self.tn_discrete_above_mostly_true(
            positive_sentiment
        ) / len(positive_sentiment)

        accuracy_negative_sentiment = self.acc_discrete_above_mostly_true(
            negative_sentiment
        )
        false_positive_negative_sentiment = self.fp_discrete_above_mostly_true(
            negative_sentiment
        ) / len(negative_sentiment)
        false_negative_negative_sentiment = self.fn_discrete_above_mostly_true(
            negative_sentiment
        ) / len(negative_sentiment)
        true_positive_negative_sentiment = self.tp_discrete_above_mostly_true(
            negative_sentiment
        ) / len(negative_sentiment)
        true_negative_negative_sentiment = self.tn_discrete_above_mostly_true(
            negative_sentiment
        ) / len(negative_sentiment)

        accuracy_neutral_sentiment = self.acc_discrete_above_mostly_true(
            neutral_sentiment
        )
        false_positive_neutral_sentiment = self.fp_discrete_above_mostly_true(
            neutral_sentiment
        ) / len(neutral_sentiment)
        false_negative_neutral_sentiment = self.fn_discrete_above_mostly_true(
            neutral_sentiment
        ) / len(neutral_sentiment)
        true_positive_neutral_sentiment = self.tp_discrete_above_mostly_true(
            neutral_sentiment
        ) / len(neutral_sentiment)
        true_negative_neutral_sentiment = self.tn_discrete_above_mostly_true(
            neutral_sentiment
        ) / len(neutral_sentiment)

        usable_positive_sentiment = utils.get_positive_sentiment(usable_predictions)
        usable_negative_sentiment = utils.get_negative_sentiment(usable_predictions)
        usable_neutral_sentiment = utils.get_neutral_sentiment(usable_predictions)

        accuracy_usable_positive_sentiment = self.acc_discrete_above_mostly_true(
            usable_positive_sentiment
        )
        false_positive_usable_positive_sentiment = self.fp_discrete_above_mostly_true(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)
        false_negative_usable_positive_sentiment = self.fn_discrete_above_mostly_true(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)
        true_positive_usable_positive_sentiment = self.tp_discrete_above_mostly_true(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)
        true_negative_usable_positive_sentiment = self.tn_discrete_above_mostly_true(
            usable_positive_sentiment
        ) / len(usable_positive_sentiment)

        accuracy_usable_negative_sentiment = self.acc_discrete_above_mostly_true(
            usable_negative_sentiment
        )
        false_positive_usable_negative_sentiment = self.fp_discrete_above_mostly_true(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)
        false_negative_usable_negative_sentiment = self.fn_discrete_above_mostly_true(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)
        true_positive_usable_negative_sentiment = self.tp_discrete_above_mostly_true(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)
        true_negative_usable_negative_sentiment = self.tn_discrete_above_mostly_true(
            usable_negative_sentiment
        ) / len(usable_negative_sentiment)

        accuracy_usable_neutral_sentiment = self.acc_discrete_above_mostly_true(
            usable_neutral_sentiment
        )
        false_positive_usable_neutral_sentiment = self.fp_discrete_above_mostly_true(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)
        false_negative_usable_neutral_sentiment = self.fn_discrete_above_mostly_true(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)
        true_positive_usable_neutral_sentiment = self.tp_discrete_above_mostly_true(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)
        true_negative_usable_neutral_sentiment = self.tn_discrete_above_mostly_true(
            usable_neutral_sentiment
        ) / len(usable_neutral_sentiment)

        return {
            "model": model,
            "total_predictions": total_predictions,
            "usable_predictions": len_usable_predictions,
            "accuracy_all": accuracy_all,
            "accuracy_usable": accuracy_usable,
            "false_positive_usable": false_positive_usable,
            "false_negative_usable": false_negative_usable,
            "true_positive_usable": true_positive_usable,
            "true_negative_usable": true_negative_usable,
            "accuracy_entertainment": accuracy_entertainment,
            "false_positive_entertainment": false_positive_entertainment,
            "false_negative_entertainment": false_negative_entertainment,
            "true_positive_entertainment": true_positive_entertainment,
            "true_negative_entertainment": true_negative_entertainment,
            "accuracy_health": accuracy_health,
            "false_positive_health": false_positive_health,
            "false_negative_health": false_negative_health,
            "true_positive_health": true_positive_health,
            "true_negative_health": true_negative_health,
            "accuracy_politics": accuracy_politics,
            "false_positive_politics": false_positive_politics,
            "false_negative_politics": false_negative_politics,
            "true_positive_politics": true_positive_politics,
            "true_negative_politics": true_negative_politics,
            "accuracy_undefined": accuracy_undefined,
            "false_positive_undefined": false_positive_undefined,
            "false_negative_undefined": false_negative_undefined,
            "true_positive_undefined": true_positive_undefined,
            "true_negative_undefined": true_negative_undefined,
            "accuracy_usable_entertainment": accuracy_usable_entertainment,
            "false_positive_usable_entertainment": false_positive_usable_entertainment,
            "false_negative_usable_entertainment": false_negative_usable_entertainment,
            "true_positive_usable_entertainment": true_positive_usable_entertainment,
            "true_negative_usable_entertainment": true_negative_usable_entertainment,
            "accuracy_usable_health": accuracy_usable_health,
            "false_positive_usable_health": false_positive_usable_health,
            "false_negative_usable_health": false_negative_usable_health,
            "true_positive_usable_health": true_positive_usable_health,
            "true_negative_usable_health": true_negative_usable_health,
            "accuracy_usable_politics": accuracy_usable_politics,
            "false_positive_usable_politics": false_positive_usable_politics,
            "false_negative_usable_politics": false_negative_usable_politics,
            "true_positive_usable_politics": true_positive_usable_politics,
            "true_negative_usable_politics": true_negative_usable_politics,
            "accuracy_usable_undefined": accuracy_usable_undefined,
            "false_positive_usable_undefined": false_positive_usable_undefined,
            "false_negative_usable_undefined": false_negative_usable_undefined,
            "true_positive_usable_undefined": true_positive_usable_undefined,
            "true_negative_usable_undefined": true_negative_usable_undefined,
            "accuracy_positive_sentiment": accuracy_positive_sentiment,
            "false_positive_positive_sentiment": false_positive_positive_sentiment,
            "false_negative_positive_sentiment": false_negative_positive_sentiment,
            "true_positive_positive_sentiment": true_positive_positive_sentiment,
            "true_negative_positive_sentiment": true_negative_positive_sentiment,
            "accuracy_negative_sentiment": accuracy_negative_sentiment,
            "false_positive_negative_sentiment": false_positive_negative_sentiment,
            "false_negative_negative_sentiment": false_negative_negative_sentiment,
            "true_positive_negative_sentiment": true_positive_negative_sentiment,
            "true_negative_negative_sentiment": true_negative_negative_sentiment,
            "accuracy_neutral_sentiment": accuracy_neutral_sentiment,
            "false_positive_neutral_sentiment": false_positive_neutral_sentiment,
            "false_negative_neutral_sentiment": false_negative_neutral_sentiment,
            "true_positive_neutral_sentiment": true_positive_neutral_sentiment,
            "true_negative_neutral_sentiment": true_negative_neutral_sentiment,
            "accuracy_usable_positive_sentiment": accuracy_usable_positive_sentiment,
            "false_positive_usable_positive_sentiment": false_positive_usable_positive_sentiment,
            "false_negative_usable_positive_sentiment": false_negative_usable_positive_sentiment,
            "true_positive_usable_positive_sentiment": true_positive_usable_positive_sentiment,
            "true_negative_usable_positive_sentiment": true_negative_usable_positive_sentiment,
            "accuracy_usable_negative_sentiment": accuracy_usable_negative_sentiment,
            "false_positive_usable_negative_sentiment": false_positive_usable_negative_sentiment,
            "false_negative_usable_negative_sentiment": false_negative_usable_negative_sentiment,
            "true_positive_usable_negative_sentiment": true_positive_usable_negative_sentiment,
            "true_negative_usable_negative_sentiment": true_negative_usable_negative_sentiment,
            "accuracy_usable_neutral_sentiment": accuracy_usable_neutral_sentiment,
            "false_positive_usable_neutral_sentiment": false_positive_usable_neutral_sentiment,
            "false_negative_usable_neutral_sentiment": false_negative_usable_neutral_sentiment,
            "true_positive_usable_neutral_sentiment": true_positive_usable_neutral_sentiment,
            "true_negative_usable_neutral_sentiment": true_negative_usable_neutral_sentiment,
        }

    def fp_binary(self, df):
        return len(df[(df["prediction"] == 1) & (df["label"] == 0)])

    def fn_binary(self, df):
        return len(df[(df["prediction"] == 0) & (df["label"] == 1)])

    def tp_binary(self, df):
        return len(df[(df["prediction"] == 1) & (df["label"] == 1)])

    def tn_binary(self, df):
        return len(df[(df["prediction"] == 0) & (df["label"] == 0)])

    def acc_binary(self, df):
        return len(df[df["prediction"] == df["label"]]) / len(df)

    def acc_discrete_includes_true(self, df):
        boolean_series = df.apply(calculate_discrete_includes_true, axis=1)
        true_count = boolean_series.sum()
        return true_count / len(df)

    def fp_discrete_includes_true(self, df):
        boolean_series = df.apply(calculate_discrete_includes_true_fp, axis=1)
        true_count = boolean_series.sum()
        return true_count / len(df)

    def fn_discrete_includes_true(self, df):
        boolean_series = df.apply(calculate_discrete_includes_true_fn, axis=1)
        true_count = boolean_series.sum()
        return true_count / len(df)

    def tn_discrete_includes_true(self, df):
        boolean_series = df.apply(calculate_discrete_includes_true_tn, axis=1)
        true_count = boolean_series.sum()
        return true_count / len(df)

    def tp_discrete_includes_true(self, df):
        boolean_series = df.apply(calculate_discrete_includes_true_tp, axis=1)
        true_count = boolean_series.sum()
        return true_count / len(df)

    def acc_discrete_above_mostly_true(self, df):
        boolean_series = df.apply(calculate_discrete_above_mostly_true, axis=1)
        true_count = boolean_series.sum()
        return true_count / len(df)

    def fp_discrete_above_mostly_true(self, df):
        boolean_series = df.apply(calculate_discrete_above_mostly_true_fp, axis=1)
        true_count = boolean_series.sum()
        return true_count / len(df)

    def fn_discrete_above_mostly_true(self, df):
        boolean_series = df.apply(calculate_discrete_above_mostly_true_fn, axis=1)
        true_count = boolean_series.sum()
        return true_count / len(df)

    def tn_discrete_above_mostly_true(self, df):
        boolean_series = df.apply(calculate_discrete_above_mostly_true_tn, axis=1)
        true_count = boolean_series.sum()
        return true_count / len(df)

    def tp_discrete_above_mostly_true(self, df):
        boolean_series = df.apply(calculate_discrete_above_mostly_true_tp, axis=1)
        true_count = boolean_series.sum()
        return true_count / len(df)

    def visualize_bins(self, file, result_dir):
        cols = [
            "length",
            "smog_index",
            "flesch_reading_ease",
            "flesch_kincaid_grade_level",
            "coleman_liau_index",
            "gunning_fog_index",
            "ari_index",
            "lix_index",
            "dale_chall_score",
            "dale_chall_known_fraction",
        ]
        df = pd.read_csv(file)
        usable_predictions = df[df["prediction"] != -1]
        result_dir = os.path.join(
            result_dir,
            "visualizations",
            utils.get_name_by_file(file),
        )
        for col in cols:
            if col in df.columns:
                self.visualize_bins_per_col(
                    df,
                    col,
                    result_dir,
                    utils.get_name_by_file(file),
                )
                self.visualize_bins_per_col(
                    usable_predictions,
                    col,
                    result_dir,
                    f"{utils.get_name_by_file(file)}_usable",
                )

    def visualize_bins_per_col(self, df, col, path, model):
        # Create bins for the column
        df = utils.col_bins(df, col)
        bin_name = f"{col}_bin"
        bins = df[bin_name].unique().sort_values()

        # Initialize dictionaries to hold metrics
        accuracy_per_bin = {}
        fp_per_bin = {}
        fn_per_bin = {}

        # Calculate metrics for each bin
        for bin in bins:
            bin_df = df[df[bin_name] == bin]
            total = len(bin_df)

            if total == 0:
                continue

            tp = len(
                bin_df[(bin_df["label"] == 1) & (bin_df["prediction"] == 1)]
            )  # True Positive
            tn = len(
                bin_df[(bin_df["label"] == 0) & (bin_df["prediction"] == 0)]
            )  # True Negative
            fp = len(
                bin_df[(bin_df["label"] == 0) & (bin_df["prediction"] == 1)]
            )  # False Positive
            fn = len(
                bin_df[(bin_df["label"] == 1) & (bin_df["prediction"] == 0)]
            )  # False Negative

            # Calculate the percentages for accuracy, false positives, and false negatives
            accuracy = (tp + tn) / total * 100
            false_positive_rate = fp / total * 100
            false_negative_rate = fn / total * 100

            # Store the values for each bin
            accuracy_per_bin[bin] = accuracy
            fp_per_bin[bin] = false_positive_rate
            fn_per_bin[bin] = false_negative_rate

        # Prepare the data for plotting
        sorted_bins = list(bins)
        accuracy_values = [accuracy_per_bin.get(bin, 0) for bin in sorted_bins]
        fp_values = [fp_per_bin.get(bin, 0) for bin in sorted_bins]
        fn_values = [fn_per_bin.get(bin, 0) for bin in sorted_bins]

        # Plot the metrics
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            sorted_bins, accuracy_values, label="Accuracy", color="green", marker="o"
        )
        ax.plot(
            sorted_bins, fp_values, label="False Positives", color="red", marker="o"
        )
        ax.plot(
            sorted_bins, fn_values, label="False Negatives", color="blue", marker="o"
        )

        # Add titles and labels
        plt.title(f"Accuracy, False Positives, and False Negatives per {col}")
        plt.xlabel(f"{col} bin")
        plt.ylabel("Percentage")
        plt.legend(loc="upper left")

        # Save the plot
        plt.tight_layout()
        news_distribution_by_col_path = os.path.join(
            path, f"{model}_news_metrics_by_{col}.png"
        )
        utils.create_file_with_directories(news_distribution_by_col_path, self.logger)
        plt.savefig(news_distribution_by_col_path)

    def visualize_domain_and_sentiment(self, file, result_dir):
        df = pd.read_csv(file)
        cols = ["sentiment", "topic"]
        result_dir = os.path.join(
            result_dir, "visualizations", utils.get_name_by_file(file)
        )
        df = pd.read_csv(file)
        usable_predictions = df[df["prediction"] != -1]
        for col in cols:
            self.visualize_grouping_per_df(
                df,
                result_dir,
                f"{utils.get_name_by_file(file)}",
                col,
            )
            self.visualize_grouping_per_df(
                df,
                result_dir,
                f"{utils.get_name_by_file(file)}_usable",
                col,
            )

    def visualize_grouping_per_df(self, df, path, model, grouping):
        # Initialize dictionaries to hold metrics
        acc_per_group = df.groupby(grouping).apply(calculate_accuracy_mean)
        plt.figure(figsize=(8, 5))
        plt.bar(acc_per_group.index, acc_per_group)
        plt.title(f"Accuracy per {grouping}")
        plt.xlabel(grouping)
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        save_path = os.path.join(path, f"{model}_metrics_by_{grouping}_accuracy.png")
        utils.create_file_with_directories(save_path, self.logger)
        plt.savefig(save_path)

        fp_per_group = df.groupby(grouping).apply(calculate_fp_mean)
        plt.figure(figsize=(8, 5))
        plt.bar(fp_per_group.index, fp_per_group)
        plt.title(f"False Positives per {grouping}")
        plt.xlabel(grouping)
        plt.ylabel("False Positives")
        plt.ylim(0, 1)
        save_path = os.path.join(path, f"{model}_metrics_by_{grouping}_fp.png")
        utils.create_file_with_directories(save_path, self.logger)
        plt.savefig(save_path)

        fn_per_group = df.groupby(grouping).apply(calculate_fn_mean)
        plt.figure(figsize=(8, 5))
        plt.bar(fn_per_group.index, fn_per_group)
        plt.title(f"False Negatives per {grouping}")
        plt.xlabel(grouping)
        plt.ylabel("False Negatives")
        plt.ylim(0, 1)
        save_path = os.path.join(path, f"{model}_metrics_by_{grouping}_fn.png")
        utils.create_file_with_directories(save_path, self.logger)
        plt.savefig(save_path)

    def plot_metric(self, metric, title, grouped_df, output_file, col):
        plt.figure(figsize=(10, 6))
        for model in grouped_df["Model"].unique():
            model_data = grouped_df[grouped_df["Model"] == model]
            plt.plot(
                model_data[f"{col}_bin"].astype(str),
                model_data[metric],
                marker="o",
                label=model,
            )
        plt.xticks(rotation=45)
        plt.xlabel(f"{col} Bins")
        plt.ylabel(metric.capitalize())
        plt.title(title)
        plt.ylim(0, 1.1)
        plt.legend(title="Model")
        plt.tight_layout()
        utils.create_file_with_directories(output_file, self.logger)
        plt.savefig(output_file)
        plt.close()


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


def calculate_discrete_includes_true(row):
    label = row["label"]
    prediction = row["prediction"]
    if label == 0 and "true" in str(prediction).lower():
        return True
    if label == 1 and "true" not in str(prediction).lower():
        return True
    return False


def calculate_discrete_includes_true_fp(row):
    label = row["label"]
    prediction = row["prediction"]
    if label == 0 and "true" not in str(prediction).lower():
        return True
    return False


def calculate_discrete_includes_true_fn(row):
    label = row["label"]
    prediction = row["prediction"]
    if label == 1 and "true" in str(prediction).lower():
        return True
    return False


def calculate_discrete_includes_true_tn(row):
    label = row["label"]
    prediction = row["prediction"]
    if label == 0 and "true" in str(prediction).lower():
        return True
    return False


def calculate_discrete_includes_true_tp(row):
    label = row["label"]
    prediction = row["prediction"]
    if label == 1 and "false" in str(prediction).lower():
        return True
    return False


def calculate_discrete_above_mostly_true(row):
    label = row["label"]
    prediction = row["prediction"]
    if (
        label == 0
        and "mostly true" in str(prediction).lower()
        or "true" == str(prediction.lower())
    ):
        return True
    if (
        label == 1
        and "mostly true" not in str(prediction).lower()
        and "true" != str(prediction.lower())
    ):
        return True
    return False


def calculate_discrete_above_mostly_true_fp(row):
    label = row["label"]
    prediction = row["prediction"]
    if (
        label == 0
        and "mostly true" not in str(prediction).lower()
        and "true" != str(prediction.lower())
    ):
        return True
    return False


def calculate_discrete_above_mostly_true_fn(row):
    label = row["label"]
    prediction = row["prediction"]
    if (
        label == 1
        and "mostly true" in str(prediction).lower()
        or "true" == str(prediction.lower())
    ):
        return True
    return False


def calculate_discrete_above_mostly_true_tn(row):
    label = row["label"]
    prediction = row["prediction"]
    if (
        label == 0
        and "mostly true" in str(prediction).lower()
        or "true" == str(prediction.lower())
    ):
        return True
    return False


def calculate_discrete_above_mostly_true_tp(row):
    label = row["label"]
    prediction = row["prediction"]
    if (
        label == 1
        and "mostly true" not in str(prediction).lower()
        and "true" != str(prediction.lower())
    ):
        return True
    return False


def calculate_accuracy_mean(sub_df):
    return (sub_df["label"] == sub_df["prediction"]).mean()


def calculate_fp_mean(sub_df):
    fns = (sub_df["label"] == 0) & (sub_df["prediction"] == 1)
    return fns.mean()


def calculate_fn_mean(sub_df):
    fns = (sub_df["label"] == 1) & (sub_df["prediction"] == 0)
    return fns.mean()


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
