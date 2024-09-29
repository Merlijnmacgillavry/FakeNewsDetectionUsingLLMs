import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import utils

psychological_columns = [
    "Model",
    "Method",
    "Sentiment",
    "Model Size",
    "Usable Predictions",
    "Accuracy",
    "Accuracy Usable",
    "False Positive",
    "False Negative",
    "True Positive",
    "True Negative",
]


def psychological_metrics_table(input_dir, result_dir, logger):
    logger.info("Creating sentiments metrics table")
    prediction_files = os.listdir(input_dir)
    result = pd.DataFrame([], columns=psychological_columns)
    for file in prediction_files:
        file = os.path.join(input_dir, file)
        if "binary" in file or "cot" in file:
            df = pd.read_csv(file)
            model = utils.get_name_by_file(file)

            positive = utils.get_positive_sentiment(df)
            neutral = utils.get_neutral_sentiment(df)
            negative = utils.get_negative_sentiment(df)

            metrics_positive = calculate_psychological_metrics_row(
                file, positive, "Positive"
            )
            metrics_neutral = calculate_psychological_metrics_row(
                file, neutral, "Neutral"
            )
            metrics_negative = calculate_psychological_metrics_row(
                file, negative, "Negative"
            )

            if "test" in model:
                metrics_negative["Model"] = metrics_negative["Model"] + "_test"
                metrics_neutral["Model"] = metrics_neutral["Model"] + "_test"
                metrics_positive["Model"] = metrics_positive["Model"] + "_test"

            if "finetuned" in model:
                metrics_positive["Model"] = metrics_positive["Model"] + "_finetuned"
                metrics_positive["Method"] = metrics_positive["Method"] + "_finetuned"

                metrics_neutral["Model"] = metrics_neutral["Model"] + "_finetuned"
                metrics_neutral["Method"] = metrics_neutral["Method"] + "_finetuned"

                metrics_negative["Model"] = metrics_negative["Model"] + "_finetuned"
                metrics_negative["Method"] = metrics_negative["Method"] + "_finetuned"

            result.loc[len(result.index)] = metrics_positive
            result.loc[len(result.index)] = metrics_neutral
            result.loc[len(result.index)] = metrics_negative

        if "discrete" in file:
            df = pd.read_csv(file)
            df = utils.change_discrete_above_mostly_true_to_binary(df)
            positive = utils.get_positive_sentiment(df)
            neutral = utils.get_neutral_sentiment(df)
            negative = utils.get_negative_sentiment(df)

            metrics_positive = calculate_psychological_metrics_row(
                file, positive, "Positive"
            )
            metrics_neutral = calculate_psychological_metrics_row(
                file, neutral, "Neutral"
            )
            metrics_negative = calculate_psychological_metrics_row(
                file, negative, "Negative"
            )
            metrics_positive["Method"] = (
                metrics_positive["Method"] + "_above_mostly_true"
            )
            metrics_neutral["Method"] = metrics_neutral["Method"] + "_above_mostly_true"
            metrics_negative["Method"] = (
                metrics_negative["Method"] + "_above_mostly_true"
            )
            result.loc[len(result.index)] = metrics_positive
            result.loc[len(result.index)] = metrics_neutral
            result.loc[len(result.index)] = metrics_negative
            # metrics = self.calculate_domain_metrics_row(file, df)
            # result.loc[len(result.index)] = metrics
            df = pd.read_csv(file)
            df = utils.change_discrete_includes_true_to_binary(df)
            positive = utils.get_positive_sentiment(df)
            neutral = utils.get_neutral_sentiment(df)
            negative = utils.get_negative_sentiment(df)

            metrics_positive = calculate_psychological_metrics_row(
                file, positive, "Positive"
            )
            metrics_neutral = calculate_psychological_metrics_row(
                file, neutral, "Neutral"
            )
            metrics_negative = calculate_psychological_metrics_row(
                file, negative, "Negative"
            )
            metrics_positive["Method"] = metrics_positive["Method"] + "_includes_true"
            metrics_neutral["Method"] = metrics_neutral["Method"] + "_includes_true"
            metrics_negative["Method"] = metrics_negative["Method"] + "_includes_true"
            result.loc[len(result.index)] = metrics_positive
            result.loc[len(result.index)] = metrics_neutral
            result.loc[len(result.index)] = metrics_negative
        if "percentage" in file:
            df = pd.read_csv(file)
            df_50 = utils.change_percentage_above_50_to_binary(df)
            df_75 = utils.change_percentage_above_75_to_binary(df)

            positive_50 = utils.get_positive_sentiment(df_50)
            neutral_50 = utils.get_neutral_sentiment(df_50)
            negative_50 = utils.get_negative_sentiment(df_50)

            positive_75 = utils.get_positive_sentiment(df_75)
            neutral_75 = utils.get_neutral_sentiment(df_75)
            negative_75 = utils.get_negative_sentiment(df_75)

            if len(positive_50) > 0:
                metrics_positive_50 = calculate_psychological_metrics_row(
                    file, positive_50, "Positive"
                )
                metrics_positive_50["Method"] = (
                    metrics_positive_50["Method"] + "_above_50"
                )
                result.loc[len(result.index)] = metrics_positive_50
            if len(neutral_50) > 0:
                metrics_neutral_50 = calculate_psychological_metrics_row(
                    file, neutral_50, "Neutral"
                )
                metrics_neutral_50["Method"] = (
                    metrics_neutral_50["Method"] + "_above_50"
                )
                result.loc[len(result.index)] = metrics_neutral_50
            if len(negative_50) > 0:
                metrics_negative_50 = calculate_psychological_metrics_row(
                    file, negative_50, "Negative"
                )
                metrics_negative_50["Method"] = (
                    metrics_negative_50["Method"] + "_above_50"
                )
                result.loc[len(result.index)] = metrics_negative_50

            if len(positive_75) > 0:
                metrics_positive_75 = calculate_psychological_metrics_row(
                    file, positive_75, "Positive"
                )
                metrics_positive_75["Method"] = (
                    metrics_positive_75["Method"] + "_above_75"
                )
                result.loc[len(result.index)] = metrics_positive_75

            if len(neutral_75) > 0:
                metrics_neutral_75 = calculate_psychological_metrics_row(
                    file, neutral_75, "Neutral"
                )
                metrics_neutral_75["Method"] = (
                    metrics_neutral_75["Method"] + "_above_75"
                )
                result.loc[len(result.index)] = metrics_neutral_75

            if len(negative_75) > 0:
                metrics_negative_75 = calculate_psychological_metrics_row(
                    file, negative_75, "Negative"
                )
                metrics_negative_75["Method"] = (
                    metrics_negative_75["Method"] + "_above_75"
                )
                result.loc[len(result.index)] = metrics_negative_75

    result = result.applymap(lambda x: round(x, 2) if isinstance(x, float) else x)

    result = result.sort_values(by=["Sentiment", "Model Size", "Method"])
    path = os.path.join(result_dir, "sentiment_metrics_table.csv")
    utils.create_file_with_directories(path, logger)
    result.to_csv(path, index=False)
    logger.info("Sentiment metrics table created")
    return result


def calculate_psychological_metrics_row(file, df, sentiment):
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
    accuracy = utils.acc_binary(df)
    accuracy_usable = utils.acc_binary(usable_predictions)
    false_positive = utils.fp_binary(usable_predictions)
    false_negative = utils.fn_binary(usable_predictions)
    true_positive = utils.tp_binary(usable_predictions)
    true_negative = utils.tn_binary(usable_predictions)
    return {
        "Model": model,
        "Method": method,
        "Sentiment": sentiment,
        "Model Size": model_size,
        "Usable Predictions": usable,
        "Accuracy": accuracy,
        "Accuracy Usable": accuracy_usable,
        "False Positive": false_positive / usable_size,
        "False Negative": false_negative / usable_size,
        "True Positive": true_positive / usable_size,
        "True Negative": true_negative / usable_size,
    }


def bar_graph_psychological_metrics_accuracy(input_dir, result_dir, raw_result, logger):
    logger.info("Creating bar graph for psychological metrics")
    unique_methods = raw_result["Method"].unique()
    for method in unique_methods:
        result = raw_result[method == raw_result["Method"]]
        sentiments = result["Sentiment"].unique()
        models = result["Model"].unique()
        if "finetuned" not in method:
            models = [model for model in models if "test" not in model]

        bar_width = 0.1
        x = np.arange(len(sentiments))  # The label locations for each domain

        # Create the plot
        plt.figure(figsize=(10, 6))
        # Loop through models and plot a bar for each one
        for i, model in enumerate(models):
            # Get the accuracies for the current model across the domains
            accuracies = result[result["Model"] == model]["Accuracy Usable"]
            # Plot the bar for the current model, with each group being offset by bar_width
            plt.bar(x + i * bar_width, accuracies, width=bar_width, label=model)

        # Add labels, title, and other plot settings
        plt.xlabel("Sentiment")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)  # Accuracy between 0 and 1

        # Set the x-axis labels
        plt.xticks(x + bar_width * (len(models) - 1) / 2, sentiments)

        # Add a legend to differentiate models
        plt.legend()

        # Show or save the plot
        plt.tight_layout()
        path = os.path.join(result_dir, f"bar_graph_accuracy_{method}.png")
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()


def bar_graph_psychological_metrics_fp_fn_methods(
    input_dir, result_dir, raw_result, logger
):
    logger.info("Creating stacked bar graph for average sentiment metrics")
    unique_methods = raw_result["Method"].unique()

    for method in unique_methods:
        # Filter data for the current method
        result = raw_result[raw_result["Method"] == method]
        sentiments = result["Sentiment"].unique()

        if "finetuned" not in method:
            result = result[~result["Model"].str.contains("test")]

        # Calculate the average False Positive and False Negative per domain
        avg_fp = result.groupby("Sentiment")["False Positive"].mean()
        avg_fn = result.groupby("Sentiment")["False Negative"].mean()

        # Create the plot
        plt.figure(figsize=(12, 7))

        # The label locations for each domain
        x = np.arange(len(sentiments))
        bar_width = 0.4  # Increase the width as there's only one bar per domain now

        # Plot the stacked bars (average FP and FN)
        fp_color = utils.colors[0]  # Blue for False Positive
        fn_color = utils.colors[1]  # Orange for False Negative

        # Plot False Positives
        plt.bar(
            x,
            avg_fp,
            width=bar_width,
            label="Average False Positive",
            color=fp_color,
        )
        # Plot False Negatives on top of False Positives
        plt.bar(
            x,
            avg_fn,
            width=bar_width,
            bottom=avg_fp,
            label="Average False Negative",
            color=fn_color,
        )

        # Add labels, title, and other plot settings
        plt.xlabel("Sentiment")
        plt.ylabel("Average Fraction")

        plt.ylim(0, 1.1)  # Dynamically set Y-axis

        # Set the x-axis labels
        plt.xticks(x, sentiments)

        # Add a legend to differentiate FP and FN
        plt.legend()

        # Show or save the plot
        plt.tight_layout()
        path = os.path.join(result_dir, f"stacked_bar_graph_avg_{method}.png")
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()


def bar_graph_psychological_metrics_fp_fn_methods_sentimental(
    input_dir, result_dir, raw_result, logger
):
    logger.info("Creating stacked bar graph for average sentiment metrics")
    unique_methods = raw_result["Method"].unique()
    raw_result["Sentiment"] = raw_result["Sentiment"].apply(
        lambda x: "Sentimental" if x in ["Positive", "Negative"] else x
    )
    for method in unique_methods:
        # Filter data for the current method
        result = raw_result[raw_result["Method"] == method]
        sentiments = result["Sentiment"].unique()

        if "finetuned" not in method:
            result = result[~result["Model"].str.contains("test")]

        # Calculate the average False Positive and False Negative per domain
        avg_fp = result.groupby("Sentiment")["False Positive"].mean()
        avg_fn = result.groupby("Sentiment")["False Negative"].mean()

        # Create the plot
        plt.figure(figsize=(12, 7))

        # The label locations for each domain
        x = np.arange(len(sentiments))
        bar_width = 0.4  # Increase the width as there's only one bar per domain now

        # Plot the stacked bars (average FP and FN)
        fp_color = utils.colors[0]  # Blue for False Positive
        fn_color = utils.colors[1]  # Orange for False Negative

        # Plot False Positives
        plt.bar(
            x,
            avg_fp,
            width=bar_width,
            label="Average False Positive",
            color=fp_color,
        )
        # Plot False Negatives on top of False Positives
        plt.bar(
            x,
            avg_fn,
            width=bar_width,
            bottom=avg_fp,
            label="Average False Negative",
            color=fn_color,
        )

        # Add labels, title, and other plot settings
        plt.xlabel("Sentiment")
        plt.ylabel("Average Fraction")

        plt.ylim(0, 1.1)  # Dynamically set Y-axis

        # Set the x-axis labels
        plt.xticks(x, sentiments)

        # Add a legend to differentiate FP and FN
        plt.legend()

        # Show or save the plot
        plt.tight_layout()
        path = os.path.join(
            result_dir, f"stacked_bar_graph_avg_{method}_sentimental.png"
        )
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()


def bar_graph_psychological_metrics_fp_fn_methods_sentimental_models(
    input_dir, result_dir, raw_result, logger
):
    logger.info("Creating stacked bar graph for average sentiment metrics")
    raw_result["Model"] = raw_result["Model"].str.replace("_test", "", regex=False)
    unique_methods = raw_result["Model"].unique()
    raw_result["Sentiment"] = raw_result["Sentiment"].apply(
        lambda x: "Sentimental" if x in ["Positive", "Negative"] else x
    )
    for method in unique_methods:
        # Filter data for the current method
        result = raw_result[raw_result["Model"] == method]
        sentiments = result["Sentiment"].unique()

        if "finetuned" not in method:
            result = result[~result["Model"].str.contains("test")]

        # Calculate the average False Positive and False Negative per domain
        avg_fp = result.groupby("Sentiment")["False Positive"].mean()
        avg_fn = result.groupby("Sentiment")["False Negative"].mean()

        # Create the plot
        plt.figure(figsize=(12, 7))

        # The label locations for each domain
        x = np.arange(len(sentiments))
        bar_width = 0.4  # Increase the width as there's only one bar per domain now

        # Plot the stacked bars (average FP and FN)
        fp_color = utils.colors[0]  # Blue for False Positive
        fn_color = utils.colors[1]  # Orange for False Negative

        # Plot False Positives
        plt.bar(
            x,
            avg_fp,
            width=bar_width,
            label="Average False Positive",
            color=fp_color,
        )
        # Plot False Negatives on top of False Positives
        plt.bar(
            x,
            avg_fn,
            width=bar_width,
            bottom=avg_fp,
            label="Average False Negative",
            color=fn_color,
        )

        # Add labels, title, and other plot settings
        plt.xlabel("Sentiment")
        plt.ylabel("Average Fraction")

        plt.ylim(0, 1.1)  # Dynamically set Y-axis

        # Set the x-axis labels
        plt.xticks(x, sentiments)

        # Add a legend to differentiate FP and FN
        plt.legend()

        # Show or save the plot
        plt.tight_layout()
        path = os.path.join(
            result_dir, f"stacked_bar_graph_avg_{method}_sentimental.png"
        )
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()


def bar_graph_psychological_metrics_fp_fn_models(
    input_dir, result_dir, raw_result, logger
):
    logger.info("Creating stacked bar graph for average psychological metrics")
    unique_models = raw_result["Model"].unique()
    unique_models = [model for model in unique_models if "test" not in model]

    for model in unique_models:
        # Filter data for the current method
        result = raw_result[
            model == raw_result["Model"].apply(lambda x: x.split("_")[0])
        ]
        sentiments = result["Sentiment"].unique()

        # Calculate the average False Positive and False Negative per domain
        avg_fp = result.groupby("Sentiment")["False Positive"].mean()
        avg_fn = result.groupby("Sentiment")["False Negative"].mean()

        # Create the plot
        plt.figure(figsize=(12, 7))

        # The label locations for each domain
        x = np.arange(len(sentiments))
        bar_width = 0.4  # Increase the width as there's only one bar per domain now

        # Plot the stacked bars (average FP and FN)
        fp_color = utils.colors[0]  # Blue for False Positive
        fn_color = utils.colors[1]  # Orange for False Negative

        # Plot False Positives
        plt.bar(
            x,
            avg_fp,
            width=bar_width,
            label="Average False Positive",
            color=fp_color,
        )
        # Plot False Negatives on top of False Positives
        plt.bar(
            x,
            avg_fn,
            width=bar_width,
            bottom=avg_fp,
            label="Average False Negative",
            color=fn_color,
        )

        # Add labels, title, and other plot settings
        plt.xlabel("Sentiment")
        plt.ylabel("Average Fraction")
        plt.ylim(0, 1.1)  # Dynamically set Y-axis

        # Set the x-axis labels
        plt.xticks(x, sentiments)

        # Add a legend to differentiate FP and FN
        plt.legend()

        # Show or save the plot
        plt.tight_layout()
        path = os.path.join(result_dir, f"stacked_bar_graph_avg_{model}.png")
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()


def box_plot_psychological_metrics(input_dir, result_dir, raw_result, logger):
    logger.info("Creating box plot for psychological metrics")
    unique_methods = raw_result["Method"].unique()
    for method in unique_methods:
        # Filter the DataFrame for the current method
        result = raw_result[method == raw_result["Method"]]
        # Create a box plot for accuracy by domain
        plt.figure(figsize=(10, 6))

        # Group data by Domain and create box plots
        sentiments = result["Sentiment"].unique()
        # if "finetuned" not in method:
        #     models = [model for model in models if "test" not in model]
        data_to_plot = [
            result[result["Sentiment"] == sentiment]["Accuracy Usable"]
            for sentiment in sentiments
        ]

        bplot = plt.boxplot(data_to_plot, labels=sentiments, patch_artist=True)
        for patch, color in zip(bplot["boxes"], utils.colors):
            patch.set_facecolor(utils.colors[1])
        # Add labels, title, and customize plot
        plt.xlabel("Sentiment")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)  # Accuracy between 0 and 1

        # Show grid for better readability
        plt.grid(True)

        # Save or show the plot
        plt.tight_layout()
        path = os.path.join(result_dir, f"box_plot_accuracy_{method}.png")
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()


def box_plot_psychological_sentimental_metrics(
    input_dir, result_dir, raw_result, logger
):
    logger.info("Creating box plot for psychological metrics")
    unique_methods = raw_result["Method"].unique()
    raw_result["Sentiment"] = raw_result["Sentiment"].apply(
        lambda x: "Sentimental" if x in ["Positive", "Negative"] else x
    )
    for method in unique_methods:
        # Filter the DataFrame for the current method
        result = raw_result[method == raw_result["Method"]]
        # Create a box plot for accuracy by domain
        plt.figure(figsize=(10, 6))

        # Group data by Domain and create box plots
        sentiments = result["Sentiment"].unique()
        # if "finetuned" not in method:
        #     models = [model for model in models if "test" not in model]
        data_to_plot = [
            result[result["Sentiment"] == sentiment]["Accuracy Usable"]
            for sentiment in sentiments
        ]

        bplot = plt.boxplot(data_to_plot, labels=sentiments, patch_artist=True)
        for patch, color in zip(bplot["boxes"], utils.colors):
            patch.set_facecolor(utils.colors[1])
        # Add labels, title, and customize plot
        plt.xlabel("Sentiment")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)  # Accuracy between 0 and 1

        # Show grid for better readability
        plt.grid(True)

        # Save or show the plot
        plt.tight_layout()
        path = os.path.join(result_dir, f"box_plot_accuracy_sentimental_{method}.png")
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()


def box_plot_psychological_sentimental_metrics_models(
    input_dir, result_dir, raw_result, logger
):
    logger.info("Creating box plot for psychological metrics")
    # raw_result["Model"] = raw_result["Model"].str.replace(
    #     "_test_finetuned", "", regex=False
    # )
    raw_result["Model"] = raw_result["Model"].str.replace("_test", "", regex=False)

    unique_methods = raw_result["Model"].unique()
    print(unique_methods)
    raw_result["Sentiment"] = raw_result["Sentiment"].apply(
        lambda x: "Sentimental" if x in ["Positive", "Negative"] else x
    )
    for method in unique_methods:
        # Filter the DataFrame for the current method
        result = raw_result[method == raw_result["Model"]]
        # Create a box plot for accuracy by domain
        plt.figure(figsize=(10, 6))

        # Group data by Domain and create box plots
        sentiments = result["Sentiment"].unique()
        # if "finetuned" not in method:
        #     models = [model for model in models if "test" not in model]
        data_to_plot = [
            result[result["Sentiment"] == sentiment]["Accuracy Usable"]
            for sentiment in sentiments
        ]

        bplot = plt.boxplot(data_to_plot, labels=sentiments, patch_artist=True)
        for patch, color in zip(bplot["boxes"], utils.colors):
            patch.set_facecolor(utils.colors[1])
        # Add labels, title, and customize plot
        plt.xlabel("Sentiment")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)  # Accuracy between 0 and 1

        # Show grid for better readability
        plt.grid(True)

        # Save or show the plot
        plt.tight_layout()
        path = os.path.join(result_dir, f"box_plot_accuracy_sentimental_{method}.png")
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()


def box_plot_psychological_metrics_models(input_dir, result_dir, raw_result, logger):
    logger.info("Creating box plot for psychological metrics")
    # raw_result["Model"] = raw_result["Model"].str.replace(
    #     "_test_finetuned", "", regex=False
    # )
    raw_result["Model"] = raw_result["Model"].str.replace("_test", "", regex=False)

    unique_methods = raw_result["Model"].unique()
    print(unique_methods)
    # raw_result["Sentiment"] = raw_result["Sentiment"].apply(
    #     lambda x: "Sentimental" if x in ["Positive", "Negative"] else x
    # )
    for method in unique_methods:
        # Filter the DataFrame for the current method
        result = raw_result[method == raw_result["Model"]]
        # Create a box plot for accuracy by domain
        plt.figure(figsize=(10, 6))

        # Group data by Domain and create box plots
        sentiments = result["Sentiment"].unique()
        # if "finetuned" not in method:
        #     models = [model for model in models if "test" not in model]
        data_to_plot = [
            result[result["Sentiment"] == sentiment]["Accuracy Usable"]
            for sentiment in sentiments
        ]

        bplot = plt.boxplot(data_to_plot, labels=sentiments, patch_artist=True)
        for patch, color in zip(bplot["boxes"], utils.colors):
            patch.set_facecolor(utils.colors[1])
        # Add labels, title, and customize plot
        plt.xlabel("Sentiment")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)  # Accuracy between 0 and 1

        # Show grid for better readability
        plt.grid(True)

        # Save or show the plot
        plt.tight_layout()
        path = os.path.join(result_dir, f"box_plot_accuracy_{method}.png")
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()
