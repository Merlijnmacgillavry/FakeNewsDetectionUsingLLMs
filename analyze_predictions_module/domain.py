import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import utils

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


def domain_metrics_table(input_dir, result_dir, logger):
    logger.info("Creating domain metrics table")
    prediction_files = os.listdir(input_dir)
    result = pd.DataFrame([], columns=domain_columns)
    for file in prediction_files:
        file = os.path.join(input_dir, file)
        if "binary" in file or "cot" in file:
            df = pd.read_csv(file)
            model = utils.get_name_by_file(file)
            entertainment = utils.get_entertainment_news(df)
            politics = utils.get_politics_news(df)
            health = utils.get_health_news(df)
            undefined = utils.get_undefined_news(df)
            defined = utils.get_defined_news(df)
            metrics_entertainment = calculate_domain_metrics_row(
                file, entertainment, "Entertainment"
            )
            metrics_politics = calculate_domain_metrics_row(file, politics, "Politics")
            metrics_health = calculate_domain_metrics_row(file, health, "Health")
            metrics_undefined = calculate_domain_metrics_row(
                file, undefined, "Undefined"
            )
            metrics_defined = calculate_domain_metrics_row(file, defined, "Defined")
            if "test" in model:
                metrics_entertainment["Model"] = (
                    metrics_entertainment["Model"] + "_test"
                )
                metrics_politics["Model"] = metrics_politics["Model"] + "_test"
                metrics_health["Model"] = metrics_health["Model"] + "_test"
                metrics_undefined["Model"] = metrics_undefined["Model"] + "_test"
                metrics_defined["Model"] = metrics_defined["Model"] + "_test"
            if "finetuned" in model:
                metrics_entertainment["Model"] = (
                    metrics_entertainment["Model"] + "_finetuned"
                )
                metrics_entertainment["Method"] = (
                    metrics_entertainment["Method"] + "_finetuned"
                )
                metrics_politics["Model"] = metrics_politics["Model"] + "_finetuned"
                metrics_politics["Method"] = metrics_politics["Method"] + "_finetuned"
                metrics_health["Model"] = metrics_health["Model"] + "_finetuned"
                metrics_health["Method"] = metrics_health["Method"] + "_finetuned"
                metrics_undefined["Model"] = metrics_undefined["Model"] + "_finetuned"
                metrics_undefined["Method"] = metrics_undefined["Method"] + "_finetuned"
                metrics_defined["Model"] = metrics_defined["Model"] + "_finetuned"
                metrics_defined["Method"] = metrics_defined["Method"] + "_finetuned"
            result.loc[len(result.index)] = metrics_entertainment
            result.loc[len(result.index)] = metrics_politics
            result.loc[len(result.index)] = metrics_health
            result.loc[len(result.index)] = metrics_undefined
            result.loc[len(result.index)] = metrics_defined
        if "discrete" in file:
            df = pd.read_csv(file)
            df = utils.change_discrete_above_mostly_true_to_binary(df)
            entertainment = utils.get_entertainment_news(df)
            politics = utils.get_politics_news(df)
            health = utils.get_health_news(df)
            undefined = utils.get_undefined_news(df)
            defined = utils.get_defined_news(df)
            metrics_entertainment = calculate_domain_metrics_row(
                file, entertainment, "Entertainment"
            )
            metrics_politics = calculate_domain_metrics_row(file, politics, "Politics")
            metrics_health = calculate_domain_metrics_row(file, health, "Health")
            metrics_undefined = calculate_domain_metrics_row(
                file, undefined, "Undefined"
            )
            metrics_defined = calculate_domain_metrics_row(file, defined, "Defined")
            metrics_entertainment["Method"] = (
                metrics_entertainment["Method"] + "_above_mostly_true"
            )
            metrics_politics["Method"] = (
                metrics_politics["Method"] + "_above_mostly_true"
            )
            metrics_health["Method"] = metrics_health["Method"] + "_above_mostly_true"
            metrics_undefined["Method"] = (
                metrics_undefined["Method"] + "_above_mostly_true"
            )
            metrics_defined["Method"] = metrics_defined["Method"] + "_above_mostly_true"
            result.loc[len(result.index)] = metrics_entertainment
            result.loc[len(result.index)] = metrics_politics
            result.loc[len(result.index)] = metrics_health
            result.loc[len(result.index)] = metrics_undefined
            result.loc[len(result.index)] = metrics_defined
            # metrics = self.calculate_domain_metrics_row(file, df)
            # result.loc[len(result.index)] = metrics
            entertainment = utils.get_entertainment_news(df)
            politics = utils.get_politics_news(df)
            health = utils.get_health_news(df)
            undefined = utils.get_undefined_news(df)
            defined = utils.get_defined_news(df)
            metrics_entertainment = calculate_domain_metrics_row(
                file, entertainment, "Entertainment"
            )
            metrics_politics = calculate_domain_metrics_row(file, politics, "Politics")
            metrics_health = calculate_domain_metrics_row(file, health, "Health")
            metrics_undefined = calculate_domain_metrics_row(
                file, undefined, "Undefined"
            )
            metrics_defined = calculate_domain_metrics_row(file, defined, "Defined")
            metrics_entertainment["Method"] = (
                metrics_entertainment["Method"] + "_includes_true"
            )
            metrics_politics["Method"] = metrics_politics["Method"] + "_includes_true"
            metrics_health["Method"] = metrics_health["Method"] + "_includes_true"
            metrics_undefined["Method"] = metrics_undefined["Method"] + "_includes_true"
            metrics_defined["Method"] = metrics_defined["Method"] + "_includes_true"
            result.loc[len(result.index)] = metrics_entertainment
            result.loc[len(result.index)] = metrics_politics
            result.loc[len(result.index)] = metrics_health
            result.loc[len(result.index)] = metrics_undefined
            result.loc[len(result.index)] = metrics_defined
        if "percentage" in file:
            df = pd.read_csv(file)
            df_50 = utils.change_percentage_above_50_to_binary(df)
            df_75 = utils.change_percentage_above_75_to_binary(df)

            entertainment_50 = utils.get_entertainment_news(df_50)
            politics_50 = utils.get_politics_news(df_50)

            health_50 = utils.get_health_news(df_50)
            undefined_50 = utils.get_undefined_news(df_50)
            defined_50 = utils.get_defined_news(df_50)

            if len(politics_50) > 0:
                metrics_entertainment_50 = calculate_domain_metrics_row(
                    file, entertainment_50, "Entertainment"
                )
                metrics_entertainment_50["Method"] = (
                    metrics_entertainment_50["Method"] + "_above_50"
                )
                result.loc[len(result.index)] = metrics_entertainment_50

            if len(politics_50) > 0:
                metrics_politics_50 = calculate_domain_metrics_row(
                    file, politics_50, "Politics"
                )
                metrics_politics_50["Method"] = (
                    metrics_politics_50["Method"] + "_above_50"
                )
                result.loc[len(result.index)] = metrics_politics_50

            if len(health_50) > 0:
                metrics_health_50 = calculate_domain_metrics_row(
                    file, health_50, "Health"
                )
                metrics_health_50["Method"] = metrics_health_50["Method"] + "_above_50"
                result.loc[len(result.index)] = metrics_health_50
            if len(undefined_50) > 0:
                metrics_undefined_50 = calculate_domain_metrics_row(
                    file, undefined_50, "Undefined"
                )
                metrics_undefined_50["Method"] = (
                    metrics_undefined_50["Method"] + "_above_50"
                )
                result.loc[len(result.index)] = metrics_undefined_50

            if len(defined_50) > 0:
                metrics_defined_50 = calculate_domain_metrics_row(
                    file, defined_50, "Defined"
                )
                metrics_defined_50["Method"] = (
                    metrics_defined_50["Method"] + "_above_50"
                )
                result.loc[len(result.index)] = metrics_defined_50

            entertainment_75 = utils.get_entertainment_news(df_75)
            politics_75 = utils.get_politics_news(df_75)
            health_75 = utils.get_health_news(df_75)
            undefined_75 = utils.get_undefined_news(df_75)
            defined_75 = utils.get_defined_news(df_75)

            if len(entertainment_75) > 0:
                metrics_entertainment_75 = calculate_domain_metrics_row(
                    file, entertainment_75, "Entertainment"
                )
                metrics_entertainment_75["Method"] = (
                    metrics_entertainment_75["Method"] + "_above_75"
                )
                result.loc[len(result.index)] = metrics_entertainment_75

            if len(politics_75) > 0:
                metrics_politics_75 = calculate_domain_metrics_row(
                    file, politics_75, "Politics"
                )
                metrics_politics_75["Method"] = (
                    metrics_politics_75["Method"] + "_above_75"
                )
                result.loc[len(result.index)] = metrics_politics_75

            if len(health_75) > 0:
                metrics_health_75 = calculate_domain_metrics_row(
                    file, health_75, "Health"
                )
                metrics_health_75["Method"] = metrics_health_75["Method"] + "_above_75"
                result.loc[len(result.index)] = metrics_health_75

            if len(undefined_75) > 0:
                metrics_undefined_75 = calculate_domain_metrics_row(
                    file, undefined_75, "Undefined"
                )
                metrics_undefined_75["Method"] = (
                    metrics_undefined_75["Method"] + "_above_75"
                )
                result.loc[len(result.index)] = metrics_undefined_75

            if len(defined_75) > 0:
                metrics_defined_75 = calculate_domain_metrics_row(
                    file, defined_75, "Defined"
                )
                metrics_defined_75["Method"] = (
                    metrics_defined_75["Method"] + "_above_75"
                )
                result.loc[len(result.index)] = metrics_defined_75

    result = result.applymap(lambda x: round(x, 2) if isinstance(x, float) else x)
    result = result.sort_values(by=["Domain", "Model Size", "Method"])
    path = os.path.join(result_dir, "domain_metrics_table.csv")
    utils.create_file_with_directories(path, logger)
    result.to_csv(path, index=False)
    logger.info("Domain metrics table created")
    return result


def calculate_domain_metrics_row(file, df, domain):
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
        "Domain": domain,
        "Model Size": model_size,
        "Usable Predictions": usable,
        "Accuracy": accuracy,
        "Accuracy Usable": accuracy_usable,
        "False Positive": false_positive / usable_size,
        "False Negative": false_negative / usable_size,
        "True Positive": true_positive / usable_size,
        "True Negative": true_negative / usable_size,
    }


def bar_graph_domain_metrics_fp_fn_methods(input_dir, result_dir, raw_result, logger):
    logger.info("Creating stacked bar graph for average domain metrics")
    unique_methods = raw_result["Method"].unique()

    for method in unique_methods:
        # Filter data for the current method
        result = raw_result[raw_result["Method"] == method]
        domains = result["Domain"].unique()
        
        if "finetuned" not in method:
            result = result[~result["Model"].str.contains("test")]

        # Calculate the average False Positive and False Negative per domain
        avg_fp = result.groupby("Domain")["False Positive"].mean()
        avg_fn = result.groupby("Domain")["False Negative"].mean()

        # Create the plot
        plt.figure(figsize=(12, 7))

        # The label locations for each domain
        x = np.arange(len(domains))
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
        plt.xlabel("Domain")
        plt.ylabel("Average Fraction")

        plt.ylim(0, 1.1)  # Dynamically set Y-axis

        # Set the x-axis labels
        plt.xticks(x, domains)

        # Add a legend to differentiate FP and FN
        plt.legend()

        # Show or save the plot
        plt.tight_layout()
        path = os.path.join(result_dir, f"stacked_bar_graph_avg_{method}.png")
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()


def bar_graph_domain_metrics_fp_fn_models(input_dir, result_dir, raw_result, logger):
    logger.info("Creating stacked bar graph for average domain metrics")
    unique_models = raw_result["Model"].unique()
    unique_models = [model for model in unique_models if "test" not in model]

    for model in unique_models:
        # Filter data for the current method
        result = raw_result[
            model == raw_result["Model"].apply(lambda x: x.split("_")[0])
        ]
        domains = result["Domain"].unique()

        # Calculate the average False Positive and False Negative per domain
        avg_fp = result.groupby("Domain")["False Positive"].mean()
        avg_fn = result.groupby("Domain")["False Negative"].mean()

        # Create the plot
        plt.figure(figsize=(12, 7))

        # The label locations for each domain
        x = np.arange(len(domains))
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
        plt.xlabel("Domain")
        plt.ylabel("Average Fraction")

        plt.ylim(0, 1.1)  # Dynamically set Y-axis

        # Set the x-axis labels
        plt.xticks(x, domains)

        # Add a legend to differentiate FP and FN
        plt.legend()

        # Show or save the plot
        plt.tight_layout()
        path = os.path.join(result_dir, f"stacked_bar_graph_avg_{model}.png")
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()


def box_plot_domain_metrics(input_dir, result_dir, raw_result, logger):
    logger.info("Creating box plot for domain metrics")
    unique_methods = raw_result["Method"].unique()
    for method in unique_methods:
        # Filter the DataFrame for the current method
        result = raw_result[method == raw_result["Method"]]
        # Create a box plot for accuracy by domain
        plt.figure(figsize=(10, 6))

        # Group data by Domain and create box plots
        domains = result["Domain"].unique()
        # if "finetuned" not in method:
        #     models = [model for model in models if "test" not in model]
        data_to_plot = [
            result[result["Domain"] == domain]["Accuracy Usable"] for domain in domains
        ]

        bplot = plt.boxplot(data_to_plot, labels=domains, patch_artist=True)
        for patch, color in zip(bplot["boxes"], utils.colors):
            patch.set_facecolor(utils.colors[1])
        # Add labels, title, and customize plot
        plt.xlabel("Domain")
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


# def bar_graph_domain_metrics_accuracy(input_dir, result_dir, raw_result, logger):
#     logger.info("Creating bar graph for domain metrics")
#     unique_methods = raw_result["Method"].unique()
#     for method in unique_methods:
#         result = raw_result[method == raw_result["Method"]]
#         domains = result["Domain"].unique()
#         models = result["Model"].unique()
#         if "finetuned" not in method:
#             models = [model for model in models if "test" not in model]

#         bar_width = 0.2
#         x = np.arange(len(domains))  # The label locations for each domain

#         # Create the plot
#         plt.figure(figsize=(10, 6))
#         print(models)

#         # Loop through models and plot a bar for each one
#         for i, model in enumerate(models):
#             # Get the accuracies for the current model across the domains
#             accuracies = result[result["Model"] == model]["Accuracy Usable"]

#             # Check if the lengths of x and accuracies match
#             if len(x) != len(accuracies):
#                 logger.warning(
#                     f"Shape mismatch for model {model}: "
#                     f"domains ({len(x)}) vs accuracies ({len(accuracies)})"
#                 )
#                 # Handle mismatch: either skip, trim, or pad
#                 accuracies = accuracies[: len(x)]  # Trim accuracies to match x length
#                 # Alternatively, you can skip the model or pad accuracies as needed

#             # Plot the bar for the current model, with each group being offset by bar_width
#             plt.bar(x + i * bar_width, accuracies, width=bar_width, label=model)

#         # Add labels, title, and other plot settings
#         plt.xlabel("Domain")
#         plt.ylabel("Accuracy")
#         plt.ylim(0, 1)  # Accuracy between 0 and 1

#         # Set the x-axis labels
#         plt.xticks(x + bar_width * (len(models) - 1) / 2, domains)

#         # Add a legend to differentiate models
#         plt.legend()

#         # Show or save the plot
#         plt.tight_layout()
#         path = os.path.join(result_dir, f"bar_graph_accuracy_{method}.png")
#         utils.create_file_with_directories(path, logger)
#         plt.savefig(path)
#         plt.close()
