import os
import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt


def raw_performance_table(input_dir, result_dir, logger):
    logger.info("Creating raw performance table")
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
        "False Predictions",
        "True Predictions",
    ]
    prediction_files = os.listdir(input_dir)
    result = pd.DataFrame([], columns=performance_columns)
    for file in prediction_files:
        file = os.path.join(input_dir, file)
        if "binary" in file or "cot" in file:
            df = pd.read_csv(file)
            metrics = calculate_raw_performance_row(file, df)
            model = utils.get_name_by_file(file)
            if "test" in model:
                metrics["Model"] = metrics["Model"] + "_test"
            if "finetuned" in model:
                metrics["Model"] = metrics["Model"] + "_finetuned"
                metrics["Method"] = metrics["Method"] + "_finetuned"
            result.loc[len(result.index)] = metrics

        if "discrete" in file:
            df = pd.read_csv(file)
            df = utils.change_discrete_above_mostly_true_to_binary(df)
            metrics = calculate_raw_performance_row(file, df)
            metrics["Method"] = metrics["Method"] + "_above_mostly_true"
            result.loc[len(result.index)] = metrics

            df = pd.read_csv(file)
            df = utils.change_discrete_includes_true_to_binary(df)
            metrics = calculate_raw_performance_row(file, df)
            metrics["Method"] = metrics["Method"] + "_includes_true"
            result.loc[len(result.index)] = metrics
            # parse discrete to binary results
        if "percentage" in file:
            print(file)
            df = pd.read_csv(file)
            df = utils.change_percentage_above_50_to_binary(df)
            metrics = calculate_raw_performance_row(file, df)
            metrics["Method"] = metrics["Method"] + "_above_50"
            result.loc[len(result.index)] = metrics

            df = pd.read_csv(file)
            df = utils.change_percentage_above_75_to_binary(df)
            metrics = calculate_raw_performance_row(file, df)
            metrics["Method"] = metrics["Method"] + "_above_75"
            result.loc[len(result.index)] = metrics

    result = result.applymap(lambda x: round(x, 2) if isinstance(x, float) else x)

    result = result.sort_values(by=["Model Size", "Method"])
    path = os.path.join(result_dir, "raw_performance_table.csv")
    latex_path = os.path.join(result_dir, "raw_performance_table.tex")
    utils.create_file_with_directories(path, logger)
    utils.create_file_with_directories(latex_path, logger)
    result.to_csv(path, index=False)
    result.to_latex(latex_path, index=False, float_format="%.2f")

    logger.info("Raw performance table created")
    return result


def calculate_raw_performance_row(file, df):
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


def plot_performance_data(df, output_dir, logger, column):
    logger.info("Creating usability plots")
    models = df["Model"].to_list()
    df_other_rows = df[
        df["Model"].str.contains("BERT") | df["Model"].str.contains("RF")
    ]
    df_other_rows["Method"] = df_other_rows["Model"]
    print(df_other_rows)
    # remove test from model names
    models = [model.replace("_test", "").replace("_finetuned", "") for model in models]
    unique_models = list(set(models))
    methods = df["Method"].unique()

    for model in unique_models:
        if "BERT" in model or "RF" in model:
            continue
        # loop over rows of the dataframe
        model_df = df[df["Model"].str.contains(model)]
        combined_df = pd.concat([model_df, df_other_rows])

        # Plotting Accuracy Usable
        plt.figure(figsize=(10, 5))
        plt.bar(combined_df["Method"], combined_df[column], color=utils.colors)

        # Set the limits and labels
        plt.ylim(0, 1.1)
        plt.xlabel("Method")
        plt.ylabel(column)
        plt.xticks(rotation=45, ha="right")

        # Save or show the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_{column}_plot.png")
        plt.close()  # Close the plot to free memory


def plot_performance_false_predictions(df, output_dir, logger):
    logger.info("Creating false negatives and positives plots")

    # Filter for BERT and RF rows
    df_other_rows = df[
        df["Model"].str.contains("BERT", case=False)
        | df["Model"].str.contains("RF", case=False)
    ]

    # Optionally modify the Method column if needed
    df_other_rows["Method"] = df_other_rows[
        "Model"
    ]  # Adjust based on your actual logic

    # Remove test and finetuned from model names
    models = (
        df["Model"]
        .str.replace("_test", "", regex=False)
        .str.replace("_finetuned", "", regex=False)
        .to_list()
    )
    unique_models = list(set(models))

    # Define a custom color palette

    for model in unique_models:
        if "BERT" in model or "RF" in model:
            continue

        # Filter the DataFrame for the current model
        model_df = df[df["Model"].str.contains(model, case=False)]

        # Combine with other relevant rows
        combined_df = pd.concat([model_df, df_other_rows]).drop_duplicates()

        # Prepare the data for the stacked bar chart
        methods = combined_df["Method"]
        false_negatives = combined_df["False Negative"]
        false_positives = combined_df["False Positive"]

        # Plotting False Negatives and False Positives
        plt.figure(figsize=(10, 5))

        plt.bar(
            methods, false_positives, color=utils.colors[0], label="False Positives"
        )
        plt.bar(
            methods,
            false_negatives,
            bottom=false_positives,
            color=utils.colors[1],
            label="False Negatives",
        )

        # Set the limits and labels
        plt.ylim(0, 1.1)  # Adjust y-axis limit
        plt.xlabel("Method")
        plt.ylabel("Fraction")
        plt.xticks(rotation=45, ha="right")
        plt.legend()

        # Save or show the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_false_negatives_and_positives.png")
        plt.close()  # Close the plot to free memory


def include_in_analysis(model):
    if "test" in model or "RF" in model or "BERT" in model:
        return False
    return True


def box_plot_accuracy_models(input_dir, result_dir, raw_result, logger):
    logger.info("Creating box plot for accuracy")
    print(raw_result)
    unique_models = raw_result["Model"].unique()
    unique_models = [model for model in unique_models if include_in_analysis(model)]
    plt.figure(figsize=(10, 5))
    data_to_plot = [
        raw_result[raw_result["Model"].str.contains(model)]["Accuracy Usable"]
        for model in unique_models
    ]
    print(data_to_plot)
    print(unique_models)
    if len(data_to_plot) == 0:
        logger.info("No data to plot")
        return
    bplot = plt.boxplot(data_to_plot, labels=unique_models, patch_artist=True)
    for patch, color in zip(bplot["boxes"], utils.colors):
        patch.set_facecolor(utils.colors[1])
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(result_dir, f"box_plot_models.png")
    utils.create_file_with_directories(path, logger)
    plt.savefig(path)
    plt.close()


def box_plot_accuracy_methods(input_dir, result_dir, raw_result, logger):
    logger.info("Creating box plot for accuracy")
    raw_result = raw_result[
        ~(
            raw_result["Model"].str.contains("BERT")
            | raw_result["Model"].str.contains("RF")
        )
    ]
    unique_methods = raw_result["Method"].unique()
    # unique_methods = [model for model in unique_methods if include_in_analysis(model)]
    plt.figure(figsize=(10, 5))
    data_to_plot = [
        raw_result[raw_result["Method"] == method]["Accuracy Usable"]
        for method in unique_methods
    ]
    if len(data_to_plot) == 0:
        logger.info("No data to plot")
        return
    bplot = plt.boxplot(data_to_plot, labels=unique_methods, patch_artist=True)
    for patch, color in zip(bplot["boxes"], utils.colors):
        patch.set_facecolor(utils.colors[1])
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(result_dir, f"box_plot_methods.png")
    utils.create_file_with_directories(path, logger)
    plt.savefig(path)
    plt.close()


def stacked_bar_graph_fp_fn_methods(input_dir, result_dir, raw_result, logger):
    logger.info(
        "Creating stacked bar graph for false positive and false negative for each model and method"
    )
    raw_result = raw_result[
        ~(
            raw_result["Model"].str.contains("BERT")
            | raw_result["Model"].str.contains("RF")
        )
    ]
    unique_methods = raw_result["Method"].unique()
    for method in unique_methods:
        result = raw_result[method == raw_result["Method"]]
        if "finetuned" not in method:
            result = result[~result["Model"].str.contains("test")]
        else:
            result["Model"] = result["Model"].apply(lambda x: x.split("_")[0])
        plt.figure(figsize=(10, 5))
        plt.bar(
            result["Model"],
            result["False Positive"],
            label="False Positive",
            color=utils.colors[0],
        )
        plt.bar(
            result["Model"],
            result["False Negative"],
            bottom=result["False Positive"],
            label="False Negative",
            color=utils.colors[1],
        )
        plt.xlabel("Model")
        plt.ylabel("Fraction")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(result_dir, f"stacked_bar_graph_{method}.png")
        utils.create_file_with_directories(path, logger)
        plt.savefig(path)
        plt.close()


# def bar_graph_acc_models(self, input_dir, result_dir, raw_result):
#         self.logger.info("Creating bar graph for accuracy")
#         unique_models = raw_result["Model"].unique()
#         unique_models = [model for model in unique_models if "test" not in model]
#         for model in unique_models:
#             result = raw_result[
#                 model == raw_result["Model"].apply(lambda x: x.split("_")[0])
#             ]
#             plt.figure(figsize=(10, 5))
#             plt.bar(result["Method"], result["Accuracy Usable"])
#             plt.ylim(0, 1.1)
#             plt.xlabel("Method")
#             plt.ylabel("Accuracy")
#             plt.title(f"Accuracy for {model}")
#             path = os.path.join(result_dir, f"bar_graph_accuracy_{model}.png")
#             utils.create_file_with_directories(path, self.logger)
#             plt.savefig(path)
#             plt.close()

#     def bar_graph_acc_methods(self, input_dir, result_dir, raw_result):
#         self.logger.info("Creating bar graph for accuracy")
#         unique_methods = raw_result["Method"].unique()
#         for method in unique_methods:
#             result = raw_result[method == raw_result["Method"]]
#             if "finetuned" not in method:
#                 result = result[~result["Model"].str.contains("test")]
#             else:
#                 result["Model"] = result["Model"].apply(lambda x: x.split("_")[0])
#             plt.figure(figsize=(10, 5))
#             plt.bar(result["Model"], result["Accuracy Usable"])
#             plt.ylim(0, 1.1)
#             plt.xlabel("Model")
#             plt.ylabel("Accuracy")
#             plt.title(f"Accuracy for {method}")
#             path = os.path.join(result_dir, f"bar_graph_accuracy_{method}.png")
#             utils.create_file_with_directories(path, self.logger)
#             plt.savefig(path)
#             plt.close()
#   def line_graph_model_size(self, input_dir, result_dir, raw_result):
#         self.logger.info("Creating line graph based on model size")
#         prediction_files = os.listdir(input_dir)
#         unique_methods = raw_result["Method"].unique()
#         # Make accuracy between 0 and 100

#         for method in unique_methods:
#             result = raw_result[raw_result["Method"] == method]
#             result = result.sort_values(by=["Model Size"])
#             plt.figure(figsize=(10, 5))
#             plt.plot(result["Model Size"], result["Accuracy Usable"], label="Accuracy")
#             plt.plot(result["Model Size"], result["Usable Predictions"], label="Usable")
#             plt.xlabel("Model Size")
#             plt.ylim(0, 1.1)
#             plt.ylabel("Accuracy")
#             plt.title(f"Accuracy vs Model Size for {method}")
#             plt.legend()
#             path = os.path.join(result_dir, f"line_graph_{method}.png")
#             utils.create_file_with_directories(path, self.logger)
#             plt.savefig(path)
#             plt.close()
