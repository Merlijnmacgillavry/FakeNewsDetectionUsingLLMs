import os

from matplotlib import pyplot as plt
import pandas as pd
import utils


def include_in_analysis(model):
    if "test" in model and not "finetuned" in model:
        return False
    return True


def text_characteristics_metrics_model(input_dir, result_dir, col, logger):
    logger.info("Calculating text length metrics")
    prediction_files = os.listdir(input_dir)
    model_options = utils.model_options
    for model in model_options:
        output_dir = os.path.join(result_dir, model)
        dfs = []
        combined_df = pd.DataFrame()
        for file in prediction_files:
            if model in file and include_in_analysis(model):
                file = os.path.join(input_dir, file)
                if "binary" in file or "cot" in file:
                    df = pd.read_csv(file)
                    model_name = utils.get_name_by_file(file)
                    df["Model"] = model_name
                    dfs.append(df)
                    # combined_df = pd.concat([combined_df, df], ignore_index=True)
                if "discrete" in file:
                    df = pd.read_csv(file)
                    df_mostly_true = utils.change_discrete_above_mostly_true_to_binary(
                        df
                    )
                    df_above_true = utils.change_discrete_includes_true_to_binary(df)
                    model_name = utils.get_name_by_file(file)
                    df_mostly_true["Model"] = f"{model_name}_above_mostly_true"
                    df_above_true["Model"] = f"{model_name}_includes_true"
                    dfs.append(df_mostly_true)
                    dfs.append(df_above_true)
                if "percentage" in file:
                    df = pd.read_csv(file)
                    df_50 = utils.change_percentage_above_50_to_binary(df)
                    df_75 = utils.change_percentage_above_75_to_binary(df)
                    model_name = utils.get_name_by_file(file)
                    df_50["Model"] = f"{model_name}_above_50"
                    df_75["Model"] = f"{model_name}_above_75"
                    dfs.append(df_50)
                    dfs.append(df_75)

        if len(dfs) == 0:
            continue
        for i, df in enumerate(dfs):
            # Calculate accuracy, FP, and FN based on the prediction and label columns
            df["correct"] = (df["prediction"] == df["label"]).astype(
                int
            )  # 1 if correct, 0 if not
            df["fp"] = ((df["prediction"] == 1) & (df["label"] == 0)).astype(
                int
            )  # 1 if false positive
            df["fn"] = ((df["prediction"] == 0) & (df["label"] == 1)).astype(
                int
            )  # 1 if false negative
            df["accuracy"] = df[
                "correct"
            ]  # Accuracy is calculated as the proportion of correct predictions
            # Concatenate the DataFrame into combined_df
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            # Bin the 'length' into 20 bins
        combined_df = utils.col_bins(combined_df, col)
        # Group by the model and the length_bin and calculate mean accuracy, FP, and FN
        grouped_df = (
            combined_df.groupby(["Model", f"{col}_bin"])
            .agg(
                {
                    "accuracy": "mean",  # Proportion of correct predictions (accuracy)
                    "fp": "mean",  # Total number of false positives
                    "fn": "mean",  # Total number of false negatives
                }
            )
            .reset_index()
        )

        plot_metric(
            "accuracy",
            "Accuracy across Text Length Bins",
            grouped_df,
            os.path.join(output_dir, f"accuracy_{col}_bins.png"),
            col,
            logger,
        )
        plot_metric(
            "fp",
            "False Positives across Text Length Bins",
            grouped_df,
            os.path.join(output_dir, f"fp_{col}_bins.png"),
            col,
            logger,
        )
        plot_metric(
            "fn",
            "False Negatives across Text Length Bins",
            grouped_df,
            os.path.join(output_dir, f"fn_{col}_bins.png"),
            col,
            logger,
        )


def text_chracteristics_metrics_method(input_dir, result_dir, col, logger):
    logger.info("Calculating text length metrics")
    prediction_files = os.listdir(input_dir)
    method_options = utils.method_options
    for method in method_options:
        output_dir = os.path.join(result_dir, method)
        dfs = []
        combined_df = pd.DataFrame()
        for file in prediction_files:
            if method in file:
                file = os.path.join(input_dir, file)
                name = utils.get_name_by_file(file)

                if ("binary" in file and "binary" in method) and not "test" in file:
                    df = pd.read_csv(file)
                    model_name = utils.get_name_by_file(file)
                    df["Model"] = model_name
                    dfs.append(df)
                    # combined_df = pd.concat([combined_df, df], ignore_index=True)
                if ("cot" in file and "cot" in method) and not "test" in file:
                    df = pd.read_csv(file)
                    model_name = utils.get_name_by_file(file)
                    df["Model"] = model_name
                    dfs.append(df)
                if "finetune" in file:
                    df = pd.read_csv(file)
                    model_name = utils.get_name_by_file(file)
                    df["Model"] = model_name
                    dfs.append(df)
                if "discrete" in file:
                    df = pd.read_csv(file)
                    df_mostly_true = utils.change_discrete_above_mostly_true_to_binary(
                        df
                    )
                    df_above_true = utils.change_discrete_includes_true_to_binary(df)
                    model_name = utils.get_name_by_file(file)
                    df_mostly_true["Model"] = f"{model_name}_above_mostly_true"
                    df_above_true["Model"] = f"{model_name}_includes_true"
                    dfs.append(df_mostly_true)
                    dfs.append(df_above_true)
                if "percentage" in file:
                    df = pd.read_csv(file)
                    df_50 = utils.change_percentage_above_50_to_binary(df)
                    df_75 = utils.change_percentage_above_75_to_binary(df)
                    model_name = utils.get_name_by_file(file)
                    df_50["Model"] = f"{model_name}_above_50"
                    df_75["Model"] = f"{model_name}_above_75"
                    dfs.append(df_50)
                    dfs.append(df_75)
        if len(dfs) == 0:
            continue
        for i, df in enumerate(dfs):
            # Calculate accuracy, FP, and FN based on the prediction and label columns
            df["correct"] = (df["prediction"] == df["label"]).astype(
                int
            )  # 1 if correct, 0 if not
            df["fp"] = ((df["prediction"] == 1) & (df["label"] == 0)).astype(
                int
            )  # 1 if false positive
            df["fn"] = ((df["prediction"] == 0) & (df["label"] == 1)).astype(
                int
            )  # 1 if false negative
            df["accuracy"] = df[
                "correct"
            ]  # Accuracy is calculated as the proportion of correct predictions
            # Concatenate the DataFrame into combined_df
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            # Bin the 'length' into 20 bins
        combined_df = utils.col_bins(combined_df, col)

        # Group by the model and the length_bin and calculate mean accuracy, FP, and FN
        grouped_df = (
            combined_df.groupby(["Model", f"{col}_bin"])
            .agg(
                {
                    "accuracy": "mean",  # Proportion of correct predictions (accuracy)
                    "fp": "mean",  # Total number of false positives
                    "fn": "mean",  # Total number of false negatives
                }
            )
            .reset_index()
        )
        # print(grouped_df)

        plot_metric(
            "accuracy",
            f"Accuracy across {col} Bins",
            grouped_df,
            os.path.join(output_dir, f"accuracy_{col}_bins.png"),
            col,
            logger,
        )
        plot_metric(
            "fp",
            f"False Positives across {col} Bins",
            grouped_df,
            os.path.join(output_dir, f"fp_{col}_bins.png"),
            col,
            logger,
        )
        plot_metric(
            "fn",
            f"False Negatives across {col} Bins",
            grouped_df,
            os.path.join(output_dir, f"fn_{col}_bins.png"),
            col,
            logger,
        )


def text_characteristics_metrics_model_avg(input_dir, result_dir, col, logger):
    logger.info("Calculating text length metrics")
    prediction_files = os.listdir(input_dir)
    model_options = utils.model_options
    for model in model_options:
        output_dir = os.path.join(result_dir, model)
        dfs = []
        combined_df = pd.DataFrame()
        for file in prediction_files:
            if model in file and include_in_analysis(model):
                file = os.path.join(input_dir, file)
                if "binary" in file or "cot" in file:
                    df = pd.read_csv(file)
                    model_name = utils.get_name_by_file(file)
                    df["Model"] = model_name
                    dfs.append(df)
                    # combined_df = pd.concat([combined_df, df], ignore_index=True)
                if "discrete" in file:
                    df = pd.read_csv(file)
                    df_mostly_true = utils.change_discrete_above_mostly_true_to_binary(
                        df
                    )
                    df_above_true = utils.change_discrete_includes_true_to_binary(df)
                    model_name = utils.get_name_by_file(file)
                    df_mostly_true["Model"] = f"{model_name}_above_mostly_true"
                    df_above_true["Model"] = f"{model_name}_includes_true"
                    dfs.append(df_mostly_true)
                    dfs.append(df_above_true)
                if "percentage" in file:
                    df = pd.read_csv(file)
                    df_50 = utils.change_percentage_above_50_to_binary(df)
                    df_75 = utils.change_percentage_above_75_to_binary(df)
                    model_name = utils.get_name_by_file(file)
                    df_50["Model"] = f"{model_name}_above_50"
                    df_75["Model"] = f"{model_name}_above_75"
                    dfs.append(df_50)
                    dfs.append(df_75)

        if len(dfs) == 0:
            continue
        for i, df in enumerate(dfs):
            # Calculate accuracy, FP, and FN based on the prediction and label columns
            df["correct"] = (df["prediction"] == df["label"]).astype(
                int
            )  # 1 if correct, 0 if not
            df["fp"] = ((df["prediction"] == 1) & (df["label"] == 0)).astype(
                int
            )  # 1 if false positive
            df["fn"] = ((df["prediction"] == 0) & (df["label"] == 1)).astype(
                int
            )  # 1 if false negative
            df["accuracy"] = df[
                "correct"
            ]  # Accuracy is calculated as the proportion of correct predictions
            # Concatenate the DataFrame into combined_df
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            # Bin the 'length' into 20 bins
        combined_df = utils.col_bins(combined_df, col)
        # Group by the model and the length_bin and calculate mean accuracy, FP, and FN
        grouped_df = (
            combined_df.groupby([f"{col}_bin"])
            .agg(
                {
                    "accuracy": "mean",  # Proportion of correct predictions (accuracy)
                    "fp": "mean",  # Total number of false positives
                    "fn": "mean",  # Total number of false negatives
                }
            )
            .reset_index()
        )

        plot_metric_avg(
            "accuracy",
            "Accuracy across Text Length Bins",
            grouped_df,
            os.path.join(output_dir, f"accuracy_{col}_bins_avg.png"),
            col,
            logger,
        )
        plot_metric_avg(
            "fp",
            "False Positives across Text Length Bins",
            grouped_df,
            os.path.join(output_dir, f"fp_{col}_bins_avg.png"),
            col,
            logger,
        )
        plot_metric_avg(
            "fn",
            "False Negatives across Text Length Bins",
            grouped_df,
            os.path.join(output_dir, f"fn_{col}_bins_avg.png"),
            col,
            logger,
        )


def text_chracteristics_metrics_method_avg(input_dir, result_dir, col, logger):
    logger.info("Calculating text length metrics")
    prediction_files = os.listdir(input_dir)
    method_options = utils.method_options
    for method in method_options:
        output_dir = os.path.join(result_dir, method)
        dfs = []
        combined_df = pd.DataFrame()
        for file in prediction_files:
            if method in file:
                file = os.path.join(input_dir, file)
                name = utils.get_name_by_file(file)

                if ("binary" in file and "binary" in method) and not "test" in file:
                    df = pd.read_csv(file)
                    model_name = utils.get_name_by_file(file)
                    df["Model"] = model_name
                    dfs.append(df)
                    # combined_df = pd.concat([combined_df, df], ignore_index=True)
                if ("cot" in file and "cot" in method) and not "test" in file:
                    df = pd.read_csv(file)
                    model_name = utils.get_name_by_file(file)
                    df["Model"] = model_name
                    dfs.append(df)
                if "finetune" in file:
                    df = pd.read_csv(file)
                    model_name = utils.get_name_by_file(file)
                    df["Model"] = model_name
                    dfs.append(df)
                if "discrete" in file:
                    df = pd.read_csv(file)
                    df_mostly_true = utils.change_discrete_above_mostly_true_to_binary(
                        df
                    )
                    df_above_true = utils.change_discrete_includes_true_to_binary(df)
                    model_name = utils.get_name_by_file(file)
                    df_mostly_true["Model"] = f"{model_name}_above_mostly_true"
                    df_above_true["Model"] = f"{model_name}_includes_true"
                    dfs.append(df_mostly_true)
                    dfs.append(df_above_true)
                if "percentage" in file:
                    df = pd.read_csv(file)
                    df_50 = utils.change_percentage_above_50_to_binary(df)
                    df_75 = utils.change_percentage_above_75_to_binary(df)
                    model_name = utils.get_name_by_file(file)
                    df_50["Model"] = f"{model_name}_above_50"
                    df_75["Model"] = f"{model_name}_above_75"
                    dfs.append(df_50)
                    dfs.append(df_75)
        if len(dfs) == 0:
            continue
        for i, df in enumerate(dfs):
            # Calculate accuracy, FP, and FN based on the prediction and label columns
            df["correct"] = (df["prediction"] == df["label"]).astype(
                int
            )  # 1 if correct, 0 if not
            df["fp"] = ((df["prediction"] == 1) & (df["label"] == 0)).astype(
                int
            )  # 1 if false positive
            df["fn"] = ((df["prediction"] == 0) & (df["label"] == 1)).astype(
                int
            )  # 1 if false negative
            df["accuracy"] = df[
                "correct"
            ]  # Accuracy is calculated as the proportion of correct predictions
            # Concatenate the DataFrame into combined_df
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            # Bin the 'length' into 20 bins
        combined_df = utils.col_bins(combined_df, col)

        # Group by the model and the length_bin and calculate mean accuracy, FP, and FN
        grouped_df = (
            combined_df.groupby([f"{col}_bin"])
            .agg(
                {
                    "accuracy": "mean",  # Proportion of correct predictions (accuracy)
                    "fp": "mean",  # Total number of false positives
                    "fn": "mean",  # Total number of false negatives
                }
            )
            .reset_index()
        )
        # print(grouped_df)

        plot_metric_avg(
            "accuracy",
            f"Accuracy across {col} Bins",
            grouped_df,
            os.path.join(output_dir, f"accuracy_{col}_bins_avg.png"),
            col,
            logger,
        )
        plot_metric_avg(
            "fp",
            f"False Positives across {col} Bins",
            grouped_df,
            os.path.join(output_dir, f"fp_{col}_bins_avg.png"),
            col,
            logger,
        )
        plot_metric_avg(
            "fn",
            f"False Negatives across {col} Bins",
            grouped_df,
            os.path.join(output_dir, f"fn_{col}_bins_avg.png"),
            col,
            logger,
        )


def plot_metric(metric, title, grouped_df, output_file, col, logger):
    plt.figure(figsize=(10, 6))
    for index, model in enumerate(grouped_df["Model"].unique()):
        model_data = grouped_df[grouped_df["Model"] == model]
        plt.plot(
            model_data[f"{col}_bin"].astype(str),
            model_data[metric],
            marker="o",
            label=model,
            color=utils.colors[index],
        )
    plt.xticks(rotation=45)
    plt.xlabel(f"{col} Bins")
    plt.ylabel(metric.capitalize())
    plt.ylim(0, 1.1)
    plt.legend(title="Model")
    plt.tight_layout()
    utils.create_file_with_directories(output_file, logger)
    plt.savefig(output_file)
    plt.close()


def plot_metric_avg(metric, title, grouped_df, output_file, col, logger):
    plt.figure(figsize=(10, 6))
    model_data = grouped_df
    plt.plot(
        model_data[f"{col}_bin"].astype(str),
        model_data[metric],
        marker="o",
        color=utils.colors[1],
    )
    plt.xticks(rotation=45)
    plt.xlabel(f"{col} Bins")
    plt.ylabel(metric.capitalize())
    plt.ylim(0, 1.1)
    # plt.legend(title="Model")
    plt.tight_layout()
    utils.create_file_with_directories(output_file, logger)
    plt.savefig(output_file)
    plt.close()
