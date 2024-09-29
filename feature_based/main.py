import pandas as pd
import step_1_liwc_features
from utils import create_file_with_directories
import step_2_extract_complexity_emotion_and_stylistic_features
import step_3_combine
import step_5_classification

# First we extract the content of the preprocessed dataset and extract the liwc features


if __name__ == "__main__":
    # Step 1 LIWC
    # df = pd.read_csv("./data/out/preprocess/FakeNewsNet/out.csv")
    # print(df.head())
    # df = step_1_liwc_features.compute_liwc_from_dict(df, "content")
    # output_path_liwc = "./data/out/baselines/FeatureBased/liwc/FakeNewsNet/out.csv"
    # create_file_with_directories(output_path)
    # df.to_csv(output_path, mode="a", index_label="index")

    # Step 2 complexity, emotion, stylistic features
    # df_no_liwc = step_2_extract_complexity_emotion_and_stylistic_features.extract_complexity_emotion_stylistic_features(
    #     "./data/out/preprocess/FakeNewsNet/out.csv"
    # )
    # output_path_no_liwc = (
    #     "./data/out/baselines/FeatureBased/no_liwc/FakeNewsNet/out.pkl"
    # )

    # create_file_with_directories(output_path_no_liwc)
    # df_no_liwc.to_pickle(output_path_no_liwc)
    # Step 3 combine
    # output_path_combined = (
    #     "./data/out/baselines/FeatureBased/combined/FakeNewsNet/out.pkl"
    # )
    # create_file_with_directories(output_path_combined)
    df_merged = step_3_combine.merge_liwc_and_remaining_features(
        "./data/out/baselines/FeatureBased/no_liwc/FakeNewsNet/out.pkl",
        "./data/out/baselines/FeatureBased/liwc/FakeNewsNet/out.csv",
    )
    # df_merged.to_pickle(output_path_combined)
    # Step 5 classification
    df_text_fakeNews = pd.read_pickle(
        "./data/out/baselines/FeatureBased/combined/FakeNewsNet/out.pkl"
    )
    df_text_fakeNews_formatted = step_5_classification.format_df(
        df_text_fakeNews, "text"
    )
    result_p, selected_text_feat_p = step_5_classification.classification_result(
        df_text_fakeNews, [], [], True
    )
    print(result_p)
