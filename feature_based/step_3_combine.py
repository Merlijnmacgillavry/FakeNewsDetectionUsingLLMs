import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords

stops = stopwords.words("english")
nltk.download("averaged_perceptron_tagger")
from nltk.tag import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize
import re, os

import pandas as pd
import numpy as np
from collections import Counter
import string


def merge_liwc_and_remaining_features(filename_remaining, filename_liwc):
    """
    function to merge LIWC features and all other features.

    path: path to the folder where Generated_features folder is saved
    filename_remaining: name of pickle file that contains all other features except LIWC
    filename_liwc: name of file that contains LIWC features

    """
    # path = '/content/gdrive/My Drive/ECIR 2021 Reproducibility/Data/'
    df_remaining_features = pd.read_pickle(filename_remaining)
    df_liwc = pd.read_csv(filename_liwc)
    print(df_remaining_features.shape[0])
    print(df_liwc.shape[0])
    # return
    # try:
    #     # format liwc dataframe and remove unwanted columns
    #     df_liwc["news_id"] = df_liwc.Filename.apply(
    #         lambda x: x.strip(".txt") if x.endswith(".txt") else x
    #     )
    #     df_liwc_ = df_liwc.drop(columns=["Segment", "Filename"])
    # except:
    #     df_liwc_ = df_liwc.drop(columns=["label"])

    # # remove duplicate rows if any
    # df_remaining_features = df_remaining_features.drop_duplicates(
    #     subset="news_id", keep="last"
    # )

    # merge both dfs by news_id
    df_merged_features = df_remaining_features.merge(df_liwc, on="content", how="inner")

    # assert if merged dataframe have required number of rows and columns
    # assert df_merged_features.shape[1] == (
    #     df_remaining_features.shape[1] + df_liwc.shape[1] - 1
    # )
    # assert (
    #     df_merged_features.shape[0]
    #     == df_remaining_features.shape[0]
    #     == df_liwc.shape[0]
    # )

    print(list(df_merged_features.columns))
    print(df_merged_features.shape[0])
    print("Merged and saved all features in file")
    return df_merged_features
    df_merged_features.to_pickle("./data/out/baselines/combined/FakeNewsNet/out.pkl")
