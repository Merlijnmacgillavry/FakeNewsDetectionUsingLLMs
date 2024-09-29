from collections import Counter
import nltk
import os
from tqdm import tqdm

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
from nltk.corpus import stopwords

stops = stopwords.words("english")
from nltk.tag import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize
import re
import liwc
from tqdm.contrib.concurrent import process_map  # or thread_map

from textblob import TextBlob
import pandas as pd
import numpy as np
import string
from tqdm import tqdm


# In case if the LIWC features are to be computed using dictionary use following code as example to extract required features
def tokenize(text):
    """
    tokenizer to tokenize input text
    """
    for match in re.finditer(r"\w+", text, re.UNICODE):
        yield match.group(0)


def compute_liwc_from_dict(df, col):
    """
    Extract LIWC features from dictionary
    """
    parse, category_names = liwc.load_token_parser(
        "./baselines/LIWC2015_English.dic"
    )  # path of LIWC dictionary

    frames = []
    for i in tqdm(range(0, len(df[col]))):
        text = df[col][i]
        text_tokens = tokenize(text)
        text_counts = Counter(
            category for token in text_tokens for category in parse(token)
        )

        liwc_value_dic = {}
        for k, v in text_counts.items():
            liwc_value_dic["content"] = text
            word_count = len([word for word in text.split(" ")])
            liwc_value_dic["WC"] = word_count
            liwc_value_dic["WPS"] = sum(
                [len(sent.split(" ")) for sent in sent_tokenize(text)]
            ) / len(sent_tokenize(text))
            liwc_value_dic[k.split(",")[0].split(" ")[0]] = (v / word_count) * 100
        frames.append(pd.DataFrame([liwc_value_dic]))
    df_liwc = pd.concat(frames)
    print(df_liwc.head())
    return df_liwc
