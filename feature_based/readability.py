# !pip install ReadabilityCalculator
# !pip install textstat


import textstat
from readcalc import readcalc
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
import re
from tqdm import tqdm
import multiprocessing as mp


from textblob import TextBlob
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import *
import string


def compute_readability_for_content(content, index):
    """Computes readability measures for a chunk of the dataframe."""
    results = []
    # for ind, row in chunk.iterrows():
    calc = readcalc.ReadCalc(content, preprocesshtml=None)
    results.append(
        {
            "index": index,
            "smog_index": calc.get_smog_index(),
            "flesch_reading_ease": calc.get_flesch_reading_ease(),
            "flesch_kincaid_grade_level": calc.get_flesch_kincaid_grade_level(),
            "coleman_liau_index": calc.get_coleman_liau_index(),
            "gunning_fog_index": calc.get_gunning_fog_index(),
            "ari_index": calc.get_ari_index(),
            "lix_index": calc.get_lix_index(),
            "dale_chall_score": calc.get_dale_chall_score(),
            "dale_chall_known_fraction": calc.get_dale_chall_known_fraction(),
        }
    )
    return results


def compute_readability(df, col):
    """Computes the readability measures of text in parallel."""
    num_workers = mp.cpu_count()
    df_split = np.array_split(df, 1)
    results = []
    entries = [(row["content"], i) for i, row in df.iterrows()]
    with mp.Pool(num_workers) as pool:
        results += pool.starmap(
            compute_readability_for_content,
            tqdm(entries, total=len(df)),
        )

    # Combine results into the original dataframe
    for result in results:
        for res in result:
            ind = res.pop("index")
            for key, value in res.items():
                df.at[ind, key] = value

    return df


def compute_readability_old(df, col):
    """computes the readability measures of text
    input:
    df = inpute dataframe
    col = column of inpute dataframe for which the readability scores will be calculated
    """

    for ind, row in tqdm(df.iterrows(), total=len(df)):
        calc = readcalc.ReadCalc(row[col], preprocesshtml=None)
        df.loc[ind, "smog_index"] = calc.get_smog_index()
        df.loc[ind, "flesch_reading_ease"] = calc.get_flesch_reading_ease()
        df.loc[ind, "flesch_kincaid_grade_level"] = (
            calc.get_flesch_kincaid_grade_level()
        )
        df.loc[ind, "coleman_liau_index"] = calc.get_coleman_liau_index()
        df.loc[ind, "gunning_fog_index"] = calc.get_gunning_fog_index()
        df.loc[ind, "ari_index"] = calc.get_ari_index()
        df.loc[ind, "lix_index"] = calc.get_lix_index()
        df.loc[ind, "dale_chall_score"] = calc.get_dale_chall_score()
        df.loc[ind, "dale_chall_known_fraction"] = calc.get_dale_chall_known_fraction()
    return df


def compute_syntactic_old(df, col):
    num_workers = mp.cpu_count()

    for ind, row in df.iterrows():
        df.loc[ind, "syllable_count"] = textstat.syllable_count(str(row[col]))
        # df.loc[ind,'lexicon_count'] = textstat.lexicon_count(str(row[col]), removepunct=True)
        df.loc[ind, "sentence_count"] = textstat.sentence_count(str(row[col]))
    return df


def compute_syntactic(df, col):
    num_workers = mp.cpu_count()
    results = []
    entries = [(row["content"], i) for i, row in df.iterrows()]
    with mp.Pool(num_workers) as pool:
        results += pool.starmap(
            compute_syntactic_for_content,
            tqdm(entries, total=len(df)),
        )

    # Combine results into the original dataframe
    for result in results:
        for res in result:
            ind = res.pop("index")
            for key, value in res.items():
                df.at[ind, key] = value

    return df


def compute_syntactic_for_content(content, index):
    """Computes readability measures for a chunk of the dataframe."""
    results = []
    # for ind, row in chunk.iterrows():
    results.append(
        {
            "index": index,
            "syllable_count": textstat.syllable_count(content),
            "sentence_count": textstat.sentence_count(content),
        }
    )
    return results
