import textstat
from readcalc import readcalc
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.lang.en import STOP_WORDS
from nltk.corpus import stopwords
from nltk import sent_tokenize

stops = list(
    set(
        stopwords.words("english")
        + list(set(ENGLISH_STOP_WORDS))
        + list(set(STOP_WORDS))
        + ["http"]
    )
)
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import re
import string
from nltk.tag import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize
import liwc
import os

from textblob import TextBlob
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emotion_features
import readability
import stylistic_features


def extract_complexity_emotion_stylistic_features(file):
    """
    Extract all the features used in paper for text in input filename.
    """
    df = pd.read_csv(file)
    # df = df.sample(frac=0.01)
    # Extract complexity features
    df = readability.compute_readability(df, "content")
    df = readability.compute_syntactic(df, "content")
    df["lexical_diversity"] = df["content"].apply(stylistic_features.lexical_diversity)
    df["wlen"] = df["content"].apply(stylistic_features.average_word_length)

    # Extract stylistic features
    df = stylistic_features.part_of_speech(df, "content")
    df = stylistic_features.numeric_features(df, "content")

    # Extract emotion features
    emo_dic_path = "./baselines/feature_based/emotion_itensity.txt"  # path to the emotion intensity lexicon dictionary
    df = emotion_features.emotion_NRC(df, emo_dic_path, "content")
    df = emotion_features.sentiment_strength_vader(df, "content")
    print(df.head())
    print("Saving extracted features completed!")
    return df
