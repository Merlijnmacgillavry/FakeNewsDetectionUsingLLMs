# -*- coding: utf-8 -*-
"""
authors: "Ashlee Milton and Anu Shrestha"
"""

import nltk
import multiprocessing as mp

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.lang.en import STOP_WORDS
from nltk.corpus import stopwords

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

import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# function that get the emotion itensity
def getEmotionItensity(word, emotion, emotion_dic):
    key = word + "#" + emotion
    try:
        return emotion_dic[key]
    except:
        return 0.0


# Check if the word is in the Lexicon
def isWordInEmotionFile(word, emotion_dic):
    result = [(key) for key in emotion_dic.keys() if key.startswith(word + "#")]
    if len(result) == 0:
        return False
    else:
        return True


# Stopping checker
def isStopWord(word):
    if word in stops:
        return True
    else:
        return False


# Assign the emotion itensity to the dictionary
def calculateEmotion(emotions, word, emotion_dic):
    emotions["Anger"] += getEmotionItensity(word, "anger", emotion_dic)
    emotions["Anticipation"] += getEmotionItensity(word, "anticipation", emotion_dic)
    emotions["Disgust"] += getEmotionItensity(word, "disgust", emotion_dic)
    emotions["Fear"] += getEmotionItensity(word, "fear", emotion_dic)
    emotions["Joy"] += getEmotionItensity(word, "joy", emotion_dic)
    emotions["Sadness"] += getEmotionItensity(word, "sadness", emotion_dic)
    emotions["Surprise"] += getEmotionItensity(word, "surprise", emotion_dic)
    emotions["Trust"] += getEmotionItensity(word, "trust", emotion_dic)


# get the emotion vector of a given text
def getEmotionVector(text, emotion_dic):
    # create the initial emotions
    emotions = {
        "Anger": 0.0,
        "Anticipation": 0.0,
        "Disgust": 0.0,
        "Fear": 0.0,
        "Joy": 0.0,
        "Sadness": 0.0,
        "Surprise": 0.0,
        "Trust": 0.0,
        "Objective": 0.0,
    }
    # parse the description
    str_ = re.sub("[^a-zA-Z]+", " ", str(text))
    pat = re.compile(r"[^a-zA-Z ]+")
    str_ = re.sub(pat, "", str_).lower()
    # split string
    words = str_.split()

    # iterate over words array
    for word in words:
        if not isStopWord(word):
            # first check if the word is in its natural form
            if isWordInEmotionFile(word, emotion_dic):
                calculateEmotion(emotions, word, emotion_dic)
            elif isWordInEmotionFile(lmtzr.lemmatize(word), emotion_dic):
                calculateEmotion(emotions, lmtzr.lemmatize(word), emotion_dic)
            elif isWordInEmotionFile(lmtzr.lemmatize(word, "v"), emotion_dic):
                calculateEmotion(emotions, lmtzr.lemmatize(word, "v"), emotion_dic)
            else:
                emotions["Objective"] += 1
    total = sum(emotions.values())
    for key in sorted(emotions.keys()):
        try:
            emotions[key] = (1.0 / total) * emotions[key]
        except:
            emotions[key] = 0
    return emotions


def NRC_dict(fileEmotion):
    emotion_df = pd.read_csv(
        fileEmotion, names=["word", "emotion", "itensity"], sep="\t"
    )

    # create the dictionary with the word/emotion/score
    emotion_dic = dict()
    for index, row in emotion_df.iterrows():
        # add first as it is given in the lexicon
        temp_key = row["word"] + "#" + row["emotion"]
        emotion_dic[temp_key] = row["itensity"]
        # add in the normal noun form
        temp_key_n = lmtzr.lemmatize(row["word"]) + "#" + row["emotion"]
        emotion_dic[temp_key_n] = row["itensity"]
        # add in the normal verb form
        temp_key_v = lmtzr.lemmatize(row["word"], "v") + "#" + row["emotion"]
        emotion_dic[temp_key_v] = row["itensity"]
    return emotion_dic


def emotion_NRC_old(df, emo_dic_path, title_or_text):
    emotion_dic = NRC_dict(emo_dic_path)
    results_emo = pd.DataFrame()
    for x, row in tqdm(df.iterrows(), total=len(df)):
        extracted_emotion = getEmotionVector(row[title_or_text], emotion_dic)
        temp = pd.DataFrame(extracted_emotion, index=[x])
        results_emo = results_emo._append(temp)
    df_emo = pd.concat([df, results_emo], axis=1)
    return df_emo


def emotion_NRC(df, emo_dic_path, col):
    num_workers = mp.cpu_count()
    emotion_dic = NRC_dict(emo_dic_path)
    results = []
    entries = [(row[col], i, emotion_dic) for i, row in df.iterrows()]
    with mp.Pool(num_workers) as pool:
        results += pool.starmap(
            emotion_NRC_for_content,
            tqdm(entries, total=len(df)),
        )

    # Combine results into the original dataframe
    for result in results:
        for res in result:
            ind = res.pop("index")
            for key, value in res.items():
                df.at[ind, key] = value
    # return df
    # emotion_dic = NRC_dict(emo_dic_path)
    # results_emo = pd.DataFrame()
    # for x, row in tqdm(df.iterrows()):
    #     extracted_emotion = getEmotionVector(row[col], emotion_dic)
    #     temp = pd.DataFrame(extracted_emotion, index=[x])
    #     results_emo = results_emo._append(temp)

    # df_emo = pd.concat([df, results_emo], axis=1)
    # return df_emo
    return df


def emotion_NRC_for_content(content, index, emotion_dic):
    extracted_emotion = getEmotionVector(content, emotion_dic)
    results = []
    results.append({"index": index, **extracted_emotion})
    # print(results)
    return results


analyzer = SentimentIntensityAnalyzer()


def sentiment_strength_vader(df, title_or_text):
    sentiments_list = []

    for x, row in tqdm(df.iterrows(), total=len(df)):
        sentence_list = sent_tokenize(str(row[title_or_text]))
        sentiments = {"compound": 0.0, "neg": 0.0, "neu": 0.0, "pos": 0.0}

        for sentence in sentence_list:
            vs = analyzer.polarity_scores(str(sentence))
            sentiments["compound"] += vs["compound"]
            sentiments["neg"] += vs["neg"]
            sentiments["neu"] += vs["neu"]
            sentiments["pos"] += vs["pos"]

        sentiments["compound"] = sentiments["compound"] / len(sentence_list)
        sentiments["neg"] = sentiments["neg"] / len(sentence_list)
        sentiments["neu"] = sentiments["neu"] / len(sentence_list)
        sentiments["pos"] = sentiments["pos"] / len(sentence_list)

        sentiments_list.append(sentiments)
    df_vader = pd.concat([df, pd.DataFrame(sentiments_list)], axis=1)
    return df_vader
