import pandas as pd
import numpy as np
from scipy import stats
from numpy.random import seed

# seed the random number generator
seed(1)

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import *
import string
from tqdm import tqdm
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")


import sys


def classify(df_final, final_features, feature_selection, k, clf):
    print(df_final.head())
    y = np.array([int(i) for i in df_final["label"].values])


    # feature selection
    # if feature_selection:
    #     if type(final_features) != list:
    #         final_features = final_features.to_list()
    #     X, selected_features = statistical_tests.stat_sig_test(
    #         df_final[final_features + ["label"]], k
    #     )

    # else:
    nans = df_final.columns[df_final.isnull().any()].tolist()
    final_features = [feature for feature in final_features if feature not in nans]
    X = df_final[final_features].values
    selected_features = final_features
    if X.shape[1] == 0:
        return (
            0,
            0,
            0,
        )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    Avg_precision_list = []
    AUROC_list = []

    print(y)
    for tr_ind, tst_ind in skf.split(X, y):
        X_train = X[tr_ind]
        X_test = X[tst_ind]
        y_train = y[tr_ind]
        y_test = y[tst_ind]
        print(X_train)
        print(y_train)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        try:
            proba = clf.predict_proba(X_test)
        except:
            proba = clf.decision_function(X_test)
        # print("finish predicting")

        accuracy = accuracy_score(y_pred, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_pred, y_test)
        try:
            AP = average_precision_score(y_test, proba[:, 1])
            AUROC = roc_auc_score(y_test, proba[:, 1])
        except:
            AP = average_precision_score(y_test, proba)
            AUROC = roc_auc_score(y_test, proba)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy)
        Avg_precision_list.append(AP)
        AUROC_list.append(AUROC)

    return (
        np.round(np.mean(AUROC_list), 3),
        # np.round(np.mean(f1_list),3),
        np.round(np.mean(Avg_precision_list), 3),
        # np.round(np.mean(accuracy_list),3),
        selected_features,
    )


def format_df(df, title_or_text):
    """
    function to convert label to make it uniform accross the datasets
    """
    df.label = df.label.apply(lambda x: 1 if x in ["fake", "Fake", "1", 1] else 0)
    new_names = [(i, title_or_text + "_" + i) for i in df.iloc[:, 3:].columns.values]
    df.rename(columns=dict(new_names), inplace=True)
    return df


def merge_title_text(df_title, df_text):
    """
    Merge all title and text features to form single dataframe
    """
    df_final = pd.merge(
        df_title, df_text, on=["news_id", "label"], how="left", suffixes=["", "_y"]
    )
    df_final.drop([x for x in df_final if x.endswith("_y")], axis=1, inplace=True)
    return df_final


def get_features(df_final):
    """
    function to get only those features used in paper of Horne and Adali plus emotion features
    """

    features_used_in_paper = [
        # "Analytic",
        "insight",
        "cause",
        "discrep",
        "tentat",
        "certain",
        "differ",
        "affiliation",
        "power",
        "reward",
        "risk",
        "work",
        "leisure",
        "money",
        "relig",
        # "Tone",
        "affect",
        "WC",
        "WPS",
        "num_nouns",
        "num_propernouns",
        "num_personalnouns",
        "num_ppssessivenouns",
        "num_whpronoun",
        "num_determinants",
        "num_whdeterminants",
        "num_cnum",
        "num_adverb",
        "num_interjections",
        "num_verb",
        "num_adj",
        "num_vbd",
        "num_vbg",
        "num_vbn",
        "num_vbp",
        "num_vbz",
        "focuspast",
        "focusfuture",
        "i",
        "we",
        "you",
        "shehe",
        "quant",
        "compare",
        # "Exclam",
        "negate",
        "swear",
        "netspeak",
        "interrog",
        "count_uppercased",
        "percentage_stopwords",
        # "AllPunc",
        # "Quote",
        "lexical_diversity",
        "wlen",
        "gunning_fog_index",
        "smog_index",
        "flesch_kincaid_grade_level",
    ]

    features_used_in_paper_ = [x for x in features_used_in_paper]

    emotion_features = [
        x
        for x in [
            "Anger",
            "Anticipation",
            "Disgust",
            "Fear",
            "Joy",
            "Sadness",
            "Surprise",
            "Trust",
            "neg",
            "pos",
            "posemo",
            "negemo",
            "anx",
        ]
    ]

    return features_used_in_paper_ + emotion_features


def classification_result(df, text, title, feature_selection):
    """
    function to automate the classification using multiple classifiers

    df: input dataframe
    text: news body features to be considered if not empty
    title: news title features to be considered if not empty
    feature_selection: if True selects statistically significant (pvalue < 0.05) features up to sqrt of training set
    """
    drop_features = [
        "lexicon_count",
        "neu",
        "compound",
        "adverb",
        "verb",
        "adj",
        "Objective",
        "anger",
        "sad",
    ]
    df.drop(
        columns=[x for x in drop_features if x in list(df.columns)],
        inplace=True,
    )
    df = df.sample(frac=0.01)

    result = pd.DataFrame()
    clfs = [
        RandomForestClassifier(class_weight="balanced", random_state=0),
    ]
    i = 0
    for n in tqdm([int(np.sqrt(df.shape[0] * 0.8))]):
        for clf in clfs:
            if text == [] and title == []:
                text_features = get_features(df)
                result.at[i, "classifier"] = str(clf).split("(")[0]
                result.at[i, "features"] = "content"
                print(text_features, feature_selection, n ,clf)
                result.at[i, "AUROC"], result.at[i, "AvgP"], selected_text_features = (
                    classify(df, text_features, feature_selection, n, clf)
                )
            else:
                result.at[i, "classifier"] = str(clf).split("(")[0]
                result.at[i, "features"] = "content"
                result.at[i, "AUROC"], result.at[i, "AvgP"], selected_text_features = (
                    classify(df, text, feature_selection, n, clf)
                )

                result.at[i + 1, "classifier"] = str(clf).split("(")[0]
                result.at[i + 1, "features"] = "title"
                (
                    result.at[i + 1, "AUROC"],
                    result.at[i + 1, "AvgP"],
                    selected_title_features,
                ) = classify(df, title, feature_selection, n, clf)
            i += 2
    return result, selected_text_features


def classification_result_old(df, text, title, feature_selection):
    """
    function to automate the classification using multiple classifiers

    df: input dataframe
    text: news body features to be considered if not empty
    title: news title features to be considered if not empty
    feature_selection: if True selects statistically significant (pvalue < 0.05) features up to sqrt of training set
    """
    drop_features = [
        "lexicon_count",
        "neu",
        "compound",
        "adverb",
        "verb",
        "adj",
        "Objective",
        "anger",
        "sad",
    ]
    df.drop(
        columns=[x for x in drop_features if x in list(df.columns)],
        inplace=True,
    )
    df = df.sample(frac=0.01)

    result = pd.DataFrame()
    clfs = [
        svm.LinearSVC(class_weight="balanced", random_state=0),
        LogisticRegression(class_weight="balanced", random_state=0),
        RandomForestClassifier(class_weight="balanced", random_state=0),
    ]
    i = 0
    for n in tqdm([int(np.sqrt(df.shape[0] * 0.8))]):
        for clf in clfs:
            if text == [] and title == []:
                text_features = get_features(df, "text")
                result.at[i, "classifier"] = str(clf).split("(")[0]
                result.at[i, "features"] = "text"
                result.at[i, "AUROC"], result.at[i, "AvgP"], selected_text_features = (
                    classify(df, text_features, feature_selection, n, clf)
                )
            else:
                result.at[i, "classifier"] = str(clf).split("(")[0]
                result.at[i, "features"] = "text"
                result.at[i, "AUROC"], result.at[i, "AvgP"], selected_text_features = (
                    classify(df, text, feature_selection, n, clf)
                )

                result.at[i + 1, "classifier"] = str(clf).split("(")[0]
                result.at[i + 1, "features"] = "title"
                (
                    result.at[i + 1, "AUROC"],
                    result.at[i + 1, "AvgP"],
                    selected_title_features,
                ) = classify(df, title, feature_selection, n, clf)
            i += 2
    return result, selected_text_features


def groupwise_features(selected_features, title_or_text, group):
    """
    function to get features within each group
    """
    complexity_features = [
        "lexical_diversity",
        "wlen",
        "gunning_fog_index",
        "smog_index",
        "flesch_kincaid_grade_level",
    ]

    psychology_features = [
        "Analytic",
        "insight",
        "cause",
        "discrep",
        "tentat",
        "certain",
        "differ",
        "affiliation",
        "power",
        "reward",
        "risk",
        "work",
        "leisure",
        "money",
        "relig",
        "Tone",
        "affect",
        "Anger",
        "Anticipation",
        "Disgust",
        "Fear",
        "Joy",
        "Sadness",
        "Surprise",
        "Trust",
        "neg",
        "pos",
        "posemo",
        "negemo",
        "anx",
    ]

    stylistic_features = [
        "WC",
        "WPS",
        "num_nouns",
        "num_propernouns",
        "num_personalnouns",
        "num_ppssessivenouns",
        "num_whpronoun",
        "num_determinants",
        "num_whdeterminants",
        "num_cnum",
        "num_adverb",
        "num_interjections",
        "num_verb",
        "num_adj",
        "num_vbd",
        "num_vbg",
        "num_vbn",
        "num_vbp",
        "num_vbz",
        "focuspast",
        "focusfuture",
        "i",
        "we",
        "you",
        "shehe",
        "quant",
        "compare",
        "Exclam",
        "negate",
        "swear",
        "netspeak",
        "interrog",
        "count_uppercased",
        "percentage_stopwords",
        "AllPunc",
        "Quote",
    ]

    selected_features_ = [x for x in selected_features]
    if group == "complexity":
        return [
            x
            for x in selected_features_
            if x.strip(title_or_text + "_") in complexity_features
        ]
    elif group == "stylistic":
        return [
            x
            for x in selected_features_
            if x.strip(title_or_text + "_") in stylistic_features
        ]
    elif group == "psychology":
        return [
            x
            for x in selected_features_
            if x.strip(title_or_text + "_") in psychology_features
        ]
