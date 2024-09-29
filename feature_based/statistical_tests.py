import pandas as pd
import numpy as np
from scipy import stats


def one_way_anova(data1, data2):
    F_value, P_value = stats.f_oneway(data1, data2)
    return F_value, P_value


def wilcoxon(data1, data2):
    """H0: the distributions of both samples are equal.
    H1: the distributions of both samples are not equal."""
    F_value, P_value = stats.ranksums(data1, data2)
    return F_value, P_value


def normality_test(df, col, data1, data2):
    # normality test
    stat1, p1 = stats.normaltest(data1)
    stat2, p2 = stats.normaltest(data2)

    normality = 1
    # interpret
    alpha = 0.05
    if (p1 < alpha) & (p2 < alpha):
        normality = 1
        # print('Sample looks Gaussian (fail to reject H0)')
        F_value, P_value = one_way_anova(data1, data2)
    else:
        normality = 0
        # print('Sample does not look Gaussian (reject H0)')
        F_value, P_value = wilcoxon(data1, data2)
    return normality, F_value, P_value


def stat_sig_test(df, topn):
    feature, normal, f_value, p_value = [], [], [], []
    for col in df.columns[:-1]:
        df.label = df.label.astype(int)
        df_real = df[df.label == 0]  # real
        df_fake = df[df.label == 1]  # fake
        # print(df_real.shape, df_fake.shape)
        # try:
        normality, F_value, P_value = normality_test(
            df, col, df_real[col], df_fake[col]
        )
        # except Exception as e:
        #   print(e)
        feature.append(col)
        normal.append(normality)
        f_value.append(F_value)
        p_value.append(P_value)

    stat_test = pd.DataFrame(
        {
            "feature": feature,
            "normality": normal,
            "F_value": f_value,
            "P_value": p_value,
        }
    )
    stat_significance_features = stat_test[stat_test["P_value"] < 0.05]["feature"]

    # print(topn, len(stat_significance_features))
    if topn < len(stat_significance_features):
        selected_features = stat_test.sort_values(by="F_value", ascending=False)[
            "feature"
        ][:topn]
    else:
        selected_features = stat_significance_features
    # print(selected_features)
    return df[selected_features].values, selected_features


## statistical T-test
from scipy.stats import ttest_ind


def t_test(df):
    df.label = df.label.apply(lambda x: 1 if x in ["fake", "Fake", "1", 1] else 0)
    real_news = df[df.label == 0]
    fake_news = df[df.label == 1]

    print(real_news.shape, fake_news.shape)

    selected_features = [
        col
        for col in df.columns
        if col not in ["label", "news_id", "news_title", "news_text"]
    ]
    frames = []
    for feature in selected_features:
        t_stat, p_value = ttest_ind(real_news[feature], fake_news[feature])
        frames.append([feature, t_stat, p_value])
    t_test_result = pd.DataFrame(frames, columns=["feature", "statistics", "Pvalue"])
    t_test_result.sort_values(by="Pvalue", ascending=True, inplace=True)
    statistical_sig_result = t_test_result[t_test_result["Pvalue"] < 0.05]
    statistical_sig_result["feature_differ"] = statistical_sig_result.apply(
        lambda x: "Real > Fake" if x["statistics"] > 0 else "Fake > Real", axis=1
    )
    return statistical_sig_result
