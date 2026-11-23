"""区分度指标。标签或特征无法形成两类时返回 None，不用 0 填充。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score


def _binary_mask(score: pd.Series, label: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"score": score, "label": label})
    frame = frame.dropna()
    frame = frame[frame["label"].isin([0, 1])]
    return frame


def roc_auc(score: pd.Series, label: pd.Series) -> float | None:
    frame = _binary_mask(score, label)
    if frame.empty or frame["label"].nunique() < 2:
        return None
    if frame["score"].nunique() < 2:
        return None
    try:
        value = float(roc_auc_score(frame["label"], frame["score"]))
    except ValueError:
        return None
    if math.isnan(value):
        return None
    return value


def ks_statistic(score: pd.Series, label: pd.Series) -> float | None:
    frame = _binary_mask(score, label)
    positive = frame.loc[frame["label"] == 1, "score"]
    negative = frame.loc[frame["label"] == 0, "score"]
    if len(positive) == 0 or len(negative) == 0:
        return None
    statistic = ks_2samp(positive, negative, method="auto").statistic
    if statistic is None or math.isnan(float(statistic)):
        return None
    return float(statistic)


def information_value(feature: pd.Series, label: pd.Series, min_bin_count: int) -> float | None:
    frame = pd.DataFrame({"x": feature, "y": label}).dropna()
    frame = frame[frame["y"].isin([0, 1])]
    if frame.empty or frame["y"].nunique() < 2 or frame["x"].nunique() < 1:
        return None

    counts = frame["x"].value_counts()
    rare = counts[counts < min_bin_count].index
    grouped = frame["x"].where(~frame["x"].isin(rare), other="__other__")
    total_pos = int((frame["y"] == 1).sum())
    total_neg = int((frame["y"] == 0).sum())
    if total_pos == 0 or total_neg == 0:
        return None

    work = pd.DataFrame({"x": grouped, "y": frame["y"].to_numpy()})
    groups = list(work.groupby("x", sort=False))
    bin_count = len(groups)
    iv = 0.0
    for _, part in groups:
        pos = float((part["y"] == 1).sum())
        neg = float((part["y"] == 0).sum())
        pos_dist = (pos + 0.5) / (total_pos + 0.5 * bin_count)
        neg_dist = (neg + 0.5) / (total_neg + 0.5 * bin_count)
        woe = math.log(pos_dist / neg_dist)
        iv += (pos_dist - neg_dist) * woe
    return float(iv)


def mutual_info(feature: pd.Series, label: pd.Series) -> float | None:
    frame = pd.DataFrame({"x": feature, "y": label}).dropna()
    frame = frame[frame["y"].isin([0, 1])]
    if frame.empty or frame["y"].nunique() < 2 or frame["x"].nunique() < 1:
        return None
    codes = pd.Categorical(frame["x"]).codes.reshape(-1, 1)
    value = mutual_info_classif(
        codes,
        frame["y"].astype(int),
        discrete_features=True,
        random_state=0,
    )[0]
    if value is None or math.isnan(float(value)):
        return None
    return float(value)
