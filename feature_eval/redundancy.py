"""同类型特征之间的相关。数值用 Spearman，类别用 Cramér's V。"""

from __future__ import annotations

import math
import warnings

import pandas as pd
from scipy.stats import chi2_contingency


def _cramers_v(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.DataFrame({"a": left, "b": right}).dropna()
    if frame.empty:
        return None
    table = pd.crosstab(frame["a"], frame["b"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2 = chi2_contingency(table, correction=False)[0]
    total = float(table.to_numpy().sum())
    denominator = total * min(table.shape[0] - 1, table.shape[1] - 1)
    if denominator <= 0 or chi2 < 0:
        return None
    return float(math.sqrt(chi2 / denominator))


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    value = left.corr(right, method="spearman")
    if value is None or pd.isna(value):
        return None
    return abs(float(value))


def pairwise_associations(rows: list[dict], config) -> list[dict]:
    report_at = config.thresholds["redundancy_report"]
    grouped = {"numeric": [], "categorical": []}
    for row in rows:
        if row.get("absent") or row.get("_values") is None:
            continue
        grouped.setdefault(row["feature_type"], []).append(row)

    pairs = []
    for kind, members in grouped.items():
        method = "spearman" if kind == "numeric" else "cramers_v"
        measure = _spearman if kind == "numeric" else _cramers_v
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                coefficient = measure(left["_values"], right["_values"])
                if coefficient is None or coefficient < report_at:
                    continue
                feature_a, feature_b = sorted((left["feature_name"], right["feature_name"]))
                pairs.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "method": method,
                        "coefficient": coefficient,
                        "kept_feature": "",
                        "dropped_feature": "",
                    }
                )
    pairs.sort(key=lambda item: (-item["coefficient"], item["feature_a"], item["feature_b"]))
    return pairs
