"""按特征计算质量指标和区分度。决策用的 AUC 在剔除哨兵和越界值之后计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_eval.decision import is_number
from feature_eval.signal import information_value, ks_statistic, mutual_info, roc_auc


def _empty_row(name: str, feature_type: str, description: str, sample_count: int) -> dict:
    return {
        "feature_name": name,
        "feature_type": feature_type,
        "description": description,
        "sample_count": sample_count,
        "non_null_count": 0,
        "missing_rate": None,
        "unique_count": None,
        "mean": None,
        "std": None,
        "min": None,
        "p01": None,
        "p25": None,
        "p50": None,
        "p75": None,
        "p99": None,
        "max": None,
        "skew": None,
        "out_of_range_rate": None,
        "sentinel_rate": None,
        "left_censored": False,
        "censor_ratio": None,
        "top1_value": None,
        "top1_ratio": None,
        "top5_coverage": None,
        "auc": None,
        "auc_stats": None,
        "auc_effective": None,
        "auc_gap": None,
        "direction": None,
        "ks": None,
        "iv": None,
        "mutual_info": None,
        "partner_feature": "",
        "partner_corr": None,
        "absent": False,
        "binary_scored": False,
        "range_configured": False,
        "hits": [],
        "_values": None,
        "_auc_unfiltered": None,
        "_missing_indicator_auc": None,
    }


def _is_sentinel(values: pd.Series, sentinels: list[float]) -> pd.Series:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.zeros(len(array), dtype=bool)
    finite = np.isfinite(array)
    for sentinel in sentinels:
        mask |= finite & np.isclose(array, sentinel, rtol=0, atol=1e-4)
    return pd.Series(mask, index=values.index)


def _attach_rank_metrics(row: dict, score: pd.Series, label: pd.Series) -> None:
    auc = roc_auc(score, label)
    row["auc"] = auc
    row["ks"] = ks_statistic(score, label)
    if auc is None:
        row["auc_effective"] = None
        row["direction"] = None
        return
    row["auc_effective"] = max(auc, 1.0 - auc)
    row["direction"] = "positive" if auc >= 0.5 else "negative"


def _distribution(values: pd.Series) -> dict:
    if values.empty:
        return {}
    quantiles = values.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    skew = values.skew() if len(values) >= 3 else np.nan
    std = values.std(ddof=1) if len(values) >= 2 else np.nan
    return {
        "mean": float(values.mean()),
        "std": None if pd.isna(std) else float(std),
        "min": float(values.min()),
        "p01": float(quantiles.loc[0.01]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(values.max()),
        "skew": None if pd.isna(skew) else float(skew),
    }


def _category_stats(values: pd.Series) -> dict:
    present = values.dropna()
    if present.empty:
        return {"unique_count": 0, "top1_value": None, "top1_ratio": None, "top5_coverage": None}
    counts = present.value_counts(normalize=True)
    return {
        "unique_count": int(present.nunique()),
        "top1_value": str(counts.index[0]),
        "top1_ratio": float(counts.iloc[0]),
        "top5_coverage": float(counts.iloc[:5].sum()),
    }


def infer_type(series: pd.Series) -> str:
    present = series.dropna()
    if present.empty:
        return "categorical"
    parsed = pd.to_numeric(present, errors="coerce")
    if int(parsed.notna().sum()) == len(present):
        return "numeric"
    return "categorical"


def evaluate_feature(entry: dict, samples: pd.DataFrame, label: pd.Series, config) -> dict:
    row = _empty_row(entry["name"], entry["feature_type"], entry["description"], len(samples))
    row["auc_stats"] = entry.get("auc_stats")
    row["absent"] = not entry["in_samples"]
    if row["absent"]:
        row["missing_rate"] = None
        return row

    raw = samples[entry["name"]]
    if entry["feature_type"] == "numeric":
        _fill_numeric(row, raw, label, config)
    else:
        _fill_categorical(row, raw, label, config)

    if is_number(row["auc"]) and is_number(row["auc_stats"]):
        row["auc_gap"] = abs(float(row["auc"]) - float(row["auc_stats"]))

    missing = raw.isna() if entry["feature_type"] == "categorical" else row["_numeric"].isna()
    observed_missing = float(missing.mean()) if entry["feature_type"] == "numeric" else float(raw.isna().mean())
    if 0 < observed_missing < 1:
        indicator = missing.astype(float)
        row["_missing_indicator_auc"] = roc_auc(indicator, label)
    return row


def _fill_numeric(row: dict, raw: pd.Series, label: pd.Series, config) -> None:
    numeric = pd.to_numeric(raw, errors="coerce")
    row["_numeric"] = numeric
    present = numeric.dropna()
    row["non_null_count"] = int(present.shape[0])
    row["missing_rate"] = float(numeric.isna().mean())
    row["unique_count"] = int(present.nunique()) if not present.empty else 0
    row.update(_distribution(present))

    name = row["feature_name"]
    bounds = config.ranges.get(name)
    sentinels = config.sentinels.get(name, [])
    row["range_configured"] = bounds is not None or bool(sentinels)
    sentinel_mask = _is_sentinel(numeric, sentinels) if sentinels else pd.Series(False, index=numeric.index)
    if bounds is None:
        out_mask = pd.Series(False, index=numeric.index)
    else:
        low, high = bounds
        out_mask = numeric.notna() & ~sentinel_mask & ((numeric < low) | (numeric > high))
    non_null = max(row["non_null_count"], 1)
    if row["range_configured"]:
        row["sentinel_rate"] = float(sentinel_mask.sum() / non_null) if row["non_null_count"] else 0.0
        row["out_of_range_rate"] = float(out_mask.sum() / non_null) if row["non_null_count"] else 0.0

    cleaned = numeric.mask(sentinel_mask | out_mask)
    score = cleaned if config.metrics.get("validity") and row["range_configured"] else numeric
    row["_values"] = score
    row["_auc_unfiltered"] = roc_auc(numeric, label)
    _attach_rank_metrics(row, score, label)

    if not present.empty and row["min"] is not None and row["p25"] is not None:
        same_floor = np.isclose(float(row["min"]), float(row["p25"]), rtol=0, atol=1e-8)
        censor_ratio = float(np.isclose(present.to_numpy(dtype=float), float(row["min"]), rtol=0, atol=1e-8).mean())
        row["censor_ratio"] = censor_ratio
        row["left_censored"] = bool(same_floor and censor_ratio >= config.thresholds["left_censor_min_ratio"])


def _fill_categorical(row: dict, raw: pd.Series, label: pd.Series, config) -> None:
    present = raw.dropna()
    row["non_null_count"] = int(present.shape[0])
    row["missing_rate"] = float(raw.isna().mean())
    row.update(_category_stats(raw))
    row["_values"] = raw
    row["iv"] = information_value(raw, label, int(config.thresholds["iv_min_bin_count"]))
    if config.metrics.get("mutual_info"):
        row["mutual_info"] = mutual_info(raw, label)

    values = present.unique()
    if 0 < len(values) <= 2:
        parsed = pd.to_numeric(pd.Series(values), errors="coerce")
        if int(parsed.notna().sum()) == len(values):
            score = pd.to_numeric(raw, errors="coerce")
            row["binary_scored"] = True
            row["_values_numeric"] = score
            _attach_rank_metrics(row, score, label)
