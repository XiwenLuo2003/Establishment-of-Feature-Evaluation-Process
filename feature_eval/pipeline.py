"""把装载、评估、冗余、决策和报告串成一次运行。"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from feature_eval.config import Config, load_config
from feature_eval.decision import apply_redundancy, build_flags, finalize_decision, is_number, single_feature_hits
from feature_eval.evaluate import evaluate_feature, infer_type
from feature_eval.loader import load_tables
from feature_eval.redundancy import pairwise_associations
from feature_eval.report import write_report, write_tables


class RunResult:
    def __init__(self, config: Config, command: str):
        self.config = config
        self.command = command
        self.rows: list[dict] = []
        self.pairs: list[dict] = []
        self.warnings: list[dict] = []
        self.sample_count = 0
        self.positive_rate = 0.0
        self.stats_total: int | None = None


def _warn(result: RunResult, level: str, code: str, message: str, feature_name: str = "") -> None:
    result.warnings.append(
        {"level": level, "code": code, "feature_name": feature_name, "message": message}
    )


def _optional_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "<na>"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def _catalog(schema: pd.DataFrame, stats: pd.DataFrame, samples: pd.DataFrame, result: RunResult) -> list[dict]:
    config = result.config
    stats_by_name = {}
    totals = []
    for _, record in stats.iterrows():
        name = str(record["feature_name"]).strip()
        stats_by_name[name] = record
        total = _optional_float(record.get("total_count")) if "total_count" in stats.columns else None
        if total is not None:
            totals.append(int(total))
    result.sample_count = int(len(samples))
    result.stats_total = totals[0] if totals else None
    if result.stats_total is not None and result.stats_total != result.sample_count:
        _warn(
            result,
            "warning",
            "row_count_mismatch",
            f"摘要 total_count={result.stats_total}，采样实际行数={result.sample_count}。决策使用采样行数。",
        )
    elif result.stats_total is not None:
        _warn(
            result,
            "info",
            "row_count_checked",
            f"摘要 total_count={result.stats_total}，采样实际行数={result.sample_count}，两者一致。",
        )

    entries = []
    seen = set()
    for _, record in schema.iterrows():
        name = str(record["feature_name"]).strip()
        if name in config.exclude or name in seen:
            continue
        seen.add(name)
        declared = str(record["feature_type"]).strip().lower()
        if declared not in {"numeric", "categorical"}:
            declared = infer_type(samples[name]) if name in samples.columns else "categorical"
            _warn(result, "warning", "type_inferred", f"无法识别的类型，已按数据推断为 {declared}。", name)
        stats_row = stats_by_name.get(name)
        if stats_row is not None and "feature_type" in stats.columns:
            stats_type = str(stats_row["feature_type"]).strip().lower()
            if stats_type in {"numeric", "categorical"} and stats_type != declared:
                _warn(
                    result,
                    "warning",
                    "type_mismatch",
                    f"摘要类型为 {stats_type}，描述表类型为 {declared}。计算以描述表为准。",
                    name,
                )
        auc_stats = None
        if stats_row is not None and "auc" in stats.columns:
            auc_stats = _optional_float(stats_row["auc"])
        description = ""
        if "description" in schema.columns and pd.notna(record.get("description")):
            description = str(record.get("description")).strip()
        entries.append(
            {
                "name": name,
                "feature_type": declared,
                "description": description,
                "in_samples": name in samples.columns,
                "auc_stats": auc_stats,
            }
        )
        if name not in samples.columns:
            _warn(result, "warning", "missing_in_samples", "描述表中存在，采样中没有该列。", name)

    for column in samples.columns:
        if column in seen or column in config.exclude:
            continue
        inferred = infer_type(samples[column])
        _warn(result, "warning", "unknown_column", f"采样中多出的列，已按 {inferred} 评估。", column)
        entries.append(
            {
                "name": column,
                "feature_type": inferred,
                "description": "",
                "in_samples": True,
                "auc_stats": None,
            }
        )
    return entries


def _label_series(samples: pd.DataFrame, result: RunResult) -> pd.Series:
    label = pd.to_numeric(samples[result.config.label_column], errors="coerce")
    valid = label.dropna()
    unexpected = valid[~valid.isin([0, 1])]
    if not unexpected.empty:
        _warn(
            result,
            "warning",
            "label_not_binary",
            f"有 {int(unexpected.shape[0])} 个标签不是 0/1，这些行不参与区分度计算。",
        )
    usable = label[label.isin([0, 1])]
    if usable.empty or usable.nunique() < 2:
        _warn(result, "warning", "label_single_class", "标签只有一个取值或全部缺失，区分度指标为空，不用 0 填充。")
        result.positive_rate = float(usable.mean()) if not usable.empty else 0.0
    else:
        result.positive_rate = float(usable.mean())
    return label


def run(data_dir: Path, config_path: Path, output_dir: Path, command: str) -> RunResult:
    config = load_config(config_path)
    result = RunResult(config, command)
    schema, stats, samples = load_tables(data_dir, config)
    label = _label_series(samples, result)
    entries = _catalog(schema, stats, samples, result)

    rows = [evaluate_feature(entry, samples, label, config) for entry in entries]
    for row in rows:
        row["hits"] = single_feature_hits(row, config)
        if is_number(row.get("auc_gap")) and float(row["auc_gap"]) > config.thresholds["auc_gap_warn"]:
            _warn(
                result,
                "warning",
                "auc_mismatch",
                f"采样 AUC {float(row['auc']):.4f} 与摘要 AUC {float(row['auc_stats']):.4f} 的绝对差为 {float(row['auc_gap']):.4f}。决策使用采样。",
                row["feature_name"],
            )

    pairs = pairwise_associations(rows, config)
    apply_redundancy(rows, pairs, config)
    for row in rows:
        finalize_decision(row, config)
        row["flags"] = ";".join(build_flags(row, config))

    result.rows = rows
    result.pairs = pairs
    for row in rows:
        row.pop("_values", None)
        row.pop("_numeric", None)
        row.pop("_values_numeric", None)
    write_tables(rows, pairs, result.warnings, output_dir)
    write_report(result, output_dir)
    return result
