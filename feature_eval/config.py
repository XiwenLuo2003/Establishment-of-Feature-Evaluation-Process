"""读取并校验 config.yaml。缺阈值时直接失败，不静默回退。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REQUIRED_THRESHOLDS = (
    "missing_drop",
    "missing_indicator_min",
    "auc_drop",
    "auc_keep",
    "iv_drop",
    "iv_keep",
    "mi_drop",
    "mi_keep",
    "top1_constant",
    "cardinality_drop",
    "cardinality_encode_min",
    "top5_sparse",
    "invalid_drop",
    "left_censor_min_ratio",
    "iv_min_bin_count",
    "redundancy_drop",
    "redundancy_report",
    "discrimination_tie",
    "auc_gap_warn",
)

REQUIRED_METRICS = (
    "completeness",
    "validity",
    "discrimination",
    "usability",
    "redundancy",
    "iv",
    "mutual_info",
)


class ConfigError(SystemExit):
    """配置不完整或非法。用 SystemExit 让命令行直接看到原因。"""


@dataclass
class Config:
    schema_file: str
    stats_file: str
    samples_file: str
    id_column: str
    label_column: str
    exclude: list[str]
    metrics: dict[str, bool]
    auc_mode: str
    thresholds: dict[str, float]
    ranges: dict[str, tuple[float, float]]
    sentinels: dict[str, list[float]]
    config_path: str
    enabled_metric_names: list[str] = field(default_factory=list)


def _require(mapping: dict, key: str, path: str):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ConfigError(f"配置缺少必填项: {path}")
    return mapping[key]


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"找不到配置文件: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("配置文件内容为空或不是映射")

    inputs = _require(raw, "inputs", "inputs")
    columns = _require(raw, "columns", "columns")
    metrics_raw = _require(raw, "metrics", "metrics")
    auc_raw = _require(raw, "auc", "auc")
    thresholds_raw = _require(raw, "thresholds", "thresholds")

    metrics: dict[str, bool] = {}
    for name in REQUIRED_METRICS:
        if name not in metrics_raw:
            raise ConfigError(f"配置缺少必填项: metrics.{name}")
        metrics[name] = bool(metrics_raw[name])

    mode = str(_require(auc_raw, "mode", "auc.mode")).strip().lower()
    if mode not in {"effective", "raw"}:
        raise ConfigError("auc.mode 只能是 effective 或 raw")

    thresholds: dict[str, float] = {}
    for name in REQUIRED_THRESHOLDS:
        if name not in thresholds_raw:
            raise ConfigError(f"配置缺少必填项: thresholds.{name}")
        thresholds[name] = float(thresholds_raw[name])

    if thresholds["auc_drop"] <= 0 or thresholds["auc_keep"] <= 0:
        raise ConfigError("AUC 阈值必须大于 0")
    if thresholds["redundancy_report"] > thresholds["redundancy_drop"]:
        raise ConfigError("redundancy_report 不能高于 redundancy_drop")

    ranges: dict[str, tuple[float, float]] = {}
    for name, bounds in (raw.get("ranges") or {}).items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ConfigError(f"ranges.{name} 需要 [下限, 上限]")
        low, high = float(bounds[0]), float(bounds[1])
        if low > high:
            raise ConfigError(f"ranges.{name} 下限大于上限")
        ranges[str(name)] = (low, high)

    sentinels: dict[str, list[float]] = {}
    for name, values in (raw.get("sentinels") or {}).items():
        if not isinstance(values, (list, tuple)):
            raise ConfigError(f"sentinels.{name} 需要列表")
        sentinels[str(name)] = [float(value) for value in values]

    exclude = [str(item) for item in _require(columns, "exclude", "columns.exclude")]
    enabled = [name for name, on in metrics.items() if on]
    return Config(
        schema_file=str(_require(inputs, "schema", "inputs.schema")),
        stats_file=str(_require(inputs, "stats", "inputs.stats")),
        samples_file=str(_require(inputs, "samples", "inputs.samples")),
        id_column=str(_require(columns, "id", "columns.id")),
        label_column=str(_require(columns, "label", "columns.label")),
        exclude=exclude,
        metrics=metrics,
        auc_mode=mode,
        thresholds=thresholds,
        ranges=ranges,
        sentinels=sentinels,
        config_path=str(config_path.resolve()),
        enabled_metric_names=enabled,
    )
