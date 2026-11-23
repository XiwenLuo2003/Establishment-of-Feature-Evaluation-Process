"""读入三张表，并把空串、空白和空值记号统一成缺失。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "na", "nan", "null", "none", "n/a", "<na>"}


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        text = cleaned[column].astype("string").str.strip()
        missing = text.isna() | text.str.lower().isin(MISSING_TOKENS)
        values = text.mask(missing, other=pd.NA).astype("object")
        values = values.where(values.notna(), other=np.nan)
        cleaned[column] = values
    return cleaned


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"找不到输入文件: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    return _clean_frame(frame)


def load_tables(data_dir: Path, config):
    schema = read_table(data_dir / config.schema_file)
    stats = read_table(data_dir / config.stats_file)
    samples = read_table(data_dir / config.samples_file)
    for name, frame, columns in (
        ("feature_schema", schema, ("feature_name", "feature_type")),
        ("feature_stats", stats, ("feature_name",)),
        ("feature_samples", samples, (config.label_column,)),
    ):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise SystemExit(f"{name} 缺少列: {', '.join(missing)}")
    if config.id_column not in samples.columns:
        raise SystemExit(f"采样表缺少主键列: {config.id_column}")
    return schema, stats, samples
