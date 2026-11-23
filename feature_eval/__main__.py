"""命令行入口：python -m feature_eval --data-dir <目录> --config config.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from feature_eval.pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估候选特征并输出筛选结论和报告")
    parser.add_argument("--data-dir", required=True, help="存放三张输入 CSV 的目录")
    parser.add_argument("--config", required=True, help="阈值和取值范围配置，YAML")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认是数据目录下的 output")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    data_dir = Path(arguments.data_dir).resolve()
    config_path = Path(arguments.config).resolve()
    output_dir = Path(arguments.output_dir).resolve() if arguments.output_dir else data_dir / "output"
    if not data_dir.is_dir():
        raise SystemExit(f"数据目录不存在: {data_dir}")
    command = "python -m feature_eval " + " ".join(sys.argv[1:] if argv is None else argv)
    result = run(data_dir, config_path, output_dir, command)
    counts: dict[str, int] = {}
    for row in result.rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    print(
        f"评估完成：保留 {counts.get('keep', 0)}，"
        f"有条件保留 {counts.get('conditional_keep', 0)}，"
        f"淘汰 {counts.get('drop', 0)}。报告：{output_dir / 'report.md'}"
    )


if __name__ == "__main__":
    main()
