"""把评估结果写成 CSV 和可读的 Markdown 报告。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUTPUT_COLUMNS = [
    "feature_name",
    "feature_type",
    "description",
    "sample_count",
    "non_null_count",
    "missing_rate",
    "unique_count",
    "mean",
    "std",
    "min",
    "p01",
    "p25",
    "p50",
    "p75",
    "p99",
    "max",
    "skew",
    "out_of_range_rate",
    "sentinel_rate",
    "left_censored",
    "top1_value",
    "top1_ratio",
    "top5_coverage",
    "auc",
    "auc_stats",
    "auc_effective",
    "auc_gap",
    "direction",
    "ks",
    "iv",
    "mutual_info",
    "partner_feature",
    "partner_corr",
    "flags",
    "decision",
    "rule_ids",
    "reasons",
    "action",
]

DECISION_LABEL = {
    "keep": "保留",
    "conditional_keep": "有条件保留",
    "drop": "淘汰",
}

FLAG_LABEL = {
    "high_missing": "缺失偏高",
    "sentinel": "含哨兵值",
    "out_of_range": "含越界值",
    "left_censored": "低端截断",
    "high_cardinality": "高基数",
    "sparse": "头部覆盖过低",
    "near_constant": "近似常量",
    "reverse_direction": "方向相反",
    "weak_signal": "弱信号",
    "stats_mismatch": "摘要与采样不一致",
    "missing_in_samples": "采样中缺列",
}


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{float(value):.2%}"


def write_tables(rows: list[dict], pairs: list[dict], warnings: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS:
        if column not in table.columns:
            table[column] = None
    table[OUTPUT_COLUMNS].to_csv(output_dir / "feature_evaluation.csv", index=False, encoding="utf-8-sig")

    redundancy = pd.DataFrame(pairs)
    redundancy_columns = [
        "feature_a",
        "feature_b",
        "method",
        "coefficient",
        "kept_feature",
        "dropped_feature",
    ]
    if redundancy.empty:
        redundancy = pd.DataFrame(columns=redundancy_columns)
    redundancy[redundancy_columns].to_csv(
        output_dir / "feature_redundancy.csv", index=False, encoding="utf-8-sig"
    )

    warning_frame = pd.DataFrame(warnings)
    warning_columns = ["level", "code", "feature_name", "message"]
    if warning_frame.empty:
        warning_frame = pd.DataFrame(columns=warning_columns)
    warning_frame[warning_columns].to_csv(
        output_dir / "validation_warnings.csv", index=False, encoding="utf-8-sig"
    )


def _metric_line(row: dict) -> str:
    if row["feature_type"] == "numeric":
        return (
            f"AUC {_fmt(row.get('auc'))}，有效 AUC {_fmt(row.get('auc_effective'))}，"
            f"方向 {row.get('direction') or '—'}，KS {_fmt(row.get('ks'))}"
        )
    text = f"IV {_fmt(row.get('iv'))}，互信息 {_fmt(row.get('mutual_info'))}"
    if row.get("binary_scored"):
        text += f"，AUC {_fmt(row.get('auc'))}"
    return text


def render_report(result) -> str:
    config = result.config
    rows = result.rows
    counts = {key: 0 for key in DECISION_LABEL}
    for row in rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1

    lines: list[str] = []
    lines.append("# 特征评估报告")
    lines.append("")
    lines.append("## 1. 运行信息")
    lines.append("")
    lines.append(f"- 采样行数：{result.sample_count}")
    lines.append(f"- 正样本占比：{_pct(result.positive_rate)}（标签 1 为点击/转化等正例）")
    lines.append(f"- 摘要 total_count：{result.stats_total if result.stats_total is not None else '—'}")
    lines.append(f"- AUC 决策口径：`{config.auc_mode}`。effective 表示用 max(AUC, 1-AUC)，反向特征不会只因为 AUC 小于 0.5 被淘汰。")
    lines.append(f"- 类别主指标：{'IV' if config.metrics.get('iv') else '互信息'}。互信息{'参与计算' if config.metrics.get('mutual_info') else '未计算'}。")
    lines.append(
        "- 本次启用的指标："
        + "、".join(config.enabled_metric_names)
    )
    if not config.metrics.get("validity"):
        lines.append("- 合法性未启用：哨兵和越界不参与淘汰，决策 AUC 不剔除这些值。")
    if not config.metrics.get("redundancy"):
        lines.append("- 冗余未启用：相关达到阈值也不会触发 D6。")
    if not config.metrics.get("iv"):
        lines.append(
            f"- 未使用 IV。类别淘汰线改为互信息 {config.thresholds['mi_drop']}，保留线 {config.thresholds['mi_keep']}。"
        )
    lines.append("")
    lines.append("选择这些指标，是因为排序模型要同时回答四件事：特征能不能用、有没有区分度、类别能不能稳定入模、是否和别的特征重复。摘要里的 AUC 只覆盖数值特征，而且会被哨兵值污染，所以决策数字一律由采样重算，摘要只做对照。")
    lines.append("")
    lines.append("类别 IV 在计算前把样本数不足 30 的取值并入 `__other__`，避免稀有取值把信息值抬高。细粒度标签几乎会被并成一箱，IV 会接近 0；这种特征由基数规则淘汰。互信息保留原始取值，只作对照，不把高基数特征救回来。二元类别同时看 IV 和 AUC，取更严的一档。")
    lines.append("")
    lines.append("本次阈值：")
    lines.append("")
    lines.append("| 参数 | 值 | 作用 |")
    lines.append("|---|---|---|")
    meaning = {
        "missing_drop": "缺失率达到该值则 D1 淘汰",
        "auc_drop": "决策 AUC 低于该值则 D3 淘汰",
        "auc_keep": "决策 AUC 达到该值才算保留线",
        "iv_drop": "IV 低于该值则 D4 淘汰",
        "iv_keep": "IV 达到该值才算类别保留线",
        "cardinality_drop": "类别唯一值达到该值则 D5 淘汰",
        "top5_sparse": "Top5 覆盖低于该值则 D5 淘汰",
        "invalid_drop": "哨兵加越界达到该值则 D7 淘汰",
        "redundancy_drop": "相关达到该值则 D6 淘汰较弱的一方",
        "redundancy_report": "相关达到该值写入冗余表",
    }
    for key, text in meaning.items():
        lines.append(f"| `{key}` | {config.thresholds[key]:.4g} | {text} |")
    lines.append("")
    lines.append("## 2. 结论汇总")
    lines.append("")
    lines.append(
        f"共评估 {len(rows)} 个特征：保留 {counts.get('keep', 0)} 个，"
        f"有条件保留 {counts.get('conditional_keep', 0)} 个，淘汰 {counts.get('drop', 0)} 个。"
    )
    lines.append("")
    lines.append("| 特征 | 类型 | 决策 | 规则 | 关键指标 | 入模前动作 |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        action = row.get("action") or "—"
        lines.append(
            f"| `{row['feature_name']}` | {row['feature_type']} | {DECISION_LABEL[row['decision']]} "
            f"| {row.get('rule_ids') or '—'} | {_metric_line(row)} | {action} |"
        )
    lines.append("")
    for decision, title in (
        ("keep", "直接保留"),
        ("conditional_keep", "有条件保留"),
        ("drop", "淘汰"),
    ):
        names = [row["feature_name"] for row in rows if row["decision"] == decision]
        body = "、".join(f"`{name}`" for name in names) if names else "无"
        lines.append(f"- {title}：{body}")
    lines.append("")
    lines.append("## 3. 特征明细")
    lines.append("")
    for row in rows:
        lines.append(f"### `{row['feature_name']}`")
        lines.append("")
        lines.append(f"- 含义：{row.get('description') or '采样中多出的列，描述表未登记'}")
        lines.append(f"- 决策：{DECISION_LABEL[row['decision']]}")
        lines.append(f"- 规则：{row.get('rule_ids') or '无'}")
        lines.append(f"- 理由：{row.get('reasons') or '达到保留线，无附加动作'}")
        if row.get("action"):
            lines.append(f"- 动作：{row['action']}")
        lines.append(
            f"- 完整性：非空 {row.get('non_null_count')} / {row.get('sample_count')}，缺失率 {_pct(row.get('missing_rate'))}"
        )
        if row["feature_type"] == "numeric":
            lines.append(
                f"- 分布：min {_fmt(row.get('min'), 3)}，P50 {_fmt(row.get('p50'), 3)}，"
                f"P99 {_fmt(row.get('p99'), 3)}，max {_fmt(row.get('max'), 3)}"
            )
            lines.append(
                f"- 合法性：哨兵 {_pct(row.get('sentinel_rate'))}，越界 {_pct(row.get('out_of_range_rate'))}，"
                f"低端截断 {'是' if row.get('left_censored') else '否'}"
            )
            if row.get("_auc_unfiltered") is not None and row.get("sentinel_rate"):
                lines.append(
                    f"- 未剔除哨兵的 AUC {_fmt(row.get('_auc_unfiltered'))}；决策使用剔除后的 AUC {_fmt(row.get('auc'))}。"
                )
        else:
            lines.append(
                f"- 可用性：唯一值 {row.get('unique_count')}，最高频 {row.get('top1_value') or '—'} "
                f"占比 {_pct(row.get('top1_ratio'))}，Top5 覆盖 {_pct(row.get('top5_coverage'))}"
            )
            lines.append(f"- 区分度：IV {_fmt(row.get('iv'))}，互信息 {_fmt(row.get('mutual_info'))}")
            if row.get("binary_scored"):
                lines.append(
                    f"- 该类别只有两个数值取值，同时计算 AUC {_fmt(row.get('auc'))}（有效 {_fmt(row.get('auc_effective'))}）。决策取更严的一档。"
                )
        if row["feature_type"] == "numeric":
            lines.append(f"- 区分度：{_metric_line(row)}。摘要 AUC {_fmt(row.get('auc_stats'))}，绝对差 {_fmt(row.get('auc_gap'))}。")
        flags = [FLAG_LABEL.get(flag, flag) for flag in str(row.get("flags") or "").split(";") if flag]
        lines.append(f"- 质量标记：{'、'.join(flags) if flags else '无'}")
        indicator = row.get("_missing_indicator_auc")
        if indicator is not None and row.get("missing_rate") is not None and float(row["missing_rate"]) >= 0.05:
            extra = "缺失率已经达到淘汰线，缺失指示不能把该特征留在模型里。" if float(row["missing_rate"]) >= config.thresholds["missing_drop"] else "若保留该特征，建议同时使用缺失指示。"
            lines.append(f"- 缺失指示 AUC {_fmt(indicator)}。{extra}")
        lines.append("")

    lines.append("## 4. 数据质量")
    lines.append("")
    lines.append("空串、纯空白、`NA`/`null`/`none` 以及数值列中无法解析的文本都计为缺失。决策使用的缺失率、分位数、AUC、IV 和基数全部来自采样。摘要只用于对照；两者的 AUC 绝对差大于配置的 `auc_gap_warn` 时记为不一致，决策仍以采样为准。")
    lines.append("")
    lines.append("数值特征先按配置剔除哨兵值和合法区间外的取值，再计算 AUC 和 KS。因此少量哨兵不会直接淘汰一个仍有区分度的特征，只会要求入模前清洗；哨兵加越界达到淘汰线，或清洗后仍然没有区分度，才会淘汰。低端截断（最小值等于 P25，且该值占比达到阈值）只附加说明，不单独淘汰。")
    lines.append("")
    quality_rows = [
        row
        for row in rows
        if any(
            flag in str(row.get("flags") or "")
            for flag in ("high_missing", "sentinel", "out_of_range", "left_censored", "stats_mismatch", "missing_in_samples", "sparse", "high_cardinality")
        )
    ]
    if not quality_rows:
        lines.append("本批特征没有触发质量标记。")
    else:
        lines.append("| 特征 | 标记 | 处理 |")
        lines.append("|---|---|---|")
        for row in quality_rows:
            flags = "、".join(
                FLAG_LABEL.get(flag, flag) for flag in str(row.get("flags") or "").split(";") if flag
            )
            lines.append(
                f"| `{row['feature_name']}` | {flags} | {DECISION_LABEL[row['decision']]}：{row.get('reasons') or '无附加规则'} |"
            )
    lines.append("")
    if result.warnings:
        lines.append("校验记录：")
        lines.append("")
        for warning in result.warnings:
            feature = f"`{warning['feature_name']}` " if warning.get("feature_name") else ""
            lines.append(f"- [{warning['level']}] {warning['code']}：{feature}{warning['message']}")
        lines.append("")

    lines.append("## 5. 冗余")
    lines.append("")
    lines.append("相关只在同类型特征之间计算。数值用 Spearman 绝对值，类别用 Cramér's V。达到展示阈值的特征对写入下表；只有达到淘汰阈值，且双方都还没因缺失、无信号、基数或非法取值被淘汰时，才淘汰区分度更弱的一方。数值和类别之间不自动淘汰。区分度差距小于 `discrimination_tie` 时，留下缺失率更低的一方。")
    lines.append("")
    if not result.pairs:
        lines.append("没有相关绝对值达到展示阈值的同类型特征对。")
    else:
        lines.append("| 特征 A | 特征 B | 方法 | 系数 | 冗余淘汰 |")
        lines.append("|---|---|---|---|---|")
        for pair in result.pairs:
            if pair.get("dropped_feature"):
                note = f"淘汰 `{pair['dropped_feature']}`，保留 `{pair['kept_feature']}`"
            else:
                note = "不淘汰"
            lines.append(
                f"| `{pair['feature_a']}` | `{pair['feature_b']}` | {pair['method']} | "
                f"{float(pair['coefficient']):.3f} | {note} |"
            )
    lines.append("")
    lines.append("## 6. 复现方式")
    lines.append("")
    lines.append("```bash")
    lines.append(result.command)
    lines.append("```")
    lines.append("")
    lines.append(f"配置文件：`{config.config_path}`")
    lines.append("")
    lines.append("同输入、同配置再运行一次，评估表中的决策、规则编号和指标应保持一致。")
    lines.append("")
    return "\n".join(lines)


def write_report(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(render_report(result), encoding="utf-8")
