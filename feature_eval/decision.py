"""单特征规则与冗余规则。阈值全部来自配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Hit:
    rule_id: str
    reason: str
    action: str = ""


def is_number(value) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number)


def decision_auc(row: dict, config) -> float | None:
    if config.auc_mode == "raw":
        value = row.get("auc")
    else:
        value = row.get("auc_effective")
    return float(value) if is_number(value) else None


def discrimination_score(row: dict, config) -> float | None:
    """冗余比较用的区分度。数值用决策 AUC，类别用 IV 或互信息。"""
    if row.get("feature_type") == "numeric":
        return decision_auc(row, config)
    if config.metrics.get("iv") and is_number(row.get("iv")):
        return float(row["iv"])
    if is_number(row.get("mutual_info")):
        return float(row["mutual_info"])
    return None


def _signal_keeps(row: dict, config) -> bool:
    """非空部分是否达到保留线。二元类别要同时满足 AUC 和类别指标。"""
    if not config.metrics.get("discrimination"):
        return True
    thresholds = config.thresholds
    kind = row.get("feature_type")
    numeric_ok = True
    if kind == "numeric" or row.get("binary_scored"):
        score = decision_auc(row, config)
        numeric_ok = score is not None and score >= thresholds["auc_keep"]
    categorical_ok = True
    if kind == "categorical":
        if config.metrics.get("iv"):
            categorical_ok = is_number(row.get("iv")) and float(row["iv"]) >= thresholds["iv_keep"]
        else:
            categorical_ok = (
                is_number(row.get("mutual_info"))
                and float(row["mutual_info"]) >= thresholds["mi_keep"]
            )
    return numeric_ok and categorical_ok


def _dedupe(hits: list[Hit]) -> list[Hit]:
    order: list[str] = []
    bucket: dict[str, Hit] = {}
    for hit in hits:
        current = bucket.get(hit.rule_id)
        if current is None:
            order.append(hit.rule_id)
            bucket[hit.rule_id] = hit
            continue
        reason = current.reason if hit.reason in current.reason else f"{current.reason}；{hit.reason}"
        action = current.action
        if hit.action and hit.action not in action:
            action = f"{action}；{hit.action}".strip("；")
        bucket[hit.rule_id] = Hit(hit.rule_id, reason, action)

    def sort_key(rule_id: str) -> tuple[int, int]:
        family = 0 if rule_id.startswith("D") else 1
        return family, int(rule_id[1:])

    return [bucket[rule_id] for rule_id in sorted(order, key=sort_key)]


def single_feature_hits(row: dict, config) -> list[Hit]:
    thresholds = config.thresholds
    metrics = config.metrics
    hits: list[Hit] = []
    kind = row.get("feature_type")

    if row.get("absent"):
        hits.append(Hit("D8", "特征描述表中存在该特征，采样数据中没有对应列，无法复核"))
        return hits

    missing_rate = row.get("missing_rate")
    if metrics.get("completeness") and is_number(missing_rate) and float(missing_rate) >= thresholds["missing_drop"]:
        hits.append(
            Hit(
                "D1",
                f"缺失率 {float(missing_rate):.2%}，达到淘汰线 {thresholds['missing_drop']:.0%}",
            )
        )

    unique_count = row.get("unique_count")
    top1_ratio = row.get("top1_ratio")
    if metrics.get("usability"):
        if is_number(unique_count) and float(unique_count) <= 1:
            hits.append(Hit("D2", f"近似常量，非空唯一值 {int(float(unique_count))} 个"))
        elif (
            kind == "categorical"
            and is_number(top1_ratio)
            and float(top1_ratio) >= thresholds["top1_constant"]
        ):
            hits.append(
                Hit(
                    "D2",
                    f"近似常量，最高频占比 {float(top1_ratio):.2%}，达到 {thresholds['top1_constant']:.0%}",
                )
            )

    if metrics.get("discrimination") and (kind == "numeric" or row.get("binary_scored")):
        raw_auc = row.get("auc")
        effective = row.get("auc_effective")
        scored = decision_auc(row, config)
        if scored is not None and scored < thresholds["auc_drop"]:
            hits.append(
                Hit(
                    "D3",
                    f"决策 AUC {scored:.4f} 低于淘汰线 {thresholds['auc_drop']:.2f}",
                )
            )
        elif scored is not None and scored < thresholds["auc_keep"]:
            hits.append(
                Hit(
                    "C3",
                    f"决策 AUC {scored:.4f} 落在弱信号区间 [{thresholds['auc_drop']:.2f}, {thresholds['auc_keep']:.2f})",
                    "弱信号，进入模型后做消融，不作为优先特征",
                )
            )
        if (
            is_number(raw_auc)
            and is_number(effective)
            and float(raw_auc) < 0.5
            and float(effective) >= thresholds["auc_keep"]
        ):
            hits.append(
                Hit(
                    "C1",
                    f"原始 AUC {float(raw_auc):.4f} 低于 0.5，有效 AUC {float(effective):.4f}，方向与标签相反",
                    "按负向使用，或取相反数后再入模",
                )
            )

    if kind == "categorical" and metrics.get("discrimination"):
        if metrics.get("iv"):
            iv = row.get("iv")
            if is_number(iv) and float(iv) < thresholds["iv_drop"]:
                hits.append(Hit("D4", f"信息值 IV {float(iv):.4f} 低于淘汰线 {thresholds['iv_drop']:.2f}"))
            elif is_number(iv) and float(iv) < thresholds["iv_keep"]:
                hits.append(
                    Hit(
                        "C3",
                        f"信息值 IV {float(iv):.4f} 落在弱信号区间 [{thresholds['iv_drop']:.2f}, {thresholds['iv_keep']:.2f})",
                        "弱信号，进入模型后做消融，不作为优先特征",
                    )
                )
        else:
            mi = row.get("mutual_info")
            if is_number(mi) and float(mi) < thresholds["mi_drop"]:
                hits.append(
                    Hit(
                        "D3",
                        f"互信息 {float(mi):.4f} 低于配置淘汰线 {thresholds['mi_drop']:.4f}，本次未使用 IV",
                    )
                )
            elif is_number(mi) and float(mi) < thresholds["mi_keep"]:
                hits.append(
                    Hit(
                        "C3",
                        f"互信息 {float(mi):.4f} 低于保留线 {thresholds['mi_keep']:.4f}，本次未使用 IV",
                        "弱信号，进入模型后做消融，不作为优先特征",
                    )
                )

    if kind == "categorical" and metrics.get("usability"):
        uniq = int(float(unique_count)) if is_number(unique_count) else 0
        top5 = row.get("top5_coverage")
        too_many = uniq >= thresholds["cardinality_drop"]
        too_sparse = is_number(top5) and float(top5) < thresholds["top5_sparse"]
        if too_many or too_sparse:
            coverage = f"{float(top5):.2%}" if is_number(top5) else "无"
            hits.append(
                Hit(
                    "D5",
                    f"类别过稀疏，唯一值 {uniq} 个，Top5 覆盖 {coverage}",
                )
            )
        elif thresholds["cardinality_encode_min"] <= uniq < thresholds["cardinality_drop"]:
            hits.append(
                Hit(
                    "C4",
                    f"类别唯一值 {uniq} 个，处于需要编码的区间",
                    "使用目标编码或哈希，不做独热",
                )
            )

    if kind == "numeric" and metrics.get("validity") and row.get("range_configured"):
        sentinel_rate = float(row.get("sentinel_rate") or 0)
        out_rate = float(row.get("out_of_range_rate") or 0)
        invalid = sentinel_rate + out_rate
        if invalid >= thresholds["invalid_drop"]:
            hits.append(
                Hit(
                    "D7",
                    f"哨兵与越界占比合计 {invalid:.2%}，达到淘汰线 {thresholds['invalid_drop']:.0%}",
                )
            )
        elif invalid > 0 and _signal_keeps(row, config):
            hits.append(
                Hit(
                    "C2",
                    f"哨兵与越界占比合计 {invalid:.2%}，剔除后仍达到保留线",
                    "入模前把哨兵值和越界值置空或截断到合法区间",
                )
            )

    if (
        kind == "numeric"
        and metrics.get("validity")
        and row.get("left_censored")
        and _signal_keeps(row, config)
    ):
        ratio = row.get("censor_ratio")
        ratio_text = f"{float(ratio):.2%}" if is_number(ratio) else "不低于阈值"
        hits.append(
            Hit(
                "C6",
                f"低端截断，最小值占比 {ratio_text}",
                "保留该特征，使用时注意低端截断",
            )
        )

    if (
        metrics.get("completeness")
        and is_number(missing_rate)
        and thresholds["missing_indicator_min"] <= float(missing_rate) < thresholds["missing_drop"]
        and _signal_keeps(row, config)
    ):
        hits.append(
            Hit(
                "C5",
                f"缺失率 {float(missing_rate):.2%}，非空部分达到保留线",
                "同时加入缺失指示特征",
            )
        )

    return _dedupe(hits)


def strictly_better(left: dict, right: dict, config) -> bool:
    """left 是否应当在冗余比较中留下、right 被淘汰。"""
    tie = config.thresholds["discrimination_tie"]
    left_score = discrimination_score(left, config)
    right_score = discrimination_score(right, config)
    if left_score is None and right_score is None:
        return left["feature_name"] < right["feature_name"]
    if left_score is None:
        return False
    if right_score is None:
        return True
    diff = left_score - right_score
    if diff >= tie:
        return True
    if diff <= -tie:
        return False
    left_missing = float(left.get("missing_rate") or 0)
    right_missing = float(right.get("missing_rate") or 0)
    if left_missing < right_missing:
        return True
    if left_missing > right_missing:
        return False
    if left_score != right_score:
        return left_score > right_score
    return left["feature_name"] < right["feature_name"]


def apply_redundancy(rows: list[dict], pairs: list[dict], config) -> None:
    """只在尚未被 D1–D5、D7、D8 淘汰的同类型特征之间执行 D6。"""
    if not config.metrics.get("redundancy"):
        for pair in pairs:
            pair["kept_feature"] = ""
            pair["dropped_feature"] = ""
        return

    blocked = {"D1", "D2", "D3", "D4", "D5", "D7", "D8"}
    survivors = {}
    for row in rows:
        hit_ids = {hit.rule_id for hit in row.get("hits", [])}
        if hit_ids.isdisjoint(blocked) and not row.get("absent"):
            survivors[row["feature_name"]] = row

    threshold = config.thresholds["redundancy_drop"]
    beaters: dict[str, list[tuple[str, float]]] = {}
    for pair in pairs:
        pair["kept_feature"] = ""
        pair["dropped_feature"] = ""
        if float(pair["coefficient"]) < threshold:
            continue
        left = survivors.get(pair["feature_a"])
        right = survivors.get(pair["feature_b"])
        if left is None or right is None:
            continue
        if strictly_better(left, right, config):
            winner, loser = left, right
        elif strictly_better(right, left, config):
            winner, loser = right, left
        else:
            continue
        beaters.setdefault(loser["feature_name"], []).append(
            (winner["feature_name"], float(pair["coefficient"]))
        )
        pair["kept_feature"] = winner["feature_name"]
        pair["dropped_feature"] = loser["feature_name"]

    for row in rows:
        winners = beaters.get(row["feature_name"])
        if not winners:
            continue
        best_name, best_corr = max(
            winners,
            key=lambda item: (
                discrimination_score(survivors[item[0]], config) or -1,
                item[1],
            ),
        )
        row["partner_feature"] = best_name
        row["partner_corr"] = best_corr
        row.setdefault("hits", []).append(
            Hit(
                "D6",
                (
                    f"与 {best_name} 的相关绝对值 {best_corr:.3f} 达到 {threshold:.2f}，"
                    "对方区分度更高或缺失更低，判定冗余"
                ),
            )
        )


def meets_keep(row: dict, config) -> bool:
    if row.get("absent"):
        return False
    thresholds = config.thresholds
    metrics = config.metrics
    if not _signal_keeps(row, config):
        return False
    if row.get("feature_type") == "categorical" and metrics.get("usability"):
        unique_count = int(float(row["unique_count"])) if is_number(row.get("unique_count")) else 0
        top1 = row.get("top1_ratio")
        if unique_count >= thresholds["cardinality_encode_min"]:
            return False
        if is_number(top1) and float(top1) >= thresholds["top1_constant"]:
            return False
    return True


def finalize_decision(row: dict, config) -> None:
    hits = _dedupe(list(row.get("hits", [])))
    drop_hits = [hit for hit in hits if hit.rule_id.startswith("D")]
    conditional_hits = [hit for hit in hits if hit.rule_id.startswith("C")]
    if drop_hits:
        decision = "drop"
        action = ""
    elif conditional_hits:
        decision = "conditional_keep"
        action = "；".join(hit.action for hit in conditional_hits if hit.action)
    elif meets_keep(row, config):
        decision = "keep"
        action = ""
    else:
        decision = "drop"
        if row.get("feature_type") == "categorical" and not row.get("binary_scored"):
            fallback = Hit("D4", "类别区分度无法计算，不能视为达到保留线")
        else:
            fallback = Hit("D3", "区分度无法计算，不能视为达到保留线")
        hits = _dedupe(hits + [fallback])
        action = ""
    row["hits"] = hits
    row["decision"] = decision
    row["rule_ids"] = ";".join(hit.rule_id for hit in hits)
    row["reasons"] = ";".join(hit.reason for hit in hits)
    row["action"] = action


FLAG_ORDER = (
    "high_missing",
    "sentinel",
    "out_of_range",
    "left_censored",
    "high_cardinality",
    "sparse",
    "near_constant",
    "reverse_direction",
    "weak_signal",
    "stats_mismatch",
    "missing_in_samples",
)


def build_flags(row: dict, config) -> list[str]:
    thresholds = config.thresholds
    flags: list[str] = []
    missing_rate = row.get("missing_rate")
    if is_number(missing_rate) and float(missing_rate) >= thresholds["missing_indicator_min"]:
        flags.append("high_missing")
    if is_number(row.get("sentinel_rate")) and float(row["sentinel_rate"]) > 0:
        flags.append("sentinel")
    if is_number(row.get("out_of_range_rate")) and float(row["out_of_range_rate"]) > 0:
        flags.append("out_of_range")
    if row.get("left_censored"):
        flags.append("left_censored")

    unique_count = int(float(row["unique_count"])) if is_number(row.get("unique_count")) else None
    top1 = row.get("top1_ratio")
    if row.get("feature_type") == "categorical" and unique_count is not None:
        if unique_count >= thresholds["cardinality_encode_min"]:
            flags.append("high_cardinality")
        top5 = row.get("top5_coverage")
        if is_number(top5) and float(top5) < thresholds["top5_sparse"]:
            flags.append("sparse")
        if unique_count <= 1 or (is_number(top1) and float(top1) >= thresholds["top1_constant"]):
            flags.append("near_constant")
    elif unique_count is not None and unique_count <= 1 and not row.get("absent"):
        flags.append("near_constant")

    if is_number(row.get("auc")) and float(row["auc"]) < 0.5:
        flags.append("reverse_direction")

    scored = decision_auc(row, config)
    weak_auc = (
        scored is not None
        and thresholds["auc_drop"] <= scored < thresholds["auc_keep"]
        and (row.get("feature_type") == "numeric" or row.get("binary_scored"))
    )
    weak_iv = (
        row.get("feature_type") == "categorical"
        and config.metrics.get("iv")
        and is_number(row.get("iv"))
        and thresholds["iv_drop"] <= float(row["iv"]) < thresholds["iv_keep"]
    )
    if weak_auc or weak_iv:
        flags.append("weak_signal")

    if (
        is_number(row.get("auc"))
        and is_number(row.get("auc_stats"))
        and abs(float(row["auc"]) - float(row["auc_stats"])) > thresholds["auc_gap_warn"]
    ):
        flags.append("stats_mismatch")
    if row.get("absent"):
        flags.append("missing_in_samples")

    return [name for name in FLAG_ORDER if name in flags]
