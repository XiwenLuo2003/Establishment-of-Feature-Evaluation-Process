"""决策规则的边界测试。不依赖官方采样，直接喂指标。"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from feature_eval.config import load_config
from feature_eval.decision import apply_redundancy, finalize_decision, single_feature_hits

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"


def row(**kwargs):
    base = {
        "feature_name": "f",
        "feature_type": "numeric",
        "missing_rate": 0.0,
        "unique_count": 100,
        "top1_ratio": None,
        "top5_coverage": None,
        "auc": 0.60,
        "auc_effective": 0.60,
        "auc_stats": 0.60,
        "iv": None,
        "mutual_info": None,
        "sentinel_rate": 0.0,
        "out_of_range_rate": 0.0,
        "left_censored": False,
        "censor_ratio": 0.0,
        "absent": False,
        "binary_scored": False,
        "range_configured": True,
        "partner_feature": "",
        "partner_corr": None,
        "hits": [],
    }
    base.update(kwargs)
    return base


def decide(sample, config):
    sample = copy.deepcopy(sample)
    sample["hits"] = single_feature_hits(sample, config)
    finalize_decision(sample, config)
    return sample


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_keep_clear_numeric(self):
        result = decide(row(), self.config)
        self.assertEqual(result["decision"], "keep")
        self.assertEqual(result["rule_ids"], "")
        self.assertEqual(result["action"], "")

    def test_auc_boundaries(self):
        self.assertEqual(decide(row(auc=0.53, auc_effective=0.53), self.config)["decision"], "conditional_keep")
        self.assertIn("C3", decide(row(auc=0.53, auc_effective=0.53), self.config)["rule_ids"])
        kept = decide(row(auc=0.55, auc_effective=0.55), self.config)
        self.assertEqual(kept["decision"], "keep")
        self.assertNotIn("C3", kept["rule_ids"])
        dropped = decide(row(auc=0.529999, auc_effective=0.529999), self.config)
        self.assertEqual(dropped["decision"], "drop")
        self.assertIn("D3", dropped["rule_ids"])

    def test_missing_boundaries(self):
        high = decide(row(missing_rate=0.80, auc=0.90, auc_effective=0.90), self.config)
        self.assertEqual(high["decision"], "drop")
        self.assertIn("D1", high["rule_ids"])
        self.assertNotIn("C5", high["rule_ids"])
        exact = decide(row(missing_rate=0.50), self.config)
        self.assertIn("D1", exact["rule_ids"])
        below = decide(row(missing_rate=0.499), self.config)
        self.assertNotIn("D1", below["rule_ids"])
        self.assertIn("C5", below["rule_ids"])
        self.assertEqual(decide(row(missing_rate=0.05), self.config)["decision"], "conditional_keep")
        self.assertNotIn("C5", decide(row(missing_rate=0.049), self.config)["rule_ids"])

    def test_reverse_direction_is_conditional(self):
        result = decide(row(auc=0.40, auc_effective=0.60), self.config)
        self.assertEqual(result["decision"], "conditional_keep")
        self.assertIn("C1", result["rule_ids"])
        weak = decide(row(auc=0.46, auc_effective=0.54), self.config)
        self.assertIn("C3", weak["rule_ids"])
        self.assertNotIn("C1", weak["rule_ids"])
        self.assertEqual(weak["decision"], "conditional_keep")

    def test_sentinel_partitions(self):
        dropped = decide(row(sentinel_rate=0.08, auc=0.70, auc_effective=0.70), self.config)
        self.assertEqual(dropped["decision"], "drop")
        self.assertIn("D7", dropped["rule_ids"])
        cleaned = decide(row(sentinel_rate=0.02, auc=0.60, auc_effective=0.60), self.config)
        self.assertEqual(cleaned["decision"], "conditional_keep")
        self.assertIn("C2", cleaned["rule_ids"])
        still_weak = decide(row(sentinel_rate=0.02, auc=0.51, auc_effective=0.51), self.config)
        self.assertEqual(still_weak["decision"], "drop")
        self.assertIn("D3", still_weak["rule_ids"])
        self.assertNotIn("C2", still_weak["rule_ids"])
        band = decide(row(sentinel_rate=0.02, auc=0.54, auc_effective=0.54), self.config)
        self.assertIn("C3", band["rule_ids"])
        self.assertNotIn("C2", band["rule_ids"])
        exact = decide(row(sentinel_rate=0.05, auc=0.70, auc_effective=0.70), self.config)
        self.assertIn("D7", exact["rule_ids"])

    def test_missing_does_not_lose_to_reverse_signal(self):
        result = decide(row(missing_rate=0.60, auc=0.40, auc_effective=0.60), self.config)
        self.assertEqual(result["decision"], "drop")
        self.assertIn("D1", result["rule_ids"])

    def test_censor_does_not_drop_a_useful_feature(self):
        result = decide(
            row(left_censored=True, censor_ratio=0.25, auc=0.60, auc_effective=0.60),
            self.config,
        )
        self.assertEqual(result["decision"], "conditional_keep")
        self.assertIn("C6", result["rule_ids"])
        weak = decide(row(left_censored=True, censor_ratio=0.25, auc=0.51, auc_effective=0.51), self.config)
        self.assertEqual(weak["decision"], "drop")
        self.assertIn("D3", weak["rule_ids"])
        self.assertNotIn("C6", weak["rule_ids"])

    def test_binary_takes_stricter_rule(self):
        low_auc = decide(
            row(
                feature_name="pay",
                feature_type="categorical",
                binary_scored=True,
                unique_count=2,
                top1_ratio=0.8,
                top5_coverage=1.0,
                iv=0.15,
                auc=0.52,
                auc_effective=0.52,
                range_configured=False,
            ),
            self.config,
        )
        self.assertEqual(low_auc["decision"], "drop")
        self.assertIn("D3", low_auc["rule_ids"])
        low_iv = decide(
            row(
                feature_name="pay",
                feature_type="categorical",
                binary_scored=True,
                unique_count=2,
                top1_ratio=0.8,
                top5_coverage=1.0,
                iv=0.01,
                auc=0.70,
                auc_effective=0.70,
                range_configured=False,
            ),
            self.config,
        )
        self.assertEqual(low_iv["decision"], "drop")
        self.assertIn("D4", low_iv["rule_ids"])

    def test_cardinality(self):
        rare = decide(
            row(
                feature_type="categorical",
                unique_count=4836,
                top1_ratio=0.01,
                top5_coverage=0.02,
                iv=0.40,
                auc=None,
                auc_effective=None,
                range_configured=False,
            ),
            self.config,
        )
        self.assertEqual(rare["decision"], "drop")
        self.assertIn("D5", rare["rule_ids"])
        encoded = decide(
            row(
                feature_type="categorical",
                unique_count=198,
                top1_ratio=0.06,
                top5_coverage=0.40,
                iv=0.20,
                auc=None,
                auc_effective=None,
                range_configured=False,
            ),
            self.config,
        )
        self.assertEqual(encoded["decision"], "conditional_keep")
        self.assertIn("C4", encoded["rule_ids"])
        self.assertIn("哈希", encoded["action"])

    def test_absent_feature(self):
        result = decide(row(absent=True, missing_rate=None, auc=None, auc_effective=None), self.config)
        self.assertEqual(result["decision"], "drop")
        self.assertIn("D8", result["rule_ids"])

    def test_redundancy_ignores_already_dropped_partner(self):
        strong = row(feature_name="strong", auc=0.66, auc_effective=0.66)
        weak = row(feature_name="weak", auc=0.51, auc_effective=0.51)
        strong["hits"] = single_feature_hits(strong, self.config)
        weak["hits"] = single_feature_hits(weak, self.config)
        pairs = [
            {
                "feature_a": "strong",
                "feature_b": "weak",
                "method": "spearman",
                "coefficient": 0.96,
                "kept_feature": "",
                "dropped_feature": "",
            }
        ]
        apply_redundancy([strong, weak], pairs, self.config)
        finalize_decision(strong, self.config)
        finalize_decision(weak, self.config)
        self.assertEqual(strong["decision"], "keep")
        self.assertNotIn("D6", strong["rule_ids"])
        self.assertEqual(weak["decision"], "drop")
        self.assertNotIn("D6", weak["rule_ids"])
        self.assertEqual(pairs[0]["dropped_feature"], "")

    def test_redundancy_tie_keeps_lower_missing(self):
        sparse = row(feature_name="sparse", auc=0.600, auc_effective=0.600, missing_rate=0.30)
        dense = row(feature_name="dense", auc=0.594, auc_effective=0.594, missing_rate=0.0)
        sparse["hits"] = single_feature_hits(sparse, self.config)
        dense["hits"] = single_feature_hits(dense, self.config)
        pairs = [
            {
                "feature_a": "dense",
                "feature_b": "sparse",
                "method": "spearman",
                "coefficient": 0.95,
                "kept_feature": "",
                "dropped_feature": "",
            }
        ]
        apply_redundancy([sparse, dense], pairs, self.config)
        finalize_decision(sparse, self.config)
        finalize_decision(dense, self.config)
        self.assertEqual(dense["decision"], "keep")
        self.assertNotIn("D6", dense["rule_ids"])
        self.assertEqual(sparse["decision"], "drop")
        self.assertIn("D6", sparse["rule_ids"])

    def test_three_way_redundancy_keeps_the_best(self):
        members = [
            row(feature_name="a", auc=0.70, auc_effective=0.70),
            row(feature_name="b", auc=0.62, auc_effective=0.62),
            row(feature_name="c", auc=0.58, auc_effective=0.58),
        ]
        for member in members:
            member["hits"] = single_feature_hits(member, self.config)
        pairs = []
        names = ["a", "b", "c"]
        for left, right in (("a", "b"), ("a", "c"), ("b", "c")):
            pairs.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "method": "spearman",
                    "coefficient": 0.95,
                    "kept_feature": "",
                    "dropped_feature": "",
                }
            )
        apply_redundancy(members, pairs, self.config)
        for member in members:
            finalize_decision(member, self.config)
        decisions = {member["feature_name"]: member["decision"] for member in members}
        self.assertEqual(decisions["a"], "keep")
        self.assertEqual(decisions["b"], "drop")
        self.assertEqual(decisions["c"], "drop")
        self.assertEqual(names[0], "a")

    def test_raise_keep_threshold_without_touching_rules(self):
        config = load_config(CONFIG)
        config.thresholds = dict(config.thresholds)
        config.thresholds["auc_keep"] = 0.62
        result = decide(row(auc=0.60, auc_effective=0.60), config)
        self.assertEqual(result["decision"], "conditional_keep")
        self.assertIn("C3", result["rule_ids"])

    def test_missing_threshold_key_fails(self):
        broken = ROOT / "output" / "_broken_config.yaml"
        broken.parent.mkdir(exist_ok=True)
        broken.write_text(
            "inputs: {schema: a.csv, stats: b.csv, samples: c.csv}\n"
            "columns: {id: sample_id, label: label, exclude: [sample_id, label]}\n"
            "metrics: {completeness: true, validity: true, discrimination: true, usability: true, redundancy: true, iv: true, mutual_info: true}\n"
            "auc: {mode: effective}\n"
            "thresholds: {missing_drop: 0.5}\n",
            encoding="utf-8",
        )
        with self.assertRaises(SystemExit):
            load_config(broken)
        broken.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
