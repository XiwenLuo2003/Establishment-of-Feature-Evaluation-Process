# 特征评估流程设计说明

本文档描述必做功能的设计：评估什么、按什么结构计算、字段如何定义、怎样给出保留或淘汰结论。实现以本文档为准。换一批特征时，只替换输入文件并调整配置，不改评估代码。

## 1. 目标与边界

流程读入三份离线文件，对每一个候选特征给出指标、质量标记、决策和理由，并写出一份可读报告。

纳入评估的列来自 `feature_schema.csv`。`sample_id` 是样本主键，`label` 是排序标签，二者都不是候选特征。

本流程做筛选，不训练排序模型，不产出上线特征服务。采样表是指标计算的事实来源；统计摘要用于对照，发现摘要与采样不一致时写入报告。

## 2. 架构设计

### 2.1 流水线

```text
feature_schema.csv ─┐
feature_stats.csv  ─┼─> 装载校验 ─> 单特征评估 ─> 冗余分析 ─> 决策 ─> 评估表 + 报告
feature_samples.csv┘         │            │            │         │
                             │            │            │         └─ 规则与阈值（config.yaml）
                             └─ 列对齐、行数、类型
```

各阶段只消费上一阶段的表，不回写原始 CSV。

| 阶段 | 职责 | 产出 |
|---|---|---|
| 装载校验 | 读三张表，统一空值，对齐特征清单，核对行数与类型 | 特征清单、样本表、校验告警 |
| 单特征评估 | 按类型计算质量指标和区分度指标 | 每个特征一行的指标记录 |
| 冗余分析 | 数值特征做 Spearman，类别特征做 Cramér's V | 特征对相关表 |
| 决策 | 按固定优先级套用规则 | 决策、命中规则、建议动作 |
| 报告 | 汇总决策、质量问题和复现信息 | `report.md` 与结果 CSV |

### 2.2 模块

| 模块 | 职责 |
|---|---|
| `loader` | 读 CSV，把空串视为缺失，解析数值，记录实际行数 |
| `catalog` | 以 schema 为主表，左连接 stats，再与采样列名核对 |
| `quality` | 缺失、取值范围、哨兵值、截断、基数、头部覆盖 |
| `signal` | 数值 AUC / KS；类别 IV / 互信息；二元类别补算 AUC |
| `redundancy` | 两两相关，只在双方都未因质量或无信号被淘汰时比较 |
| `decision` | 规则引擎，输出 `keep` / `conditional_keep` / `drop` |
| `report` | 生成 Markdown 报告和明细 CSV |
| `config` | 阈值、业务取值范围、哨兵值、排除列 |

入口是一条命令：`python -m feature_eval --data-dir <目录> --config config.yaml`。随机过程只出现在等频分箱的边界并列值上，分箱使用固定分位点，不抽样，结果可复现。

### 2.3 技术选型

使用 Python 3、pandas、numpy、scikit-learn。40 000 行、15 个特征可以在单机一次性算完。AUC 用 `roc_auc_score`，互信息用 `mutual_info_classif`。不引入训练框架。

### 2.4 可复用性

下列内容全部放在 `config.yaml`，换特征时默认不改代码：

- 标签列、主键列、排除列
- 缺失、AUC、IV、基数、相关的阈值
- 各数值特征的业务取值范围和哨兵值列表
- 输入文件名

代码只依赖三张表的契约：schema 至少有特征名、类型、描述；stats 至少有特征名和本文数据字典中的统计列；samples 至少有主键、标签和 schema 中的特征列。多出来的特征列记入告警并照常评估；schema 有而采样没有的特征直接淘汰，理由是无法复核。

## 3. 指标体系

指标分五组。前四组按特征独立计算，第五组在特征之间计算。

### 3.1 完整性

| 指标 | 定义 | 用途 |
|---|---|---|
| `missing_rate` | 采样中空值数 / 样本数 | 缺失过高则不可入模 |
| `non_null_count` | 非空样本数 | AUC、IV 的样本量 |

空串、仅空白、无法解析的数值都计为缺失。`user_realtime_intention` 在摘要中缺失率 0.8006，按缺失规则会进入淘汰候选。

缺失率在 5% 到 50% 之间时，额外计算「是否缺失」这个 0/1 指示与标签的 AUC。指示本身有区分度时，建议保留原特征并同时使用缺失指示。缺失率达到 50% 及以上时，不再用指示补救。

### 3.2 取值合法性

数值特征按配置的业务区间判断越界，并单独统计哨兵值。

| 特征 | 合法区间 | 哨兵值 | 说明 |
|---|---|---|---|
| `user_click_rate_7d` | [0, 1] | 无 | 点击率 |
| `user_duration_avg_7d` | [0, 86400] | 无 | 秒；最小值堆在 1 时记截断 |
| `item_ctr_7d` | [0, 1] | 无 | 物料点击率 |
| `item_author_fans` | [0, 1e8] | 9999999 | 粉丝数长尾 |
| `user_last_active_gap` | [0, 3650] | 无 | 天 |
| `cross_user_item_match` | [0, 1] | 无 | 匹配度 |
| `item_quality_score` | [0, 1] | 999999 | 质量分应为 0~1 |
| `user_realtime_intention` | [0, 1] | 无 | 意图强度 |
| `item_freshness_hour` | [0, 87600] | 无 | 小时 |

`sentinel_rate` 是哨兵值占非空样本的比例。`out_of_range_rate` 是落在合法区间外、且不属于哨兵列表的比例。摘要里 `item_author_fans` 最大值 9999999、`item_quality_score` 最大值 999999，而两者的中位数都远小于这些上限，流水线把这类点记为哨兵，区分度在剔除哨兵后重算。

若某数值特征的最小值等于 P25，且该取值占比达到 20% 及以上，记 `left_censored=true`。`user_duration_avg_7d` 的最小值和 P25 都是 1，符合这一检查的触发条件，最终以采样占比确认。截断只作为标记和建议，不单独导致淘汰。

摘要中的均值、标准差在强偏态下会失真（粉丝数偏度 9.89，质量分偏度 9.91）。报告中的分布使用采样重算的分位数；摘要均值只作对照。

### 3.3 区分度

标签按二分类处理。指标只在特征非空、标签非空的样本上计算。

数值特征：

| 指标 | 定义 | 用途 |
|---|---|---|
| `auc` | 非空样本上的 ROC AUC | 与摘要中的 `auc` 对照 |
| `auc_effective` | `max(auc, 1-auc)` | 反向特征仍保留区分能力 |
| `direction` | `auc >= 0.5` 为 `positive`，否则 `negative` | 入模时的方向 |
| `ks` | 正负样本累积分布差的最大值 | 辅助观察分离程度 |
| `auc_gap` | 采样 AUC 与摘要 AUC 的绝对差 | 大于 0.03 记不一致 |

`user_last_active_gap` 摘要 AUC 0.3990，`item_freshness_hour` 摘要 AUC 0.4193。两者的有效 AUC 分别为约 0.60 和约 0.58，方向记为负向：间隔越短、内容越新，越倾向正样本。决策使用有效 AUC，报告同时给出原始 AUC 和建议方向。

类别特征：

| 指标 | 定义 | 用途 |
|---|---|---|
| `iv` | 等频不适用；按取值分组的 WOE 信息值 | 主区分度 |
| `mutual_info` | 与标签的互信息 | 与 IV 交叉验证 |
| `auc` | 仅当非空取值不超过 2 个时，按数值再算 AUC | 覆盖 `user_pay_flag` 这类 0/1 字段 |

IV 的单箱样本数不足 30 时，把尾部取值并入 `__other__` 后再算，避免稀有取值把 IV 抬高。

摘要没有类别特征的 AUC，类别结论以采样计算的 IV 为准。

### 3.4 可用性

| 指标 | 适用 | 用途 |
|---|---|---|
| `unique_count` | 全部 | 常量、基数 |
| `top1_ratio` | 类别 | 接近 1 视为常量 |
| `top5_coverage` | 类别 | 头部覆盖过低视为过稀疏 |
| `cardinality_ratio` | 类别 | 唯一值数 / 样本数 |

`item_tag` 有 4836 个取值，Top5 覆盖 0.0268，不能作为原始类别入模。`user_interest_tag` 有 198 个取值，基数可接受，但需要编码而不是独热。`user_city_level`、`item_category`、`user_device_brand` 取值个数都在 10 以内。

### 3.5 冗余

| 指标 | 对象 | 规则用阈值 |
|---|---|---|
| Spearman 绝对值 | 数值 × 数值 | 0.90 |
| Cramér's V | 类别 × 类别 | 0.90 |

相关在非空成对样本上计算。只对「尚未因缺失、无信号、基数被淘汰」的特征做冗余淘汰，避免用一个已淘汰特征和另一个特征互相解释。相关达到阈值时，保留有效区分度更高的一个；区分度差小于 0.01 时，保留缺失率更低的一个。

数值与类别之间不做自动淘汰。报告列出最高相关对，供人工判断。

## 4. 淘汰与保留逻辑

每条规则有固定编号。一个特征可以命中多条规则，决策按下面的优先级取最严的一档，全部命中规则都写入理由。

优先级：`drop` > `conditional_keep` > `keep`。

### 4.1 淘汰 `drop`

命中任意一条即淘汰。

| 规则 | 条件 | 理由模板 |
|---|---|---|
| D1 | `missing_rate >= 0.50` | 缺失率过高 |
| D2 | `unique_count <= 1`，或类别 `top1_ratio >= 0.99` | 近似常量 |
| D3 | 数值特征剔除哨兵后 `auc_effective < 0.53` | 区分度不足 |
| D4 | 类别特征 `iv < 0.02` | 区分度不足 |
| D5 | 类别 `unique_count >= 1000`，或 `top5_coverage < 0.05` | 过稀疏，原始类别不能稳定入模 |
| D6 | 与仍保留的特征相关绝对值 `>= 0.90`，且对方区分度更高 | 与某特征冗余 |
| D7 | `sentinel_rate + out_of_range_rate >= 0.05` | 非法取值占比过高，分布不可用 |
| D8 | schema 中存在，采样中不存在 | 无法复核 |

D3、D4 使用清洗后的指标：先去掉哨兵和越界值，再算 AUC 或 IV。因此「又脏又没有信号」会同时命中质量和区分度规则。

### 4.2 有条件保留 `conditional_keep`

未命中淘汰规则时，命中任意一条则定为有条件保留，并给出入模前动作。

| 规则 | 条件 | 建议动作 |
|---|---|---|
| C1 | 原始 `auc < 0.5` 且 `auc_effective >= 0.55` | 按负向使用，或取相反数 |
| C2 | 哨兵或越界占比在 `(0, 0.05)`，剔除后仍满足保留线 | 入模前把哨兵置空或截断到合法区间 |
| C3 | `0.53 <= auc_effective < 0.55`，或类别 `0.02 <= iv < 0.10` | 弱信号，进入模型后做消融，不作为优先特征 |
| C4 | 类别唯一值个数在 `[50, 1000)` | 使用目标编码或哈希，不做独热 |
| C5 | `0.05 <= missing_rate < 0.50`，且非空部分达到保留线 | 同时加入缺失指示 |
| C6 | `left_censored=true` 且区分度达到保留线 | 保留，并在报告中说明低端截断 |

`user_pay_flag` 虽在 schema 中标为类别，取值只有 0/1，同时走类别 IV 和数值 AUC。决策取两套规则中更严的一档，报告写明两条指标。

### 4.3 保留 `keep`

未命中 D 规则和 C 规则，且满足：

- 数值：清洗后 `auc_effective >= 0.55`
- 类别：`iv >= 0.10`，唯一值个数小于 50，Top1 占比小于 0.99

### 4.4 由摘要可以直接预判的部分

下面是根据 `feature_stats.csv` 做出的预判。最终结论以采样重算为准，预判不写进结果文件。

| 特征 | 预判 | 依据 |
|---|---|---|
| `user_realtime_intention` | 淘汰 | 缺失率 0.8006，摘要 AUC 0.4926 |
| `item_author_fans` | 淘汰 | 摘要 AUC 0.4982，且存在 9999999 |
| `item_tag` | 淘汰 | 4836 个取值，Top5 覆盖 0.0268 |
| `item_quality_score` | 有条件保留倾向 | 摘要 AUC 0.5561，但最大值 999999，需剔除后重算 |
| `user_last_active_gap` | 有条件保留倾向 | 摘要 AUC 0.3990，有效区分度约 0.60，负向 |
| `item_freshness_hour` | 有条件保留倾向 | 摘要 AUC 0.4193，有效区分度约 0.58，负向 |
| `user_click_rate_7d` | 保留倾向 | 无缺失，摘要 AUC 0.6096，取值在 0~1 |
| `item_ctr_7d` | 保留倾向 | 无缺失，摘要 AUC 0.6292 |
| `cross_user_item_match` | 保留倾向 | 无缺失，摘要 AUC 0.5971 |
| `user_duration_avg_7d` | 保留或有条件保留 | 摘要 AUC 0.5599；若确认低端截断则命中 C6 |

`user_city_level`、`item_category`、`user_device_brand`、`user_interest_tag`、`user_pay_flag` 要等 IV 算出后再决策。

## 5. 字段设计

### 5.1 输入契约

`feature_schema.csv`

| 列 | 必需 | 说明 |
|---|---|---|
| `feature_name` | 是 | 与采样列名、stats 特征名一致 |
| `feature_type` | 是 | `numeric` 或 `categorical` |
| `description` | 否 | 写入报告，不参与计算 |

`feature_stats.csv` 的列见第 6 节。缺列时对应对照项记为空，不中断流程。

`feature_samples.csv`

| 列 | 必需 | 说明 |
|---|---|---|
| `sample_id` | 是 | 主键，不参与评估 |
| `label` | 是 | 0/1 标签 |
| 其余列 | 与 schema 对应 | 候选特征 |

### 5.2 评估结果 `feature_evaluation.csv`

一行一个特征。

| 列 | 类型 | 说明 |
|---|---|---|
| `feature_name` | string | 特征名 |
| `feature_type` | string | `numeric` / `categorical` |
| `description` | string | 业务含义 |
| `sample_count` | int | 实际读入样本数 |
| `non_null_count` | int | 非空数 |
| `missing_rate` | float | 缺失率 |
| `unique_count` | int | 非空唯一值个数 |
| `mean` `std` `min` `p01` `p25` `p50` `p75` `p99` `max` | float | 数值特征的采样分布，类别为空 |
| `skew` | float | 采样偏度 |
| `out_of_range_rate` | float | 越界占比 |
| `sentinel_rate` | float | 哨兵占比 |
| `left_censored` | bool | 低端是否截断 |
| `top1_value` | string | 类别最高频取值 |
| `top1_ratio` | float | 最高频占比 |
| `top5_coverage` | float | 前五取值覆盖 |
| `auc` | float | 采样 AUC，类别高基数为空 |
| `auc_stats` | float | 摘要 AUC |
| `auc_effective` | float | 有效 AUC |
| `auc_gap` | float | 与摘要 AUC 的绝对差 |
| `direction` | string | `positive` / `negative` / 空 |
| `ks` | float | KS |
| `iv` | float | 信息值 |
| `mutual_info` | float | 互信息 |
| `partner_feature` | string | 冗余对比中更优的对方特征 |
| `partner_corr` | float | 与对方的相关绝对值 |
| `flags` | string | 质量标记，分号分隔 |
| `decision` | string | `keep` / `conditional_keep` / `drop` |
| `rule_ids` | string | 命中规则，如 `C1;C2` |
| `reasons` | string | 可读理由，分号分隔 |
| `action` | string | 入模前动作；保留且无附加动作时为空 |

`flags` 的取值：`high_missing`、`sentinel`、`out_of_range`、`left_censored`、`high_cardinality`、`sparse`、`near_constant`、`reverse_direction`、`weak_signal`、`stats_mismatch`、`missing_in_samples`。

### 5.3 冗余明细 `feature_redundancy.csv`

只保留相关绝对值达到 0.70 的特征对，便于报告展示；淘汰仍只在 0.90 生效。

| 列 | 说明 |
|---|---|
| `feature_a` `feature_b` | 特征对，按名字排序，保证一对只出现一次 |
| `method` | `spearman` 或 `cramers_v` |
| `coefficient` | 相关绝对值 |
| `kept_feature` | 若触发 D6，保留的一方；未触发则为空 |
| `dropped_feature` | 若触发 D6，淘汰的一方 |

### 5.4 校验告警 `validation_warnings.csv`

| 列 | 说明 |
|---|---|
| `level` | `info` / `warning` |
| `code` | 如 `row_count_mismatch`、`auc_mismatch`、`unknown_column` |
| `feature_name` | 可空 |
| `message` | 具体差异，例如摘要 40000 行、采样实际行数 |

已知需要核对的一项：题目描述采样约 5 万条，摘要 `total_count` 为 40000，采样文件行数约 40002（含表头）。装载时写下实际数据行数，并与 `total_count` 比较。

## 6. 数据字典

### 6.1 候选特征

| 特征 | 类型 | 含义 | 评估分支 |
|---|---|---|---|
| `user_click_rate_7d` | numeric | 用户近 7 天点击率 | 数值 |
| `user_duration_avg_7d` | numeric | 用户近 7 天平均停留时长（秒） | 数值 |
| `item_ctr_7d` | numeric | 物料近 7 天点击率 | 数值 |
| `item_author_fans` | numeric | 作者粉丝数 | 数值 |
| `user_last_active_gap` | numeric | 距上次活跃天数 | 数值 |
| `cross_user_item_match` | numeric | 用户与物料标签匹配度 | 数值 |
| `user_city_level` | categorical | 城市等级 | 类别 |
| `item_category` | categorical | 物料一级品类 | 类别 |
| `item_tag` | categorical | 物料细粒度标签 | 类别 |
| `user_device_brand` | categorical | 设备品牌 | 类别 |
| `user_interest_tag` | categorical | 用户兴趣标签 | 类别 |
| `user_pay_flag` | categorical | 付费标识 0/1 | 类别 + 二元 AUC |
| `item_quality_score` | numeric | 物料质量分 | 数值 |
| `user_realtime_intention` | numeric | 实时意图强度 | 数值 |
| `item_freshness_hour` | numeric | 发布距今小时数 | 数值 |

### 6.2 统计摘要列

| 列 | 含义 | 在流程中的用法 |
|---|---|---|
| `feature_name` | 特征名 | 关联键 |
| `feature_type` | 类型 | 与 schema 不一致时记告警，计算以 schema 为准 |
| `total_count` | 摘要样本数 | 与采样行数对照 |
| `non_null_count` | 摘要非空数 | 对照 |
| `missing_rate` | 摘要缺失率 | 对照，决策用采样缺失率 |
| `unique_count` | 摘要唯一值 | 对照 |
| `mean` `std` `min` `p25` `p50` `p75` `max` `skew` | 分布 | 对照；强偏态时不以摘要均值为准 |
| `auc` | 摘要 AUC | 写入 `auc_stats`，计算 `auc_gap` |
| `top1_value` `top1_ratio` `top5_coverage` | 类别头部 | 对照，决策用采样重算值 |

### 6.3 决策枚举

| 值 | 含义 |
|---|---|
| `keep` | 可直接加入排序模型 |
| `conditional_keep` | 可以加入，但必须先完成 `action` 中的处理 |
| `drop` | 不加入本次模型 |

## 7. 报告内容

`report.md` 按以下顺序书写，保证不打开 CSV 也能讲清结论。

1. 运行信息：数据目录、样本行数、正样本占比、配置中的阈值。
2. 结论汇总：保留、有条件保留、淘汰的数量和特征名单。
3. 特征明细：每个特征的关键指标、决策、规则编号和理由。
4. 数据质量：缺失、哨兵、越界、截断、摘要与采样差异。
5. 冗余：系数达到 0.70 的特征对，以及因此淘汰的特征。
6. 复现方式：命令行和配置文件路径。

## 8. 关键设计决策

1. **用有效 AUC 而不是原始 AUC 做淘汰。** 活跃间隔和新鲜度的摘要 AUC 低于 0.5，与标签方向相反，仍可能有用。原始方向写进 `direction`，避免入模时用反。
2. **区分度在剔除哨兵后计算。** 质量分和粉丝数的均值被极大值拉动。若先用脏分布算指标，会把「少量哨兵」和「特征本身无信号」混在一起。
3. **缺失、无信号、高基数优先于相关。** 冗余只在两个都还能用的特征之间取舍。
4. **类别主指标用 IV，数值主指标用 AUC。** 摘要已经给出数值 AUC，采样重算后可以对照。类别没有现成 AUC，IV 直接按取值评估，不依赖编码。二元类别补 AUC，是为了让 `user_pay_flag` 这类字段和数值规则对齐。
5. **阈值写进配置。** 0.50、0.53、0.55、0.02、0.10、1000、0.90 都是本次业务上的筛选线，不是代码常量。下次换特征只改配置。
6. **采样是计算来源，摘要是对照。** 题目给出的摘要可能和采样行数、极值不一致。决策数字来自采样，差异进入告警。

## 9. 建议目录

```text
特征评估流程搭建/
├── DESIGN.md
├── RULE.md
├── TEST.md
├── PROBLEM.md
├── README.md
├── requirements.txt
├── config.yaml
├── feature_schema.csv
├── feature_stats.csv
├── feature_samples.csv
├── feature_eval/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── loader.py
│   ├── evaluate.py
│   ├── signal.py
│   ├── redundancy.py
│   ├── decision.py
│   ├── report.py
│   └── pipeline.py
├── tests/
│   └── test_decision.py
└── output/
    ├── feature_evaluation.csv
    ├── feature_redundancy.csv
    ├── validation_warnings.csv
    └── report.md
```

`README.md` 在实现阶段编写，说明环境、命令和输出位置。
