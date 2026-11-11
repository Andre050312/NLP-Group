# 中性方面的情感干扰：实验与报告指南

## 当前状态

这部分研究的问题是：**其他方面的正负评价，是否干扰了目标方面的中性判断？**

已实现诊断工具、40 对合成候选样例和移除 CLS 特征的消融开关。
尚未拿到组内真实模型的 checkpoint、逐条预测或原始 PR 数据，因此本文件不填造准确率、显著性或改进结论。
GitHub 核查基于 `41fd25405c00b7bf00dcd6512da48e6c22dcad84`：只有 main，没有 PR，未提交所述 bias/gate 诊断结果。

研究属于给定正确方面的情感三分类。不要与自动方面抽取或端到端 F1 混用。
保留公开数据原始三分类标签，不把冲突或标注不确定改成中性。

## 1. 用已有预测生成统一报告

先激活项目环境，从仓库根目录运行。以下 `runs/同学的实验目录` 必须换成实际路径。
支持增强版 `*_gold_predictions.jsonl`，以及基线的 `predictions.jsonl`。
需要完整导出的所有方面；只给错例会扭曲指标与邻近方面统计。

```powershell
python -m neutral_diagnostics analyze --predictions runs/同学的实验目录/test_gold_predictions.jsonl --gold-xml Database/Restaurant3/test.xml --dataset restaurant --split test --run-id pair_seed42 --model-label pair_control --training-seed 42 --output runs/analysis_pair_seed42_test
```

输出：

| 文件 | 用途 |
|---|---|
| report.md | Accuracy、Macro-F1、每类 P/R/F1/support 与中性分组的可读表格 |
| analysis.json | 完整指标、混淆矩阵、输入 SHA-256、评价目标指纹、置信度与 bootstrap 区间 |
| neutral_pr_probability.csv | 中性对其余类，以 p(neutral) 排序的完整 PR 点与 AP |
| neutral_pr_logit_margin.csv | 以 log p(neutral) − max(log p(negative), log p(positive)) 排序的 PR 点与 AP |
| neutral_pr.svg | 可独立打开的两种排序 PR 曲线，附中性占比基线，可用于报告插图 |
| neutral_errors_for_review.csv | 中性错例、最近极性方面、匹配情况和人工归因空列 |

logit margin 对应对中性 logit 加 bias 后与另外两类 argmax 竞争的分数。
概率为 0 时用 1e-30 截断，避免取对数的数值错误；极端饱和概率的原始 logit 排序无法从 JSONL 完全恢复。
工具不在测试集上推荐最佳阈值。不得把局部平坦的 PR 曲线直接解释为排序无效。

中性分组定义：同一句里是否存在另一个标注为正面/负面的方面。
“全部标注方面中性”包含只有一个中性方面的句子，不表示整句一定没有其他未标注的情感。
错例与邻近方面标签一致仅是关联；如果邻居同时含正负标签，则“匹配任一邻居”没有区分力。
分别报告匹配数/全部中性错例，以及匹配数/存在极性邻居的中性错例。
人工检查时填写 `reviewed_reason`、`reviewer`，区分方面干扰、否定、隐含态度、歧义、标注疑点和其他错误。

默认进行 1000 次按**相同句子文本**分组的 percentile bootstrap，输出 95% 区间。
它只描述当前固定模型的抽样不确定性；不包含训练种子、验证划分或同评论不同句的相关性。
不会产生“数据集性能上限”或“因果归因”的结论。

验证集使用同一 run-id，另设 `--split validation`。如果没有该文件，可从 history 中取得相应验证指标；
或对真实训练时的验证 XML 运行 evaluate。不能重新随机切一个验证集替代原划分。
对其他数据集改用对应 gold XML；对 OOF 用 `--split oof`，不传原始 train XML，因为 CV 已排除预留集合重叠样本。

```powershell
python -m neutral_diagnostics compare --analyses runs/analysis_pair_seed42_validation/analysis.json runs/analysis_pair_seed42_test/analysis.json --output runs/analysis_comparison
```

compare 输出逐次结果、按模型/数据/划分分组的均值与样本标准差，以及同一 run-id 的验证减测试差值。
同一数据集/划分的评价目标不一致会报错。多个重复实验必须给出不同的真实 training-seed；
一次结果不产生标准差。不同 split 的 gap 只是描述，不能据此认定系统性偏差。
若使用多个验证划分，请在 dataset 名中明确划分标识并分开解释，不能冒充固定划分的训练随机性实验。

## 2. 40 对控制样例

样例在 `data/diagnostics/neutral_scope_pairs.csv`：30 对中性目标、5 对正面和5 对负面目标作对照，共80句。
每对只把另一个方面的 `excellent` 改为 `awful`，目标方面的表述不变，目标在前/在后交替。
这是为本项目起草的合成诊断集，不是官方测试数据，不代表自然评论的完整分布。

```powershell
python -m neutral_diagnostics validate-pairs
```

所有样例初始为 `pending`。请由两位同学独立核查目标标签、干扰方面、自然程度及目标含义是否保持不变，
讨论分歧后将通过的行改为 `approved` 并填写真实 `reviewer`；不合适的行设为 `rejected`。
保留两人独立标注记录，以实际复核后的 CSV 为准，不把模型起草的候选标签写成人工金标准。
工具检查机械条件，但不能代替语义复核。

有真实模型后：

```powershell
python -m neutral_diagnostics predict-pairs --checkpoint runs/同学的实验目录/best --output runs/scope_pair_control --device auto
```

支持本项目保存的 BERT 基线、pair、joint checkpoint，均只评价提供方面的分类。
未经复核的探索性检查可增加 `--include-unreviewed`；结果会明确记录 `human_review_complete: false`。
rejected 行始终排除。输出 pair_metrics.json、pair_results.csv、predictions.jsonl。

关注三个量：

1. 两句是否都预测正确（both_correct）。
2. 不相关方面极性翻转后，目标预测是否保持不变（invariant）以及中性概率变化。
3. 中性目标是否出现“正面邻居时判正面、负面邻居时判负面”（follows_distractor_polarity）。

必须同时报告正负目标对照。永远输出中性的模型也能获得很高的不变率，不能只展示 invariant。
该实验提供受控输入干预证据，仍不能单凭它证明模型内部只读取句级极性。
若根据这些例子选择模型，把它们称为开发诊断集；最终确认应使用新起草且未用于选模型的样例。

## 3. 一个明确的消融：去掉分类头的 CLS 特征

新增 `--no-cls-feature`，pair/joint/train/cross-validate 均支持。默认仍使用 CLS。
原 checkpoint 缺少 use_cls 字段时按 True 加载，保持旧模型兼容。
新模型将 use_cls 写进 settings.json，推理自动还原。

变化只在分类头：pair 使用“评论中的目标表示 + 第二段方面表示”，joint 使用“目标表示 + 方面引导上下文”。
编码器的 CLS token 和注意力仍存在；方面 token 自身也能编码全局信息，因此这个消融**没有隔绝全句信息**。
分类头参数数目随输入维度减少，应同时报告 parameters。不能把结果完全归因于去掉某个捷径。

建议先使用当前主数据 restaurant，进行已有的3折分组交叉评价，原官方测试仅用于预先排除重叠：

```powershell
python -m absa_enhanced cross-validate --mode pair --dataset restaurant --output runs/scope_cv_control_seed42 --folds 3 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_macro_f1 --plan-only
python -m absa_enhanced cross-validate --mode pair --dataset restaurant --output runs/scope_cv_no_cls_seed42 --folds 3 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_macro_f1 --no-cls-feature --plan-only
```

核查两组 fold_assignment.json 完全一致。实际训练时把 `--plan-only` 替换为 `--resume`，其余参数保持一致。
总计6次训练；正式运行需预训练模型与相应计算资源。本次交付不声称已完成真实 DeBERTa 训练。
CV 的每折内层验证用于选轮次，外层留出用于评价；不要根据外层结果反复搜索大量设置。
oof_predictions.jsonl 可用 analyze 分析，split 设为 oof。
如果还需要训练随机性验证，固定 split-seed 42，另跑 seed 43/44，分别保存目录并预先固定方案。
固定划分多种子与多折评价回答不同问题，不混为一谈。

官方测试集已用于多轮讨论时，应在报告说明探索过程；不能把它重新描述为从未使用的最终独立测试。
2015/2016 派生测试集中性仅27/28个，2014为196个；跨版本成绩必须注明样本量和转换协议。

## 4. 报告可采用的表述与待填结果

已实测的数字（control / no CLS / gate / ordinal / clause、阈值上限、验证-测试差距、40 对受控样例）见 [中性诊断实测结果](neutral_results.md)，可按该文件填写下表。

方法表述：

> 我们研究其他方面情感对中性目标判断的干扰。除三分类整体及各类指标外，按同句是否含其他极性方面统计中性召回率，并进行控制其他方面极性、保持目标描述不变的成对诊断。我们增加移除分类头 CLS 特征的消融，使用相同分组划分和模型选择规则比较。

结果表由真实运行后填写，当前均为待测：

| 模型 | 数据/划分 | Macro-F1 | 中性 P/R/F1 | 中性 support | 成对均正确率 | 正负对照均正确率 |
|---|---|---|---|---|---|---|
| control | 待填 | 待测 | 待测 | 待核对 | 待测 | 待测 |
| no CLS | 同 control | 待测 | 待测 | 必须相同 | 待测 | 待测 |

如果原先 gate 结果复核成立，可写“该配置在当前划分上的验证收益未迁移至测试集”。
如果阈值收益小，可写“已测试范围内阈值调整收益有限”。
如果同句干扰和受控实验均支持假设，可写“结果支持跨方面情感干扰的解释”。
不要写“任何参数化都救不了”“中性 F1 0.65 是数据集上限”或“置信度高证明模型确信地读句级情感”。

## 5. 个人贡献记录

可在 Author Contributions 中按实际完成情况写：

> [姓名]负责中性类别的错误诊断与评价协议，起草并组织人工复核控制样例，实现统一指标、PR 排序与置信度分析，设计 CLS 特征消融，完成 [实际运行的实验] 并分析 [实际观察]。

不要把尚未跑的实验写成已完成，也不要把同学训练的底座归为自己的实现。
按课程要求说明工具辅助情况，保留自己对标注、实验和结论的理解与复核。

## 参考

- SemEval 原论文：https://aclanthology.org/S14-2004/
- 方面作用域干扰：https://aclanthology.org/2023.findings-emnlp.727/
- 校准：https://proceedings.mlr.press/v70/guo17a.html
- 模型选择偏差：https://jmlr.org/papers/v11/cawley10a.html
- PR 与 AP 定义：https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

正式报告将上述来源整理为课程要求的 IEEE 格式。
