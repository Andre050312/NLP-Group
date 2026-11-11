# 中性诊断：实测结果（Restaurant3，split-seed 42）

本文只记录**已经跑出来**的数字，供 [实验指南](neutral_scope_study.md) 的结果表填写。所有配置使用同一套 1120 个 gold 方面（同一 `--split-seed 42`），差异只来自一个因素。

固定设置：`--mode pair --dataset restaurant --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --split-seed 42 --class-weighting none --selection-metric gold_accuracy`。按验证 Accuracy 选模，测试集只评一次。GPU 为 RTX 5060 Laptop 8GB，BF16，单次训练约 18 分钟。

`control` 与 `--no-cls-feature` 各有 **seed 42 / 43 / 44 三次训练**，因此这两条线可以给出种子方差和配对比较；`gate` / `ordinal` / `clause` 只有 seed 42，**不能与前三者直接比较名次**（见第 3b 节）。

## 1. 主表（单种子，仅作参照）

| 配置 | Acc | Macro-F1 | 中性 P | 中性 R | 中性 F1 | 中性 F1 95% CI（句级 bootstrap） | 中性 AP |
|---|---|---|---|---|---|---|---|
| control（含 CLS） | 0.8786 | 0.8076 | 0.782 | 0.531 | 0.632 | [0.552, 0.704] | 0.725 |
| no CLS（`--no-cls-feature`） | 0.8786 | 0.8053 | 0.790 | 0.500 | 0.613 | [0.529, 0.685] | **0.745** |
| gate 头 | 0.8723 | 0.8018 | 0.709 | 0.536 | 0.610 | [0.527, 0.683] | 0.712 |
| ordinal 头 | 0.8688 | 0.7923 | 0.697 | 0.515 | 0.592 | [0.508, 0.663] | 0.687 |
| clause 上下文 | 0.8429 | 0.7438 | 0.727 | 0.408 | 0.523 | [0.442, 0.599] | 0.648 |

**这一整张表都是 seed 42 单个种子**，不能当成配置之间的定论：control 与 no CLS 已有 3 种子（第 3a、3b 节），`gate` / `ordinal` / `clause` 只有 1 个种子。五个配置的中性 F1 置信区间全部互相重叠；种子方差（0.025–0.032）大于其中四个配置之间的差异（见 3b）。本表只用于看量级和方向。

中性 support 一律 196。中性 AP 的随机基线是中性的占比 0.175，所以 0.65–0.75 说明排序信号远强于随机，不是"没有信息"。

## 2. 阈值能买到多少：排序已经决定了上限

对同一份打分扫全部阈值（诊断用，不在测试集上选阈值）：

| 配置 | argmax 处中性 F1 | 任意阈值可达的最高中性 F1 |
|---|---|---|
| control | 0.632 | 0.658 |
| no CLS | 0.613 | **0.689** |
| gate | 0.610 | 0.673 |
| ordinal | 0.592 | 0.656 |
| clause | 0.523 | 0.610 |

结论有两条，都很硬：

1. 对 control，任何阈值最多把中性 F1 抬到 0.658（+0.026）。这是**该 checkpoint 的排序上限**，不是数据集上限，但足以说明"再加权重/调阈值"拿不到大收益。
2. no CLS 的 argmax 中性 F1 最低，可达上限却最高（0.689）。它不是排序更差，而是**决策边界偏了**：去掉 CLS 后中性被系统性压低。这两件事需要分开处理，正好由 `--neutral-offset` 解决。

## 3. 中性偏移（验证集选 k，测试集只报一次）

两条线各自的验证集选中值：

| 配置 | seed | 选出的 k | Acc（k=1 → k） | Macro-F1 | 中性 F1 |
|---|---|---|---|---|---|
| control | 42 | 2.50 | 0.8786 → 0.8804 | 0.8076 → 0.8114 | 0.632 → 0.641 |
| control | 43 | 1.50 | 0.8830 → 0.8821 | 0.8159 → 0.8166 | 0.640 → 0.645 |
| control | 44 | 1.25 | 0.8670 → 0.8661 | 0.7909 → 0.7915 | 0.595 → 0.598 |
| no CLS | 42 | 2.00 | 0.8786 → 0.8848 | 0.8053 → 0.8192 | 0.613 → 0.644 |
| no CLS | 43 | 1.00 | 0.8884 → 0.8884 | 0.8278 → 0.8278 | 0.669 → 0.669 |
| no CLS | 44 | 2.00 | 0.8679 → 0.8705 | 0.7955 → 0.8039 | 0.591 → 0.617 |

汇总（均值 ± 样本标准差，n = 3 个训练种子）：

| 配置 | 指标 | 未加偏移 | 加上偏移 |
|---|---|---|---|
| control | Macro-F1 | 0.8048 ± 0.0128 | 0.8065 ± 0.0132 |
| control | 中性 F1 | 0.6224 ± 0.0244 | 0.6281 ± 0.0259 |
| no CLS | Macro-F1 | 0.8095 ± 0.0166 | **0.8170 ± 0.0121** |
| no CLS | 中性 F1 | 0.6239 ± 0.0402 | **0.6432 ± 0.0260** |

**偏移对 no CLS 的收益明显大于对 control 的收益**（Macro-F1 +0.0075 vs +0.0017）。原因在第 2 节：去掉 CLS 后中性被系统性压低，而排序其实没问题，所以补一个偏移能兑现的余量更大；control 的中性边界本来就接近最优，偏移几乎没有空间。

偏移本身的安全性：6 个 run 里**没有一次让测试 Macro-F1 变差**（最小 +0.0000）。它修的是"决策边界随种子抖动"，并把种子间标准差压小（no CLS 0.0166 → 0.0121，中性 0.0402 → 0.0260）。

## 3a. control 与 no CLS 的配对比较

同一 split、同一批测试方面，按 seed 一一配对：

| 比较 | seed 42 | seed 43 | seed 44 | 均值 ± SD | 配对 t |
|---|---|---|---|---|---|
| Macro-F1，k=1 | −0.0023 | +0.0118 | +0.0046 | +0.0047 ± 0.0071 | +1.15 |
| 中性 F1，k=1 | −0.0197 | +0.0284 | −0.0040 | +0.0016 ± 0.0245 | +0.11 |
| 中性 AP（排序） | +0.0199 | −0.0060 | +0.0026 | +0.0055 ± 0.0132 | — |
| **Macro-F1，各自加偏移后** | **+0.0078** | **+0.0112** | **+0.0124** | **+0.0105 ± 0.0024** | **+7.65** |
| Accuracy，各自加偏移后 | +0.0045 | +0.0062 | +0.0045 | +0.0051 ± 0.0010 | +8.50 |
| 中性 F1，各自加偏移后 | +0.0030 | +0.0238 | +0.0184 | +0.0151 ± 0.0108 | +2.41 |

这张表是本轮最重要的结果，读法分三层：

1. **k=1 时两者统计上无法区分**：Macro-F1 差 +0.0047（t=1.15），中性 F1 差 +0.0016（t=0.11），AP 差 +0.0055 且符号不一致（seed 43 为负）。所以"去掉 CLS 提升了中性"这个说法**在未校准的模型上不成立**。
2. **两条线各自加验证集选的偏移后，no CLS 稳定领先**：三个种子同号，Macro-F1 差 +0.0105 ± 0.0024（t=7.65），Accuracy +0.0051 ± 0.0010（t=8.50）。领先的原因不是排序更好（AP 几乎相同），而是**no CLS 有更大的可校准余量**（第 2 节：argmax 0.613 / 上限 0.689，差距 0.076；control 是 0.632 / 0.658，差距 0.026）。
3. **但 n=3 不能做经典显著性宣称**。配对 t 依赖正态假设；换成不依赖分布的符号检验，3/3 同号的 p = 0.25。所以正确表述是"三个种子上方向一致、幅度稳定（SD 仅 0.0024），但样本量不足以给出 p<0.05 的分布无关结论"。要写进报告的话，至少补到 5 个种子。

另外一个必须承认的事实：**这条领先完全依赖中性偏移这个后处理**。不用偏移时两者等价；用偏移时 no CLS 更好。如果团队不想引入验证集调参，那 no CLS 就是"和 control 一样好、参数更少"的等价替换，而不是改进。

## 3b. 种子方差比大部分配置差异更大

| 配置（k=1） | Macro-F1 三种子 | 极差 |
|---|---|---|
| control | 0.8076 / 0.8159 / 0.7909 | 0.0250 |
| no CLS | 0.8053 / 0.8278 / 0.7955 | 0.0323 |

对比第 1 节单种子的配置差异：`gate` 相对 control 是 −0.0058，`ordinal` 是 −0.0153，`clause` 是 −0.0638。

结论：

- **`gate` / `ordinal` 与 control 的差距完全落在种子噪声里**（0.025–0.032），不能声称它们更差。
- 只有 `clause` 的 −0.0638 明显超出噪声，是唯一可以下结论的结构性差异。
- 第 1 节表格里凡是单种子的逐项比较，都需要先做多种子才能引用。

## 3c. 排序质量：no CLS 并没有把中性排得更好

| 配置（3 种子均值） | argmax 中性 F1 | 中性 AP | 任意阈值可达最高中性 F1 |
|---|---|---|---|
| control | 0.6224 | 0.7222 | 0.6572 |
| no CLS | 0.6239 | 0.7277 | 0.6761 |

AP 差 +0.0055 ± 0.0132、符号不一致；argmax 中性 F1 差 +0.0016。**两者排序能力等价。**

但"任意阈值可达上限"的差是 +0.019（no CLS 更高），且三个种子里有两次更高（0.6888 vs 0.6576、0.6602 vs 0.6194，seed 43 反了：0.6791 vs 0.6947）。这条量度对少量样本很敏感，只能当作"no CLS 的分数分布更容易被重新阈值化"的弱线索，不能当结论。

## 3d. 综合判断

| 问题 | 答案 |
|---|---|
| 去掉 CLS 让中性 F1 变高了吗？ | 未校准：没有（+0.0016，t=0.11）。校准后：高了 +0.0151，但 t=2.41 且符号检验不显著。 |
| 去掉 CLS 让中性排序变好吗？ | 没有。AP 差 +0.0055，符号不一致。 |
| 那 +0.0105 的 Macro-F1 从哪来？ | 从**可校准余量**：no CLS 的 argmax 边界被压低 0.076，control 只有 0.026，所以同一个偏移在 no CLS 上兑现更多。 |
| 能写进报告的最强表述 | "在相同分组划分和相同模型选择规则下，去掉分类头 CLS 特征不改变中性排序质量，但使决策边界更容易通过单一验证集偏移校准；三个训练种子上该组合的 Macro-F1 一致高于含 CLS 的对照（+0.0105 ± 0.0024）。" |
| 不能写的表述 | "去掉 CLS 显著提升中性 F1"、"no CLS 排序更好"、"中性 F1 0.64 是数据集上限"。 |

## 4. 同句邻近方面：相关与因果分开看

### 4.1 观测（测试集 1120 个方面）

| 中性分组 | support | control 召回 | no CLS 召回 |
|---|---|---|---|
| 全部 | 196 | 0.531 | 0.500 |
| 同句存在极性方面 | 83 | 0.458 | 0.410 |
| 同句标注全部为中性 | 113 | 0.584 | 0.566 |

中性错例与某个人工标注邻居极性相同：control 41/92（0.446），no CLS 44/98（0.449）。在"存在极性邻居"的中性错例中比例是 0.911 / 0.898。这只是关联；邻居同时含正负极性时"匹配任一邻居"没有区分力。

去掉 CLS 后这个比例几乎没变（0.446 → 0.449），所以**CLS 特征不是同句极性串扰的主要通道**。

### 4.2 受控干预（40 对合成样例，`--include-unreviewed`，40 行全部 pending）

每对只把干扰方面的 `excellent`/`nice` 换成 `awful`/`terrible`，目标方面表述不动。

| 模型 | 中性目标 不变率 | 中性目标 两句都对 | 跟随干扰极性 | 中性概率平均变化 |
|---|---|---|---|---|
| control | 0.200 | 0.200 | 0.367 | 0.354 |
| no CLS | 0.433 | 0.433 | 0.333 | 0.257 |
| clause | **1.000** | 0.867 | 0.000 | **0.000** |
| 极性对照（control） | 1.000 | 1.000 | 0.000 | 0.002 |

读法：

- control 上，只翻转**另一个方面**的一个词，目标方面的中性概率平均移动 0.354，80% 的对里预测翻转。极性对照完全不变（0.002），说明这不是模型对所有输入都敏感，而是专门对同句极性敏感。这是同句串扰的**受控输入证据**，比 4.1 的相关性更强。
- no CLS 把这个敏感度降到 0.257，不变率 0.20 → 0.433。这个方向与"CLS 携带句级捷径"一致，但只减轻一半，且第 3a 节显示未校准时测试集中性 F1 并没有因此变好 —— 所以**不能把 CLS 说成串扰的主因**。
- clause 的不变率 1.000 是**构造出来的**，不是学出来的：分句裁剪把干扰方面整段从输入里删掉，目标分句两个条件下逐 token 相同，前向完全相同（概率变化恰好 0.000）。所以它不能用来证明模型内部作用域变好，只能说明"如果输入里没有干扰，它就不会被干扰"。而真测试集上 clause 反而最差（中性 F1 0.523，Macro-F1 0.7438），说明模型确实需要跨分句语境，问题出在**归属**而不是**读取**。
- 中性目标上 `both_correct` 是 30 对里的 6 对 / 13 对，二项噪声很大，不要单独引用。

样例仍是 pending，未做双人独立复核，因此这部分只能作为开发期诊断，不能当作人工金标准结论。

## 5. 验证集与测试集

| 配置 | Macro-F1 验证−测试（seed 42） | 中性 F1 验证−测试（seed 42） | 验证中性 support | 测试中性 support |
|---|---|---|---|---|
| control | −0.0121 | +0.0530 | 113 | 196 |
| no CLS | −0.0154 | +0.0476 | 113 | 196 |

中性在验证集上被高估约 5 个点，而整体 Macro-F1 反而在测试集上更高。这正是"用验证集调中性相关超参会带乐观偏差"的具体表现：**任何只在验证集上看到的 neutral 提升，都必须拿测试集复核。**

## 6. 复现命令

```powershell
$env:HF_HOME = "D:\NLP-cache\huggingface"   # 中文路径下 SentencePiece 需要 ASCII 缓存目录

# 五条训练（--sentiment-head / --context-scope / --no-cls-feature 三选一或组合）
python -m absa_enhanced train --mode pair --dataset restaurant --output runs/pair_heads_softmax_s42 `
  --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --seed 42 --split-seed 42 `
  --class-weighting none --selection-metric gold_accuracy --device auto
# 其余把 --output 换掉并加 --no-cls-feature / --sentiment-head gate / --sentiment-head ordinal / --context-scope clause

# 多种子复现（seed 43/44，split-seed 固定 42）；两条线各跑一遍
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_neutral_seeds.ps1 -Config control -Seeds 43,44
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_neutral_seeds.ps1 -Config nocls -Seeds 43,44

# 只读已有产物的汇总（不需要 GPU）
python scripts/compare_seed_spread.py     # 两配置逐种子 + 配对差
python scripts/ranking_quality.py         # 中性 AP 与可达阈值上限
python scripts/aggregate_offset_seeds.py  # 各自加偏移后的配对比较

# 统一报告（AP、置信度、校准、句级 bootstrap）
python -m neutral_diagnostics analyze --predictions runs/pair_nocls_s42/test_gold_predictions.jsonl `
  --gold-xml Database/Restaurant3/test.xml --dataset restaurant3 --split test `
  --run-id pair_nocls_seed42 --model-label pair_no_cls --training-seed 42 --output runs/analysis_no_cls

# 阈值上限：读 neutral_pr_probability.csv 逐点算 F1 取最大

# 验证集选中性偏移 + 配对 bootstrap（--output 可分别保存两条线）
python scripts/tune_neutral_offset.py --runs runs/pair_nocls_s42 runs/pair_nocls_s43 runs/pair_nocls_s44 `
  --metric macro_f1 --bootstrap 2000 --output runs/offset_nocls.json
python scripts/tune_neutral_offset.py --runs runs/pair_heads_softmax_s42 runs/pair_ctrl_s43 runs/pair_ctrl_s44 `
  --metric macro_f1 --bootstrap 2000 --output runs/offset_control.json

# 受控样例
python -m neutral_diagnostics predict-pairs --checkpoint runs/pair_nocls_s42/best `
  --output runs/scope_pair_nocls --include-unreviewed --device auto
```

## 7. 下一步

1. ~~seed 43/44 复现~~ 已完成，见第 3 节：6 个 run 里偏移没有一次让测试变差，均值改善、方差下降。
2. ~~补 control 的 seed 43/44~~ 已完成，见第 3a 节：k=1 时两者等价；各自加偏移后 no CLS 三个种子一致领先 +0.0105 Macro-F1（t=7.65）。但 n=3 的符号检验 p=0.25，**补到 5 个种子才能做分布无关的显著性宣称**。
3. 用 `cross-validate --folds 3` 在同一分组划分上比 control / no CLS，外层留出才是无偏估计。
4. 40 对样例需要两人独立复核后转 `approved`，现在的结论只是探索性。
5. 若要把中性 F1 再往上推，需要改表征（例如按分句给注意力先验、或对每句其他方面做显式对比），而不是继续调阈值——第 2 节已经给出阈值的天花板。注意第 3b 节：任何新配置都必须至少 3 个种子才能和现有结果比较。
6. 需要决定的一件事：**no CLS 的领先依赖中性偏移这个后处理**。若不接受验证集调参，应把 no CLS 写成"与 control 等价、分类头参数更少"，而不是"更好"。
