# 中性诊断：实测结果（Restaurant3，split-seed 42，seed 42）

本文只记录**已经跑出来**的数字，供 [实验指南](neutral_scope_study.md) 的结果表填写。所有配置使用同一套 1120 个 gold 方面（同一 `--split-seed 42`），差异只来自一个因素。

固定设置：`--mode pair --dataset restaurant --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_accuracy`。按验证 Accuracy 选模，测试集只评一次。GPU 为 RTX 5060 Laptop 8GB，BF16，单次训练约 18 分钟。

## 1. 主表

| 配置 | Acc | Macro-F1 | 中性 P | 中性 R | 中性 F1 | 中性 F1 95% CI（句级 bootstrap） | 中性 AP |
|---|---|---|---|---|---|---|---|
| control（含 CLS） | 0.8786 | 0.8076 | 0.782 | 0.531 | 0.632 | [0.552, 0.704] | 0.725 |
| no CLS（`--no-cls-feature`） | 0.8786 | 0.8053 | 0.790 | 0.500 | 0.613 | [0.529, 0.685] | **0.745** |
| gate 头 | 0.8723 | 0.8018 | 0.709 | 0.536 | 0.610 | [0.527, 0.683] | 0.712 |
| ordinal 头 | 0.8688 | 0.7923 | 0.697 | 0.515 | 0.592 | [0.508, 0.663] | 0.687 |
| clause 上下文 | 0.8429 | 0.7438 | 0.727 | 0.408 | 0.523 | [0.442, 0.599] | 0.648 |

**五个配置的中性 F1 置信区间全部互相重叠**，单看这一列不能区分优劣。中性 support 一律 196。

中性 AP 的随机基线是中性的占比 0.175，所以 0.65–0.75 说明排序信号远强于随机，不是"没有信息"。

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

## 3. 单因素干预：中性偏移（验证集选 k，测试集只报一次）

| 配置 | 验证集选出的 k | 测试 Acc（k=1 → 选中 k） | 测试 Macro-F1 | 测试中性 F1 | 配对 bootstrap 结论 |
|---|---|---|---|---|---|
| control | 2.5 | 0.8786 → 0.8804 | 0.8076 → 0.8114 | 0.632 → 0.641 | 包含 0，不能声称提升 |
| no CLS，seed 42 | 2.0 | 0.8786 → 0.8848 | 0.8053 → 0.8192 | 0.613 → 0.644 | 排除 0 |
| no CLS，seed 43 | 1.0 | 0.8884 → 0.8884 | 0.8278 → 0.8278 | 0.669 → 0.669 | 区间为零，验证集判定"不需要偏移" |
| no CLS，seed 44 | 2.0 | 0.8679 → 0.8705 | 0.7955 → 0.8039 | 0.591 → 0.617 | 中性 F1 排除 0，Macro-F1 包含 0 |

三个种子（同一 split-seed 42 划分、同一批 1120 个测试方面）：

| 指标 | 未加偏移 | 加上验证集选出的偏移 | 每个种子的差值 |
|---|---|---|---|
| Accuracy | 0.8783 ± 0.0103 | 0.8812 ± 0.0094 | +0.0062 / ±0.0000 / +0.0027 |
| Macro-F1 | 0.8095 ± 0.0166 | **0.8170 ± 0.0121** | +0.0139 / ±0.0000 / +0.0085 |
| 中性 F1 | 0.6239 ± 0.0402 | **0.6432 ± 0.0260** | +0.0319 / ±0.0000 / +0.0260 |

（均值 ± 样本标准差，n = 3 个训练种子。）

读法：

1. **三个种子里偏移一次都没有变差**，平均把 Macro-F1 抬高 0.0074、中性 F1 抬高 0.0193，并把种子间标准差从 0.0166 压到 0.0121（中性从 0.0402 压到 0.0260）。这说明它修的是"决策边界随种子抖动"，不是重新排序。
2. 增益大小取决于种子的校准状态：seed 43 的 k=1 已经最优，验证集也正确判出"不需要偏移"；seed 42/44 验证集选了 k=2.0，测试集随之改善。**用验证集选 k 在这 3 个种子上都是安全的**（没有一次把测试集搞差）。
3. 只有一个种子（42）的 Macro-F1 差值区间排除 0。**因此不能写"显著提升"**，只能写"在该划分上稳定不劣，且均值改善、方差减少"。样本只有 3 个种子，标准差本身也不稳定。

仍然要写明的限制：k 在只有 113 个中性方面的验证集上从网格里选，选中值自带不确定性；validate→test 的收益依赖验证集与测试集的中性分布相近。

## 3b. 种子方差比配置差异更大

同一配置（no CLS，k=1）在 seed 42/43/44 上的测试 Macro-F1 是 0.8053 / 0.8278 / 0.7955，**极差 0.0323，标准差 0.0166**。

对比第 1 节：`gate` 相对 control 是 −0.0058，`ordinal` 是 −0.0153，`clause` 是 −0.0638。也就是说 **`gate` / `ordinal` 与 control 的差距完全落在种子噪声里**，而 `clause` 的降幅（−0.064）明显超出噪声，是唯一可以下结论的结构性差异。第 1 节表格里凡是单种子的逐项比较，都需要先做多种子才能引用。

同样，no CLS 三种子均值（0.8095 ± 0.0166）与 control 单种子（0.8076）无法区分优劣 —— 要说"两者等价"必须补 control 的 seed 43/44。

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
- no CLS 把这个敏感度降到 0.257，不变率 0.20 → 0.433，方向与"CLS 携带句级捷径"一致，但只减轻一半，且测试集中性 F1 并没有因此变好。
- clause 的不变率 1.000 是**构造出来的**，不是学出来的：分句裁剪把干扰方面整段从输入里删掉，目标分句两个条件下逐 token 相同，前向完全相同（概率变化恰好 0.000）。所以它不能用来证明模型内部作用域变好，只能说明"如果输入里没有干扰，它就不会被干扰"。而真测试集上 clause 反而最差（中性 F1 0.523，Macro-F1 0.7438），说明模型确实需要跨分句语境，问题出在**归属**而不是**读取**。
- 中性目标上 `both_correct` 是 30 对里的 6 对 / 13 对，二项噪声很大，不要单独引用。

样例仍是 pending，未做双人独立复核，因此这部分只能作为开发期诊断，不能当作人工金标准结论。

## 5. 验证集与测试集

| 配置 | Macro-F1 验证−测试 | 中性 F1 验证−测试 | 验证中性 support | 测试中性 support |
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

# 多种子复现（seed 43/44，split-seed 固定 42）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_neutral_seeds.ps1

# 统一报告（AP、置信度、校准、句级 bootstrap）
python -m neutral_diagnostics analyze --predictions runs/pair_nocls_s42/test_gold_predictions.jsonl `
  --gold-xml Database/Restaurant3/test.xml --dataset restaurant3 --split test `
  --run-id pair_nocls_seed42 --model-label pair_no_cls --training-seed 42 --output runs/analysis_no_cls

# 阈值上限：读 neutral_pr_probability.csv 逐点算 F1 取最大

# 验证集选中性偏移 + 配对 bootstrap
python scripts/tune_neutral_offset.py --runs runs/pair_nocls_s42 --metric macro_f1 --bootstrap 2000

# 受控样例
python -m neutral_diagnostics predict-pairs --checkpoint runs/pair_nocls_s42/best `
  --output runs/scope_pair_nocls --include-unreviewed --device auto
```

## 7. 下一步

1. ~~seed 43/44 复现~~ 已完成，见第 3、3b 节：偏移稳定不劣、均值改善、方差下降；但种子方差本身大于 `gate`/`ordinal` 的差异。
2. 补 control 的 seed 43/44，才能判断 no CLS 与 control 是否等价（目前单种子 control 不可比）。
3. 用 `cross-validate --folds 3` 在同一分组划分上比 control / no CLS，外层留出才是无偏估计。
4. 40 对样例需要两人独立复核后转 `approved`，现在的结论只是探索性。
5. 若要把中性 F1 再往上推，需要改表征（例如按分句给注意力先验、或对每句其他方面做显式对比），而不是继续调阈值——第 2 节已经给出阈值的天花板。注意第 3b 节：任何新配置都必须至少 3 个种子才能和现有结果比较。
