# 餐厅三分类情感分析：Windows 运行指南

项目仅保留餐厅领域的 negative（负面）、neutral（中性）、positive（正面）数据。支持 BERT 基线、DeBERTa 联合模型和 DeBERTa 方面条件句对模型。标签是针对评价对象的方面级情感，不是整句标签。

## 1. 创建并激活环境

安装 Miniconda 或 Anaconda，打开支持 conda 的 PowerShell：

```powershell
conda create --name nlp python=3.12 pip -y
conda activate nlp
cd "D:\projects\NLP-Group"
```

项目路径替换为本机实际路径。也可以将环境放到指定位置，两种创建方式二选一：

```powershell
conda create --prefix "D:\envs\nlp" python=3.12 pip -y
conda activate "D:\envs\nlp"
```

以后打开终端时只需激活已有环境，不重复创建。确认解释器：

```powershell
python -c "import sys; print(sys.executable)"
```

## 2. 安装依赖

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
$env:HF_HOME = Join-Path $PWD.Path ".cache\huggingface"
```

一个 requirements 文件供全部模型使用。Windows 指定 CUDA 12.8 版 PyTorch，GPU 加速需兼容的 NVIDIA 显卡和驱动；也可使用 CPU。Transformers 固定为 4.57.1，DeBERTa tokenizer 依赖 sentencepiece、protobuf。

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

模型命令的 `--device auto` 自动选择 CUDA 或 CPU；明确使用 GPU 可改成 `--device cuda`。

## 3. 准备三分类餐厅数据

同步项目时带上 `resources/restaurant_datasets_3class.zip` 和整个 scripts 目录，运行：

```powershell
python scripts/download_datasets.py --offline
python scripts/download_datasets.py --check
```

应校验 12 个 XML。离线模式不联网；普通模式本地优先、随后网络回退，支持 `--timeout 60 --retries 3`。原始混合标签来源只在转换时暂存内存，不会作为数据集安装。每个文件核对源哈希和派生哈希，允许 Git 引起的 CRLF/LF 换行差异。

```powershell
python scripts/download_datasets.py
```

默认目录：

```text
Database/
├── Restaurant3/                 # 2014，train/test XML 与 JSONL
├── Restaurant2015/              # 2015，统一为显式方面 XML
├── Restaurant2016/              # 2016，统一为显式方面 XML
├── MAMS/data/MAMS-ATSA/raw/      # train/val/test，方面词任务
├── MAMS/data/MAMS-ACSA/raw/      # train/val/test，类别任务
├── manifest.json
├── download_manifest.json
├── overlap_report.json
└── README.md
```

| 训练参数 | 对应目录 | 训练 / 验证 / 测试方面目标 |
|---|---|---|
| `--dataset restaurant` | Restaurant3 | 3602 / 内部划分 / 1120 |
| `--dataset restaurant2015` | Restaurant2015 | 1188 / 内部划分 / 525 |
| `--dataset restaurant2016` | Restaurant2016 | 1714 / 内部划分 / 601 |
| `--dataset mams` | MAMS-ATSA | 11186 / 1332 / 1336 |

MAMS-ACSA 保留供类别级研究，现有方面词模型不接受其作为输入。逐类数量、来源与过滤统计见各目录 manifest 和 `scripts/restaurant_sources.json`。

**不要直接合并这些数据集。** 不同 SemEval 年份有大量重合，2016 训练集与 2015 测试集有 662 句相同文本。MAMS 与 SemEval 也有交叉重合。若合并，需对全部留出测试和验证文本进行去重隔离，再建立新的评估协议。

### 在其他电脑清理旧数据

旧 ASAP、混合 SemEval2014、IMDb 和旧混合数据包不再保留。仅在新集合全部校验通过后执行：

```powershell
python scripts/clean_non_threeclass_data.py
python scripts/clean_non_threeclass_data.py --apply
```

第一条只预览，第二条删除明确列出的旧路径，不删除模型或 runs。若只复制新增文件到另一台电脑，旧目录不会自动消失，需要执行清理命令。

## 4. 数据规则

- Restaurant3 从 SemEval-2014 过滤 conflict，不改原句与方面偏移。
- Restaurant3 分类测试目标仍是 1,120 个，但 joint 抽取 gold 从旧原始版的 1,134 个变为 1,120 个；新旧抽取与端到端分数不能直接混比。
- Restaurant2015/2016 将 Opinions 转换为 aspectTerms，排除 NULL 隐式对象、OutOfScope 句子、无效位置、同跨度情感冲突、重叠跨度；相同跨度相同情感折叠计数。属于派生基准，不将分数直接冒充官方未经处理协议。
- MAMS 原始三分类文件和划分不变。
- 验证集只从训练来源划分，测试不参与调参。训练与测试重复原句从训练侧移除，并记录统计。
- 子词内部结束的方面（例如 dress cod）保留原始字符位置，分类使用覆盖子词；joint 模式屏蔽无法精确表示的 BIO 监督，严格抽取指标不修改 gold 边界。

## 5. 运行测试

```powershell
python -m unittest discover -s tests -v
```

测试数据准备、划分、模型前向/反向、保存加载和命令行流程。微型模型测试不下载真实编码器，也不能证明真实效果。本文不声称最新代码已运行全部模型测试。

## 6. DeBERTa 句对分类（Accuracy 为主）

```text
评论上下文 + 当前方面 → DeBERTa → CLS + 原文方面向量 + 查询方面向量 → MLP → 情感
```

```powershell
python -m absa_enhanced train --mode pair --dataset restaurant --output runs/deberta_pair_split42_seed42 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_accuracy --device auto
```

`--split-seed 42` 固定验证文本，只改变 `--seed 42/43/44` 测训练随机性。省略 split seed 时保留历史行为：跟随 seed。MAMS 已有验证集，不重新划分。

默认梯度检查点、自动 BF16/FP16 和梯度累积，微批量 2、累积 4，有效批量 8。显存不足可改为 `--batch-size 1 --grad-accum 8`；必要时缩短 `--max-length`。pair 的长度包含评论、方面及特殊 token，围绕方面裁剪；不使用 stride。

pair 没有抽取头，抽取损失有效权重固定为 0。其 `aspect_extraction` 和 `end_to_end` 为 null，不能把给定方面的分类成绩当成自动抽取系统效果。

## 7. 联合模型及其他数据集

联合模型共享编码器进行 BIO 抽取与情感分类，使用方面引导的全局/局部注意力和门控融合。窗口长度 192、重叠 64，局部范围默认方面两侧 16 个子词。

```powershell
python -m absa_enhanced train --mode joint --dataset restaurant --output runs/deberta_joint_control_split42_seed42 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --stride 64 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_accuracy --device auto
```

切换数据集，只改 dataset 和输出目录：

```powershell
python -m absa_enhanced train --mode pair --dataset restaurant2015 --output runs/pair_restaurant2015 --seed 42 --split-seed 42 --class-weighting none --device auto
python -m absa_enhanced train --mode pair --dataset restaurant2016 --output runs/pair_restaurant2016 --seed 42 --split-seed 42 --class-weighting none --device auto
python -m absa_enhanced train --mode pair --dataset mams --output runs/pair_mams --seed 42 --class-weighting none --device auto
```

自定义来源可用 `--dataset custom --train 路径 --validation 路径 --test 路径`。输出目录须为空，没有实现中断续训；重试使用新目录。

联合损失为情感交叉熵 + `--extraction-weight` × BIO 交叉熵，默认 0.5。默认类别加权为 sqrt，默认标签平滑 0.05；上面实验显式指定不加权以便比较。其他消融：`--no-fusion`、`--extraction-weight 0`、`--smoothing 0`。关闭抽取损失时仅比较分类，不把其抽取结果当成有效系统。

## 8. 中性（neutral）增强：情感头、上下文范围与校准

中性只占测试方面 17.5%（196/1120），且语料定义是"对该方面没有表达观点"，这与"有观点但是正面/负面"是两个问题。普通三分类 softmax 把两者压在同一个决策面上，于是中性被系统性低估（recall 低、precision 高）。下面三项分别从**参数化、上下文范围、决策阈值**入手，都可单独关闭，默认值等于原行为。

### 情感头 `--sentiment-head`

| 取值 | 输出 | 训练目标 | 说明 |
|---|---|---|---|
| `softmax`（默认） | 3 类 logits | 交叉熵 | 原基线，旧 checkpoint 全部兼容 |
| `ordinal` | 2 个累积 logits | 两个二分类 BCE | 负 < 中 < 正；中性是极性之间的"带"（CORAL） |
| `gate` | 1 个"有无观点" + 2 个极性 logits | BCE + 二分类 CE | 先判该方面有没有观点，再判极性；中性 = 无观点 |

概率都归一到和为 1，标签顺序固定 negative/neutral/positive：

- `ordinal`：p₀=σ(l₀)、p₁=σ(l₀+l₁)，P(neg)=1−p₀、P(neutral)=p₀(1−p₁)、P(pos)=p₀p₁。
- `gate`：P(neutral)=1−σ(l₀)，P(neg)=σ(l₀)·softmax(l₁:)[neg]，P(pos)=σ(l₀)·softmax(l₁:)[pos]。

### 上下文范围 `--context-scope`（仅 pair 模式）

- `sentence`（默认）：围绕方面裁剪整句，行为不变。
- `clause`：只在方面所在的分句内取上下文，按 `; . ! ? ,` 与 `but/however/although/though/whereas/yet/while/except` 切分；分句少于 2 个词时向相邻分句扩展。

动机是情感作用域通常是分句级的："the food was great, but the service was slow" 里两个方面的极性相反，整句表示对其中一个就是混淆项。

### 中性偏移 `--neutral-offset`（`evaluate` 与 `predict`）

把 P(neutral) 乘以 k 再归一化，等价于给中性 logit 加 log k。k 只能在**验证集**上选，测试集只用于报告最终分数。`scripts/tune_neutral_offset.py` 会读训练时保存的验证与测试预测，扫一遍候选值、按验证指标选 k，再给出测试集在该 k 下的成绩。

```powershell
python -m absa_enhanced evaluate --checkpoint runs/deberta_pair_split42_seed42/best --test Database/Restaurant3/test.xml --output runs/pair_eval_offset --neutral-offset 2.5 --device auto
```

### 集成与校准工具

```powershell
# 平均多个 run 的逐方面概率（只读已有产物，不需要 GPU）
python scripts/ensemble_runs.py --runs runs/a runs/b runs/c --output runs/ensemble --bootstrap 2000
# 在验证集上选中性偏移，输出测试集成绩与配对 bootstrap 置信区间
python scripts/tune_neutral_offset.py --runs runs/a runs/b --metric macro_f1
```

两个脚本都要求各 run 预测的是同一批 gold 方面（同一 split seed），否则拒绝合并。配对 bootstrap 回答的是"这个差值是不是噪声"，报告里应写区间而不是只写均值。

### 实测（Restaurant3，split-seed 42，seed 42，6 epoch，`--class-weighting none`，按验证 Accuracy 选模）

完整数据、置信区间与复现命令见 [中性诊断实测结果](docs/neutral_results.md)。

| 配置 | test Acc | test Macro-F1 | neutral P / R / F1 | neutral F1 95% CI |
|---|---|---|---|---|
| `softmax`（对照） | 0.8786 | 0.8076 | 0.782 / 0.531 / 0.632 | [0.552, 0.704] |
| `--no-cls-feature` | 0.8786 | 0.8053 | 0.790 / 0.500 / 0.613 | [0.529, 0.685] |
| `gate` | 0.8723 | 0.8018 | 0.709 / 0.536 / 0.610 | [0.527, 0.683] |
| `ordinal` | 0.8688 | 0.7923 | 0.697 / 0.515 / 0.592 | [0.508, 0.663] |
| `clause` 上下文 | 0.8429 | 0.7438 | 0.727 / 0.408 / 0.523 | [0.442, 0.599] |
| 三头概率集成 | 0.8839 | 0.8166 | 0.778 / 0.536 / 0.634 | — |
| `softmax` + 中性偏移 2.5 | 0.8804 | 0.8114 | 0.748 / 0.561 / 0.641 | — |
| **`--no-cls-feature` + 中性偏移 2.0** | **0.8848** | **0.8192** | **0.779 / 0.551 / 0.644** | — |

读数时注意：

- 前五行中性 F1 的句级 bootstrap 区间**全部互相重叠**，只比较这一列无法区分优劣。
- 单种子、1120 个目标、中性仅 196 个，**2 个 F1 点以内基本是噪声**。表末行相对同 run 未加偏移的配对 bootstrap 是唯一排除 0 的改动（Macro-F1 +0.0139，95% CI [+0.0040, +0.0265]；中性 F1 +0.0319，[+0.0107, +0.0584]）。它来自"CLS 让中性排序略好但决策边界偏低"这一组合，仍需 seed 43/44 复现。其余对比的中性偏移相对对照区间都包含 0，不要写成"显著提升"。
- 把 P(neutral) 乘到扫描过的任意阈值，对照模型的中性 F1 上限只有 0.658（argmax 0.632），`--no-cls-feature` 是 0.689（argmax 0.613）。阈值只能补回决策边界的偏移，**排序本身决定了天花板**。
- **验证集会高估中性**：对照验证 neutral F1 0.685 / 测试 0.632，`gate` 验证 0.720 / 测试 0.611。用验证集调中性相关超参数会把乐观偏差带进结论，最终数字必须来自官方测试集，且测试集只评一次。
- `clause` 上下文把验证与测试的 neutral F1 拉平到 0.52 左右，说明整句上下文既带来泛化也带来混淆：直接切掉分句会丢掉有用信息，不能只当噪声处理。
- 中性错误的 45% 直接复制了同一句里邻近方面的极性；混句中的中性 recall（0.458）低于纯中性句（0.584）。40 对受控样例给出更强的证据：只翻转同句另一个方面的一个词，中性目标的中性概率平均移动 0.354，80% 的对里预测翻转，而极性对照完全不变。这类错误需要跨方面归属能力，不是调阈值能解决的。

`--no-cls-feature` 是去掉了分类头里的 CLS 向量（编码器与注意力不变，方面 token 自身仍编码全局信息），因此它不是"隔绝全句信息"的消融。它在受控样例上把对干扰方面的敏感度从 0.354 降到 0.257，但测试集中性 F1 没有随之提高；收益只有在叠加验证集选出的中性偏移后才出现。

`neutral_diagnostics` 的输出（AP、置信度、校准、句级 bootstrap、40 对受控样例）与本节表格的关系：它提供统一口径与区间，本节数字均已用它复算。两条使用方法上的约束：不要在这份报告上选阈值（测试集只评一次），不要把"匹配邻居极性"写成因果。受控样例的 40 行目前仍是 `pending`，需两人独立复核后才能作为金标准引用。

## 9. 保存与结果

默认按验证 Accuracy 选择 `best/`，同分用 Macro-F1 决胜；另存 `best_macro_f1/`。`--selection-metric gold_macro_f1` 可反过来选主模型，joint 还支持 end_to_end_f1。连续 2 轮主选择键未改善则早停，训练结束后只评估主模型。

```powershell
Get-Content .\runs\deberta_pair_split42_seed42\test_metrics.json
Get-Content .\runs\deberta_pair_split42_seed42\checkpoint_index.json
```

| 输出 | 内容 |
|---|---|
| best/、best_macro_f1/ | 权重、编码器配置、tokenizer、架构与窗口参数 |
| run_config.json | 版本、参数、划分种子、类别权重、对齐统计 |
| history.json | 每轮训练与验证结果 |
| checkpoint_index.json | 最佳轮次与保存规则 |
| training_summary.json | 耗时、最佳验证分数、峰值分配显存 |
| test_metrics.json | gold_aspect_sentiment Accuracy/Macro-F1，joint 另有抽取及联合 F1 |
| test_gold_predictions.jsonl | 逐方面分类预测与概率 |
| test_provided_aspects.jsonl | pair 模式提供方面时的逐句输出 |
| test_end_to_end.jsonl | joint 自动抽取输出 |

这两种模型的样本组织和监督不同，同 epoch 不等同于同计算预算，比较时报告设置和训练成本。额外 checkpoint 增加磁盘占用。

## 10. 预测与独立评估

指定方面：

```powershell
python -m absa_enhanced predict --checkpoint runs/deberta_pair_split42_seed42/best --text "The food was excellent, but the service was slow." --aspect service --device auto
```

同词出现多次时提供 `--aspect-start`。pair 自动模式需要已有 joint 抽取器：

```powershell
python -m absa_enhanced predict --checkpoint runs/deberta_pair_split42_seed42/best --extractor-checkpoint runs/deberta_joint_control_split42_seed42/best --text "The food was excellent, but the service was slow." --device auto
```

先抽取再释放第一模型，随后加载分类器；不会同时让两个模型驻留 GPU。未抽到方面返回空列表。两阶段预测不自动生成全测试集联合指标。

```powershell
python -m absa_enhanced evaluate --checkpoint runs/deberta_pair_split42_seed42/best --test Database/Restaurant3/test.xml --output runs/pair_evaluation --device auto
```

旧 checkpoint 仍可加载。置信度为未经校准的 softmax 概率。

## 11. 分组 K 折交叉验证

交叉验证评估模型对训练样本划分的稳定性，不保证提高单个模型 Accuracy。当前支持 pair 和 joint；默认 5 折，复用相同训练超参数，按所选验证指标保存每折模型。

### 协议与隔离

- 默认只把原始 train 放入 CV 样本池；官方 test 不进入训练或评分，仅用于提前排除重复文本/评论。不会在每折反复评估官方测试集。
- 外层分成 K 折，每次留出一折只做最终评价。
- 外层剩余部分再分出独立的内层验证集，用于早停和 checkpoint 选择。外层留出折不参与选轮次。
- 使用 StratifiedGroupKFold 尽量平衡方面类别比例，同一句所有方面及完全相同文本永不跨组；分组条件下不能保证各折数量完全相等。
- `--group-by auto` 对 Restaurant2015/2016 按原评论分组，并将跨评论的完全重复句连接到同一组；其他数据按原句分组。自定义评论数据可显式使用 `--group-by review`，要求句子 ID 保留 review_id:sentence_id 格式。
- 每折从同一原始预训练 checkpoint 重新初始化并独立训练，默认训练种子相同，划分由 `--split-seed` 固定。不会用上一折训练权重作为下一折初始化。
- MAMS 原 val 默认不进入样本池；明确添加 `--include-validation` 才将原 train+val 一起交叉验证。官方 test 始终保留。
- 这是“外层交叉评价 + 内层 holdout 选轮次”，不是完整的内层超参数网格搜索。反复根据外层 CV 分数选择方案后，该分数也不再是最终独立测试分数。

依赖文件新增 scikit-learn，先更新：

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### 先只生成划分

此命令不下载编码器、不训练，生成分折 XML 和分组清单：

```powershell
python -m absa_enhanced cross-validate --mode pair --dataset restaurant --output runs/cv_pair_restaurant_split42 --folds 5 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_accuracy --device auto --plan-only
```

确认后，使用同一配置，加 `--resume`、去掉 `--plan-only` 开始训练：

```powershell
python -m absa_enhanced cross-validate --mode pair --dataset restaurant --output runs/cv_pair_restaurant_split42 --folds 5 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_accuracy --device auto --resume
```

也可直接运行第二条命令使用一个不存在的新输出目录，不要求先生成划分。5 折依次执行以控制显存，训练总成本大致为数次单模型训练；默认每折保存 Accuracy 与 Macro-F1 两份 checkpoint，应预留数 GB 到十余 GB 磁盘，实际取决于编码器。

joint 模式把 `--mode pair` 改成 `--mode joint`，同时使用新输出目录。其他餐厅数据可改变 `--dataset`。类别独立组过少、某折缺少类别或无法产生内层验证集时，代码会在训练前报错，需减少折数或调整划分种子。

### 查看结果

```powershell
Get-Content .\runs\cv_pair_restaurant_split42\cv_summary.json
```

```text
runs/cv_pair_restaurant_split42/
├── cv_config.json               # 协议、参数、源文件与实现哈希
├── fold_assignment.json         # 每句组号及其外层留出折
├── fold_01/
│   ├── splits/                  # train / validation / held_out XML
│   ├── fold_manifest.json       # 内外层句子、组与类别计数
│   ├── run/                     # 独立模型、history 和外层评分
│   └── completed.json           # 完成标记及结果哈希
├── fold_02/ ... fold_05/
├── fold_metrics.csv
├── oof_predictions.jsonl
└── cv_summary.json
```

- `fold_statistics.gold_aspect_sentiment.accuracy`：各折 Accuracy 的均值与样本标准差。
- `fold_statistics.gold_aspect_sentiment.macro_f1`：各折 Macro-F1 的均值与样本标准差。
- `oof_gold_aspect_sentiment`：每个方面恰好被其未参与训练的外层模型预测一次，合并后计算的整体指标。它不等于各折 F1 的简单平均。
- joint 额外汇总各折抽取与端到端指标；pair 不提供不存在的抽取成绩。
- 每折 `run/test_metrics.json` 的 test 指外层 held_out，不是官方 test。

`--resume` 只跳过源数据、参数、实现哈希匹配且结果完整的已完成折；不支持恢复某折训练到一半的优化器状态。遇到未完成的 run 目录会中止，不自动删除实验。代码或参数改变时使用新输出目录。

新增离线测试包含小模型的 3 折命令行流程，全部测试耗时会增加。本文未声称最新交叉验证代码已运行训练或通过测试，应在目标 Windows 环境执行验证。

## 12. BERT 基线

原始混合数据已删除，使用三分类 Restaurant3：

```powershell
python -m bert_baseline train --model google-bert/bert-base-uncased --train Database/Restaurant3/train.xml --test Database/Restaurant3/test.xml --output runs/bert_restaurant_threeclass --epochs 3 --batch-size 4 --learning-rate 2e-5 --max-length 128 --seed 42 --device auto
```

## 13. 常见问题

- 数据缺失：带上全量三分类 ZIP，运行离线恢复和检查，不重新创建笔记本或 ASAP 目录。
- XML 下载失败：脚本支持备用地址、HTTPS HF 镜像参数和重试；也可 `--import-from` 指定原始三分类目录/ZIP。不要关闭 SSL 校验。
- 模型下载失败：检查网络或将 --encoder 指向完整本地模型目录。
- 显存不足：降低 batch-size、增加 grad-accum，保留检查点及混合精度。
- 非有限损失：尝试较低学习率或 --precision no，全精度更占显存。
- 找不到模块：同步整个 absa_enhanced、scripts、tests 目录及 restaurant_sources.json，从项目根目录运行。
- Tokenizer 性能提示和 Windows redirect 提示通常不是失败原因，以 traceback 与退出码为准。

## 来源

中性方面错误诊断、40 对控制样例和 `--no-cls-feature` 消融见 [实验指南](docs/neutral_scope_study.md)。
个人分支与 PR 提交步骤见 [贡献指南](docs/GITHUB_CONTRIBUTION.md)。这些工具不预设真实模型的实验结果。

- 模型：https://huggingface.co/microsoft/deberta-v3-base
- 句对方法：https://aclanthology.org/N19-1035/
- SemEval-2014：https://alt.qcri.org/semeval2014/task4/
- SemEval-2015：https://alt.qcri.org/semeval2015/task12/
- SemEval-2016：https://alt.qcri.org/semeval2016/task5/
- MAMS：https://github.com/siat-nlp/MAMS-for-ABSA

数据派生规则与源版本必须随结果报告，遵循原发布方使用条件。注意力、多任务与局部上下文已有研究，本项目不声称组合是全新算法。
