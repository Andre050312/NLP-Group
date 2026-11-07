# Windows 情感分析：BERT 基线与 DeBERTa 多任务增强版

本项目共用一个 `requirements.txt`：

| 入口 | 用途 |
|---|---|
| `python -m bert_baseline` | 原 BERT-base 句对分类基线，需提供方面词 |
| `python -m absa_enhanced train --mode joint` | DeBERTa 联合模型：方面抽取与情感分类 |
| `python -m absa_enhanced train --mode pair` | DeBERTa 方面条件句对分类：以 Accuracy 为主要目标 |

以下步骤使用 Windows PowerShell，从创建独立 Python 环境开始。命令中的项目路径为示例，请替换为本机实际路径。

## 1. 创建并激活虚拟环境

先安装 Miniconda 或 Anaconda，使 PowerShell 可以使用 `conda` 命令。在支持 Conda 的 PowerShell 终端中创建独立环境：

```powershell
conda create --name nlp python=3.12 pip -y
conda activate nlp
```

如果希望指定环境保存位置，可以改用以下方式。它与上面的命名环境方式二选一，路径可自行更改：

```powershell
conda create --prefix "D:\envs\nlp" python=3.12 pip -y
conda activate "D:\envs\nlp"
```

进入项目根目录，确认其中包含 `requirements.txt`、`bert_baseline/`、`absa_enhanced/` 和 `tests/`：

```powershell
cd "D:\projects\NLP-Group"
python --version
python -c "import sys; print(sys.executable)"
```

最后一条命令应指向刚创建的环境。以后每次打开终端，都先激活所选环境，再进入项目目录。

## 2. 安装依赖与选择计算设备

基线和增强版使用同一个依赖文件：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Windows 依赖配置安装 `torch==2.10.0+cu128`，使用官方 CUDA 12.8 包索引；GPU 加速需要支持该运行时的 NVIDIA 显卡及驱动。Transformers 固定为 `4.57.1`，DeBERTa tokenizer 使用 sentencepiece 和 protobuf。无须额外安装 torchvision、torchaudio。

检查 PyTorch 与 GPU 可用性（没有 GPU 也可执行）：

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

下文命令使用 `--device auto`，CUDA 可用时选择 GPU，否则使用 CPU。也可显式传 `--device cuda` 或 `--device cpu`；CPU 训练通常更慢。当前 CUDA 版 PyTorch 也支持 CPU 计算。

可选：将模型缓存放在项目目录下，或将路径改成其他有足够空间的位置。该设置只对当前终端有效：

```powershell
$env:HF_HOME = Join-Path $PWD.Path ".cache\huggingface"
```

## 3. 数据

```text
Database/
├── SemEval2014/
│   ├── ABSA_Gold_TestData/
│   │   ├── Restaurants_Test_Gold.xml
│   │   └── Laptops_Test_Gold.xml
│   └── SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/
│       ├── Restaurants_Train_v2.xml
│       └── Laptop_Train_v2.xml
└── MAMS/data/MAMS-ATSA/raw/
    ├── train.xml
    ├── val.xml
    └── test.xml
```

训练入口的 `--dataset restaurant`、`laptop`、`mams` 自动定位数据。使用统一下载脚本准备 XML，不需要 Git 或额外 Python 依赖：

```powershell
python scripts/download_datasets.py
```

默认准备 SemEval 餐厅/笔记本的训练与测试 XML（4 个），以及 MAMS-ATSA、MAMS-ACSA 的 train/val/test（6 个），共 10 个 XML。**同步项目时同时复制 `resources/absa_xml_bundle.zip`**，这是项目已有原始 XML 的离线包。脚本按以下顺序寻找每个文件：

1. 校验目标位置已存在的文件。
2. 从 `--import-from` 指定的本地目录/ZIP、项目离线包或错放在嵌套目录中的 XML 恢复。
3. 本地没有可用文件时才联网，失败后自动尝试备用地址。

现有目录可以保留。损坏或版本不匹配的旧文件替换前备份为 `.invalid-文件哈希`，新文件先写临时文件，XML 解析、字符位置和固定 SHA-256 校验成功后再原子替换。Git 导致的 CRLF/LF 换行差异可接受，不会改写实际文件；其他内容差异将报错。脚本不下载 ASAP CSV 或执行来源仓库代码。

### 网络下载失败时使用离线包

确认 `resources/absa_xml_bundle.zip` 已放入项目，执行：

```powershell
python scripts/download_datasets.py --offline
python scripts/download_datasets.py --check
```

`--offline` 不进行任何网络请求；离线包缺失时会明确提示，不静默退回联网。也可以从另一台机器复制的原始数据目录或 ZIP 导入，路径替换为实际位置：

```powershell
python scripts/download_datasets.py --offline --import-from "D:\datasets\absa_xml_bundle.zip"
```

导入会按文件内容哈希辨别多个同名 `train.xml`，并恢复标准目录层级，不要求来源目录恰好与目标目录一致。

只下载 SemEval：

```powershell
python scripts/download_datasets.py --dataset semeval
```

只下载 MAMS：

```powershell
python scripts/download_datasets.py --dataset mams
```

离线检查已有文件（不下载、不修改）：

```powershell
python scripts/download_datasets.py --check
```

下载来源固定到已记录的仓库版本，所有文件须通过内置 SHA-256 校验。下载报告、实际来源、文件哈希和样本数保存在 `Database/download_manifest.json`，`--check` 只检查、不联网也不写文件。来源及使用说明见 `Database/README.md`。

联网超时可调整为 `--timeout 60 --retries 3`；SemEval 支持用 `--hf-endpoint` 指定可访问的 HTTPS Hugging Face 镜像，镜像内容同样必须通过哈希检查。无需关闭 SSL 校验，也不需要管理员权限。某一个文件失败不会阻止检查其他文件，最终会列出失败项目并返回非零退出码。

下载脚本保留原始 XML 的标签和字符位置，不会把 `dress cod` 自动改成 `dress code`。这类真实子串标注可通过 XML 校验，其子词边界兼容由模型读取代码处理；因此遇到旧版对齐报错时应更新代码，而不是反复下载或手工改测试集。

协议：

- 情感三分类：negative、neutral、positive。conflict 不参加情感训练/分类评价；抽取任务仍学习 conflict 方面。
- SemEval 验证文本选择复用原基线划分逻辑与 seed。增强版保留原训练来源中没有三分类目标的句子，提供额外抽取背景监督；这些句子不增加三分类指标的样本数。
- MAMS 使用原 train/val/test。同一句不同方面不能跨集合；训练与验证文本重合时报错。
- 保留完整测试集；从训练、验证中排除与测试完全相同的文本并记录数量。测试成绩不用于选超参数或 epoch。
- 检查 XML 字符偏移与窗口覆盖。若方面边界落在子词内部，情感分类使用覆盖该方面的子词，保留原始标签与字符跨度；对应 BIO token 屏蔽为 `-100`，不训练错误的扩展边界。数量和示例记录在 `run_config.json` 的 `token_alignment`。端到端评价仍使用原始跨度，因此无法精确表示的方面仍影响抽取召回率，不会被改写或从评价中删除。
- ASAP、MAMS-ACSA 是类别任务，不支持本版读取器；跨数据集实验先检查重复，不直接拼接随机划分。

## 4. 运行测试

仅新增测试：

```powershell
python -m unittest discover -s tests -p "test_enhanced.py" -v
```

全部测试：

```powershell
python -m unittest discover -s tests -v
```

新增测试使用本地小词表和随机初始化的微型 BERT，不下载预训练模型。覆盖重叠窗口、重复方面、conflict、联合梯度、无方面句子、BIO 解码、联合指标、保存加载，以及命令行训练/评价/两种预测。测试通过不等于真实 DeBERTa 训练效果已经验证。

## 5. Restaurant 主实验

### 方面条件句对分类（新增）

每个训练样本输入“评论上下文 + 当前方面”两段序列，使方面信息参与编码器的每一层。分类头融合 CLS、评论中方面的池化向量和第二段方面向量。长评论按目标位置裁剪，保证当前方面保留；`--max-length` 包含两段和特殊 token，`--stride` 在 pair 模式不使用。保留原始字符位置，兼容子词内部结束的标注。

```text
评论上下文 + 方面词 → DeBERTa 联合编码 → CLS + 评论方面表示 + 查询方面表示 → MLP → 情感
```

执行下面的命令，先保持与已有不加权实验一致的损失配置，测试输入方式的变化：

```powershell
python -m absa_enhanced train --mode pair --dataset restaurant --encoder microsoft/deberta-v3-base --output runs/deberta_pair_split42_seed42 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_accuracy --device auto
```

pair 模式没有 BIO 抽取头，抽取损失有效权重固定为 0（记录在 `effective_extraction_weight`），只训练有三分类标签的方面。`--local-radius`、`--no-fusion` 不影响 pair 架构。结果里的 `gold_aspect_sentiment` 是实际分类指标；`aspect_extraction`、`end_to_end` 为 `null`，不能把给定方面的分类成绩当成自动抽取系统成绩。

同划分、同训练种子和同选择规则的 joint 对照：

```powershell
python -m absa_enhanced train --mode joint --dataset restaurant --output runs/deberta_joint_control_split42_seed42 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --stride 64 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_accuracy --device auto
```

joint 模式每个窗口可以同时训练多个方面并使用抽取背景句，pair 模式逐方面编码，训练样本数和优化器更新数因此不同。同轮数不是严格等算力比较，应同时报告训练时间与样本组织差异。

### 固定划分与训练随机性

`--split-seed` 只控制从训练来源划出的验证文本，`--seed` 控制模型初始化、dropout 和数据顺序。多种子实验保留 `--split-seed 42`，只改变 `--seed 42/43/44`，让验证集保持一致。MAMS 已有验证集，不受 split seed 改变。

为兼容历史实验，省略 `--split-seed` 时仍沿用 `--seed` 划分；`run_config.json` 的 `resolved_split_seed` 记录实际值。以前未指定 split seed 的 seed 43/44 实验，其验证集与新固定 split 42 实验不同，不能直接视为同一划分的随机初始化对照。PowerShell 启动脚本新增 `-Mode` 和 `-SplitSeed`，默认固定 split 42。

### 原联合模型训练

可选：先用独立目录跑 1 epoch：

```powershell
python -m absa_enhanced train --dataset restaurant --output runs/enhanced_restaurant_smoke --epochs 1 --batch-size 2 --grad-accum 4 --max-length 192 --stride 64 --device auto
```

正式训练：

```powershell
python -m absa_enhanced train --dataset restaurant --encoder microsoft/deberta-v3-base --output runs/deberta_joint_restaurant_seed42 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --stride 64 --seed 42 --device auto
```

默认微批量 2、累积 4，有效批量 8。默认开启梯度检查点，CUDA 下自动启用混合精度；实际显存需求取决于输入长度、方面数量与设备配置。首次需联网下载编码器。每个 epoch 输出验证指标；默认按**给定正确方面的验证 Accuracy**保存主模型 `best/`，Accuracy 相同时以 Macro-F1 决胜。同时保存验证 Macro-F1 最佳的 `best_macro_f1/`（同分时以 Accuracy 决胜）。连续 2 轮主选择指标及其决胜指标均未改善则早停；训练结束后只对主模型做一次测试。

`checkpoint_index.json` 记录每个保存标准对应的目录、轮次和验证分数。两项目标分别保存，不表示同一个模型一定同时最大化它们；额外 checkpoint 会增加磁盘占用。使用 `--selection-metric gold_macro_f1` 可恢复原主选择规则，此时 `best/` 对应 Macro-F1 最佳，`best_accuracy/` 对应 Accuracy 最佳。该规则只改变 checkpoint 选择，不改变训练损失。

重点研究端到端系统时，可在独立实验改为 `--selection-metric end_to_end_f1`；不要根据测试表现反复切换标准。输出目录必须为空；没有实现中断续训，失败后应使用新目录重试。

使用 NVIDIA GPU 时，也可从已激活环境的 PowerShell 运行启动脚本，并显式指定通用缓存路径：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\train_windows.ps1 -Dataset restaurant -Output runs/deberta_joint_restaurant_seed42 -CacheDir "$PWD\.cache\huggingface"
```

该执行策略只用于这一次脚本进程。直接使用 `python -m ...` 不需要修改任何执行策略。

## 6. 查看指标与错误案例

```powershell
Get-Content .\runs\deberta_joint_restaurant_seed42\test_metrics.json
```

| 字段 | 含义 |
|---|---|
| `gold_aspect_sentiment.accuracy` | 给定正确方面的准确率，可与相同评价协议下的 BERT 基线比较 |
| `gold_aspect_sentiment.macro_f1` | 给定正确方面的 Macro-F1，可与相同评价协议下的 BERT 基线比较 |
| `aspect_extraction.f1` | 方面字符跨度严格匹配 F1，包含 conflict 方面 |
| `end_to_end.f1` | 字符跨度与情感同时正确的 F1，不能与分类 F1 混比 |

端到端三分类评价排除 gold conflict 对，并忽略**准确命中 conflict 标注跨度**的预测，其余错误预测照常计入假阳性。这是项目协议，不应直接对照不同评价协议的论文。对比基线时，需确认测试集版本、标签过滤和目标样本数一致；代码不硬编码评价样本数。

`test_gold_predictions.jsonl` 保存给定正确方面的逐条情感预测，可检查中性被判为正面的案例。`test_end_to_end.jsonl` 保存每句的自动抽取结果。

## 7. 推理

### 句对模型

指定方面进行分类：

```powershell
python -m absa_enhanced predict --checkpoint runs/deberta_pair_split42_seed42/best --text "The food was excellent, but the service was slow." --aspect service --device auto
```

可用已训练的联合模型作为第一阶段抽取器，再由句对模型重新判断各方面情感。将 `runs/joint_experiment/best` 换成实际联合模型路径：

```powershell
python -m absa_enhanced predict --checkpoint runs/deberta_pair_split42_seed42/best --extractor-checkpoint runs/joint_experiment/best --text "The food was excellent, but the service was slow." --device auto
```

推理先运行抽取器，再释放其模型，加载句对分类器，避免两个编码器同时驻留 GPU。没有抽到方面时返回空列表。该两阶段预测不自动生成全测试集端到端指标。

评估给定正确方面的分类：

```powershell
python -m absa_enhanced evaluate --checkpoint runs/deberta_pair_split42_seed42/best --test Database/SemEval2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml --output runs/deberta_pair_evaluation --device auto
```

pair 模式逐句的预测文件名为 `test_provided_aspects.jsonl`，避免与自动端到端结果混淆。旧 joint checkpoint 不需要转换，加载器自动识别两种模型格式。

### 联合模型

自动抽取方面并预测：

```powershell
python -m absa_enhanced predict --checkpoint runs/deberta_joint_restaurant_seed42/best --text "The food was excellent, but the service was slow." --device auto
```

指定方面：

```powershell
python -m absa_enhanced predict --checkpoint runs/deberta_joint_restaurant_seed42/best --text "The food was excellent, but the service was slow." --aspect service --device auto
```

同词出现多次时，增加 `--aspect-start` 指定字符起点。自动模式可能漏抽或无输出，程序不会人工补全；置信度是未经校准的 softmax 概率。

单独评估：

```powershell
python -m absa_enhanced evaluate --checkpoint runs/deberta_joint_restaurant_seed42/best --test Database/SemEval2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml --output runs/deberta_restaurant_evaluation --device auto
```

## 8. MAMS 与 Laptop 独立实验

```powershell
python -m absa_enhanced train --dataset mams --output runs/deberta_joint_mams_seed42 --epochs 6 --batch-size 2 --grad-accum 4 --seed 42 --device auto
```

```powershell
python -m absa_enhanced train --dataset laptop --output runs/deberta_joint_laptop_seed42 --epochs 6 --batch-size 2 --grad-accum 4 --seed 42 --device auto
```

自定义 XML：`--dataset custom --train 路径 --validation 路径 --test 路径`；validation 可省略，从训练来源按句划分。

## 9. 消融实验

固定划分、种子、预算与验证选择规则，实验使用不同输出目录：

| 实验 | 在增强版训练命令上调整 |
|---|---|
| 同结构 BERT 对照 | `--encoder google-bert/bert-base-uncased` |
| 完整 DeBERTa | 默认配置 |
| 去除局部融合 | `--no-fusion` |
| 去除类别加权 | `--class-weighting none` |
| 去除抽取辅助损失 | `--extraction-weight 0`；只比较 gold 方面分类，抽取头未被有效监督，不能当成正常端到端系统 |

原 BERT 句对基线与增强版在结构、上下文长度、额外监督、训练策略上均不同，不把全部收益归因于底座。最终建议 seed 42/43/44，报告均值和标准差，不能保证单次提升稳定成立。

## 10. 输出

```text
runs/deberta_joint_restaurant_seed42/
├── best/
│   ├── model.pt                 # 完整多任务权重
│   ├── settings.json            # 架构、窗口参数
│   ├── encoder_config/          # 编码器配置
│   └── tokenizer/               # 分词器
├── run_config.json              # 参数、版本、类别权重、去重统计
├── checkpoint_index.json        # 不同验证指标最佳模型的目录与轮次
├── best_macro_f1/               # 默认 Accuracy 主规则下另存的 Macro-F1 最佳模型
├── split_manifest.json          # 划分文本与 ID
├── history.json                 # 每轮训练与验证指标
├── training_summary.json        # 耗时、最佳成绩、CUDA峰值分配显存
├── validation_best_metrics.json
├── validation_best_gold_predictions.jsonl
├── validation_best_end_to_end.jsonl
├── test_metrics.json
├── test_gold_predictions.jsonl
└── test_end_to_end.jsonl
```

移动 checkpoint 时保留整个 `best/`。这是自定义多任务模型，使用本项目命令加载，不用 `AutoModelForSequenceClassification` 直接读取 `model.pt`。

## 11. 常见问题

- `test_actual_downloaded_xml` 失败：该项需要实际数据文件，其余离线测试使用临时小样本。执行 `Get-ChildItem .\Database -Recurse -Filter *.xml | Select-Object FullName`，按“数据”一节补齐文件并检查是否多套了目录；测试报错会列出缺失路径。只同步代码不会自动同步数据。
- 显存不足：改为 `--batch-size 1 --grad-accum 8`；仍不足可用 `--max-length 128 --stride 32`。保留混合精度与检查点；重试换新输出目录。
- `Tokenizer cannot represent exact aspect boundary`：同步最新 `absa_enhanced/data.py` 和 `cli.py`。某些源标注结束于子词内部（例如原文 `dress code` 标成 `dress cod`），新代码保留情感样本、屏蔽不可精确表示的 BIO 监督并记录统计；不修改原始 XML。
- 模型下载失败：检查网络，或完整下载配置/权重/tokenizer 后将 `--encoder` 指向本地目录。
- Fast tokenizer 或 protobuf 错误：确认已安装统一 requirements，重新打开终端并激活项目环境，不跳过 tokenizer 依赖。
- 非有限损失：新目录尝试 `--precision no` 或更低学习率；全精度更占显存，降低 batch-size。
- 找不到模块：将整个 `absa_enhanced/` 同步到 Windows，并从项目根目录运行。
- Redirects not supported in Windows：通常是 PyTorch 日志提示，是否失败看最终退出码/traceback。

## 12. 原 BERT 基线

旧入口保留，既有实验无需覆盖。重新运行时用新目录：

```powershell
python -m bert_baseline train --model google-bert/bert-base-uncased --train "Database/SemEval2014/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml" --test Database/SemEval2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml --output runs/bert_restaurant_new_seed42 --epochs 3 --batch-size 4 --learning-rate 2e-5 --max-length 128 --seed 42 --device auto
```

## 13. 模型架构

```text
评论 → Fast Tokenizer → 重叠窗口 → DeBERTa-v3-base（共享编码器）
                                   ├─ BIO 分类头 → 自动方面抽取
                                   └─ 方面表示
                                        ├─ 方面引导全局注意力
                                        └─ 方面引导局部注意力
                                                   ↓
                                                门控融合
                                                   ↓
                            CLS + 方面表示 + 融合上下文 → MLP → 情感
```

- 一个编码器共享给抽取和情感两项任务。情感训练使用标注方面，自动推理使用预测方面，分别报告两种成绩。
- 全局上下文指当前窗口内全部有效 token；局部上下文为方面两侧各 16 个子词，可用 `--local-radius` 调整。
- 长句使用重叠窗口，检查每个标注方面至少被一个窗口完整覆盖；同一方面的跨窗口情感概率取平均，自动预测按原句字符跨度去重。
- 总损失：情感交叉熵 + `0.5 × BIO交叉熵`。BIO O/B/I 类权重为 0.3/1/1，padding 与特殊 token 不参与损失。
- 默认情感类别权重为训练类别频次的平方根倒数，标签平滑 0.05；可关闭进行消融。
- 编码器学习率 `1e-5`，新增头 `1e-4`；使用 warmup、学习率衰减、梯度裁剪和验证集早停。
- 默认梯度检查点；CUDA 自动选择 BF16（若支持）或 FP16，FP16 使用梯度缩放；支持梯度累积。
- 编码器可换为 `google-bert/bert-base-uncased`，在同结构下比较模型底座。

本版实现监督多任务学习，不包含额外领域预训练、CRF、观点词抽取或大模型伪标签。实际收益应通过验证集选择和独立测试集评估确认。

## 参考

- 模型：https://huggingface.co/microsoft/deberta-v3-base
- DeBERTa-v3 论文：https://arxiv.org/abs/2111.09543
- 方面辅助句与句对分类：https://aclanthology.org/N19-1035/
- 模型 API：https://huggingface.co/docs/transformers/model_doc/deberta-v2
- SemEval：https://alt.qcri.org/semeval2014/task4/
- MAMS：https://github.com/siat-nlp/MAMS-for-ABSA

方面注意力、多任务学习、局部上下文均有相关研究。本项目组合这些机制用于课程实验，不声称是全新算法。
