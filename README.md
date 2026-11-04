# Windows 情感分析：BERT 基线与 DeBERTa 多任务增强版

本项目共用一个 `requirements.txt`：

| 入口 | 用途 |
|---|---|
| `python -m bert_baseline` | 原 BERT-base 句对分类基线，需提供方面词 |
| `python -m absa_enhanced` | DeBERTa-v3-base 增强版，自动抽取方面并判断情感，也支持给定方面分类 |

增强版已编写。**本轮没有执行依赖安装、测试、模型下载、训练或推理**；以下命令由你在 Windows 上运行。效果是否超过基线需由实际实验验证。

## 1. 架构

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

- 一个编码器共享给抽取、情感两项任务。情感训练使用标注方面，自动推理使用预测方面，分别报告两种成绩。
- 全局上下文指当前窗口内全部有效 token；局部上下文为方面两侧各 16 个子词，可用 `--local-radius` 调整。
- 长句使用重叠窗口，检查每个标注方面至少被一个窗口完整覆盖；同一方面的跨窗口情感概率取平均，自动预测按原句字符跨度去重。
- 总损失：情感交叉熵 + `0.5 × BIO交叉熵`。BIO O/B/I 类权重为 0.3/1/1，padding 与特殊 token 不参与损失。
- 默认情感类别权重为训练类别频次的平方根倒数，标签平滑 0.05；可关闭进行消融。不能预先保证改善中性类别。
- 编码器学习率 `1e-5`，新增头 `1e-4`；warmup、学习率衰减、梯度裁剪、验证集早停。
- 默认梯度检查点；CUDA 自动选择 BF16（若支持）或 FP16，FP16 使用梯度缩放；支持梯度累积。
- 编码器可换为 `google-bert/bert-base-uncased`，在同结构下比较模型底座。

本版包含监督多任务学习，不包含额外领域预训练、CRF、观点词抽取或大模型伪标签。DeBERTa-v3 的预训练目标与 BERT MLM 不同，未直接套用普通 MLM 继续预训练。

## 2. Windows 安装：沿用 D:\NLP

PowerShell 中运行，项目放在 `C:\Users\ROG\Desktop\NLP-Group`：

```powershell
conda activate D:\NLP
cd "C:\Users\ROG\Desktop\NLP-Group"
python -c "import sys; print(sys.executable)"
```

应输出 `D:\NLP\python.exe`。仅当环境尚未创建时先执行：

```powershell
conda create --prefix D:\NLP python=3.12 pip -y
conda activate D:\NLP
```

安装全部依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

只有一个依赖入口；原 Mac `requirements-lock.txt` 已移除。Windows 使用 `torch==2.10.0+cu128`，requirements 已配置官方 CUDA 12.8 包索引，针对你的 RTX 5060 Laptop 与现有驱动。Transformers 固定到 `4.57.1`，可能将原环境的 5.x 降级，以固定接口。两个入口共用这些依赖。DeBERTa tokenizer 需要 sentencepiece 和 protobuf，不能跳过。无须额外安装 torchvision、torchaudio 或 CUDA Toolkit。

检查 GPU：

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x=torch.randn(128,128,device='cuda'); print((x@x).mean().item())"
```

将缓存放到 D 盘（新终端需要再次设置）：

```powershell
$env:HF_HOME = "D:\NLP-cache\huggingface"
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

`--dataset restaurant`、`laptop`、`mams` 自动定位数据。若未下载，仅在目标目录不存在时运行：

```powershell
git clone --depth 1 https://huggingface.co/datasets/alexcadillon/SemEval2014Task4 Database/SemEval2014
git clone --depth 1 https://github.com/siat-nlp/MAMS-for-ABSA.git Database/MAMS
```

协议：

- 情感三分类：negative、neutral、positive。conflict 不参加情感训练/分类评价；抽取任务仍学习 conflict 方面。
- SemEval 验证文本选择复用原基线划分逻辑与 seed。增强版保留原训练来源中没有三分类目标的句子，提供额外抽取背景监督；这些句子不增加三分类指标的样本数。
- MAMS 使用原 train/val/test。同一句不同方面不能跨集合；训练与验证文本重合时报错。
- 保留完整测试集；从训练、验证中排除与测试完全相同的文本并记录数量。测试成绩不用于选超参数或 epoch。
- 检查 XML 字符偏移、精确 token 对齐与窗口覆盖，不能表示的标注报错，不静默跳过。
- ASAP、MAMS-ACSA 是类别任务，不支持本版读取器；跨数据集实验先检查重复，不直接拼接随机划分。

## 4. 运行测试（由你执行）

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

可选：先用独立目录跑 1 epoch：

```powershell
python -m absa_enhanced train --dataset restaurant --output runs/enhanced_restaurant_smoke --epochs 1 --batch-size 2 --grad-accum 4 --max-length 192 --stride 64 --device cuda
```

正式训练：

```powershell
python -m absa_enhanced train --dataset restaurant --encoder microsoft/deberta-v3-base --output runs/deberta_joint_restaurant_seed42 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --stride 64 --seed 42 --device cuda
```

默认微批量 2、累积 4，有效批量 8；默认启用梯度检查点与自动混合精度，适配 8GB 显卡的起始配置，实际显存需运行确认。首次需联网下载编码器。每个 epoch 会输出验证指标，按**给定正确方面的验证 Macro-F1**保存最佳模型，连续 2 轮未改善则早停；最终在最佳模型上做一次测试。

重点研究端到端系统时，可在独立实验改为 `--selection-metric end_to_end_f1`；不要根据测试表现反复切换标准。输出目录必须为空；没有实现中断续训，失败后应使用新目录重试。

也可运行启动脚本（从激活 D:\NLP 的 PowerShell 调用）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\train_windows.ps1 -Dataset restaurant -Output runs/deberta_joint_restaurant_seed42
```

该执行策略只用于这一次脚本进程。直接使用 `python -m ...` 不需要修改任何执行策略。

## 6. 查看指标与错误案例

```powershell
Get-Content .\runs\deberta_joint_restaurant_seed42\test_metrics.json
```

| 字段 | 含义 |
|---|---|
| `gold_aspect_sentiment.accuracy` | 给定正确方面的准确率，可与原 BERT 84.46% 比较 |
| `gold_aspect_sentiment.macro_f1` | 给定正确方面的 Macro-F1，可与原 BERT 76.57% 比较 |
| `aspect_extraction.f1` | 方面字符跨度严格匹配 F1，包含 conflict 方面 |
| `end_to_end.f1` | 字符跨度与情感同时正确的 F1，不能与分类 F1 混比 |

端到端三分类评价排除 gold conflict 对，并忽略**准确命中 conflict 标注跨度**的预测，其余错误预测照常计入假阳性。这是项目协议，不应直接对照不同评价协议的论文。Restaurant 的三分类目标预期与原基线相同（你运行的版本为 1,120 个），代码不硬编码该数。

`test_gold_predictions.jsonl` 保存给定正确方面的逐条情感预测，可检查中性被判为正面的案例。`test_end_to_end.jsonl` 保存每句的自动抽取结果。

## 7. 推理

自动抽取方面并预测：

```powershell
python -m absa_enhanced predict --checkpoint runs/deberta_joint_restaurant_seed42/best --text "The food was excellent, but the service was slow." --device cuda
```

指定方面：

```powershell
python -m absa_enhanced predict --checkpoint runs/deberta_joint_restaurant_seed42/best --text "The food was excellent, but the service was slow." --aspect service --device cuda
```

同词出现多次时，增加 `--aspect-start` 指定字符起点。自动模式可能漏抽或无输出，程序不会人工补全；置信度是未经校准的 softmax 概率。

单独评估：

```powershell
python -m absa_enhanced evaluate --checkpoint runs/deberta_joint_restaurant_seed42/best --test Database/SemEval2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml --output runs/deberta_restaurant_evaluation --device cuda
```

## 8. MAMS 与 Laptop 独立实验

```powershell
python -m absa_enhanced train --dataset mams --output runs/deberta_joint_mams_seed42 --epochs 6 --batch-size 2 --grad-accum 4 --seed 42 --device cuda
```

```powershell
python -m absa_enhanced train --dataset laptop --output runs/deberta_joint_laptop_seed42 --epochs 6 --batch-size 2 --grad-accum 4 --seed 42 --device cuda
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

- 显存不足：改为 `--batch-size 1 --grad-accum 8`；仍不足可用 `--max-length 128 --stride 32`。保留混合精度与检查点；重试换新输出目录。
- 模型下载失败：检查网络，或完整下载配置/权重/tokenizer 后将 `--encoder` 指向本地目录。
- Fast tokenizer 或 protobuf 错误：确认已安装统一 requirements，重开终端激活 D:\NLP，不跳过 tokenizer 依赖。
- 非有限损失：新目录尝试 `--precision no` 或更低学习率；全精度更占显存，降低 batch-size。
- 找不到模块：将整个 `absa_enhanced/` 同步到 Windows，并从项目根目录运行。
- Redirects not supported in Windows：通常是 PyTorch 日志提示，是否失败看最终退出码/traceback。

## 12. 原 BERT 基线

旧入口保留，既有实验无需覆盖。重新运行时用新目录：

```powershell
python -m bert_baseline train --model google-bert/bert-base-uncased --train "Database/SemEval2014/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml" --test Database/SemEval2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml --output runs/bert_restaurant_new_seed42 --epochs 3 --batch-size 4 --learning-rate 2e-5 --max-length 128 --seed 42 --device cuda
```

## 参考

- 模型：https://huggingface.co/microsoft/deberta-v3-base
- DeBERTa-v3 论文：https://arxiv.org/abs/2111.09543
- 模型 API：https://huggingface.co/docs/transformers/model_doc/deberta-v2
- SemEval：https://alt.qcri.org/semeval2014/task4/
- MAMS：https://github.com/siat-nlp/MAMS-for-ABSA

方面注意力、多任务学习、局部上下文均有相关研究。本项目组合这些机制用于课程实验，不声称是全新算法。
