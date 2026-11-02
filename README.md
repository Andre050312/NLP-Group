# BERT-base 方面级情感分析基线

这是给定方面词的情感分类（ATSA），不是自动方面抽取。输入为 `[CLS] 评论 [SEP] 方面词 [SEP]`，使用预训练 BERT 和线性分类头联合微调。默认使用 `google-bert/bert-base-uncased`，三分类：negative、neutral、positive；明确排除并统计 conflict，传 `--include-conflict` 可选四分类。

## 环境

在项目根目录运行，建议 Python 3.11+：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`requirements-lock.txt` 记录本机验证环境。首次训练需要联网下载模型，缓存放在项目内：

```bash
export HF_HOME="$PWD/.cache/huggingface"
```

## 训练与最终测试：Restaurant

```bash
python -m bert_baseline train \
  --train "Database/SemEval2014/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml" \
  --test Database/SemEval2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml \
  --output runs/bert_restaurant_seed42 \
  --epochs 3 --batch-size 16 --learning-rate 2e-5 --max-length 128 --seed 42
```

验证集从原训练集按原句抽取 15%；完全相同文本（即使 ID 不同）始终同组。官方 SemEval 的两个领域各发现一个训练/测试重复文本：指定 `--test` 时保留完整测试集，从训练/验证中移除与测试完全相同的句子，并在终端及 `run_config.json` 记录移除样本数。训练与验证之间若有重复则报错。验证 Macro-F1 选择最佳 checkpoint，训练结束后只对最佳模型做一次测试。输出目录必须为空，防止覆盖旧实验。

长句以标注方面的位置为中心截取上下文，保留完整方面；重复方面词依据字符位置定位。最大长度包含句对和特殊 token。无法保留目标时明确报错。

自动优先 CUDA，其次 Apple MPS，最后 CPU，可用 `--device cpu` 指定。若显存不足，先把 batch-size 减到 4 或 8。模型全参数微调；加 `--freeze-encoder` 可冻结编码器（这时建议另调分类头学习率，例如 1e-3）。

## MAMS-ATSA

```bash
python -m bert_baseline train \
  --train Database/MAMS/data/MAMS-ATSA/raw/train.xml \
  --validation Database/MAMS/data/MAMS-ATSA/raw/val.xml \
  --test Database/MAMS/data/MAMS-ATSA/raw/test.xml \
  --output runs/bert_mams_seed42 --seed 42
```

Laptop 用同目录的 `Laptop_Train_v2.xml` 和 `ABSA_Gold_TestData/Laptops_Test_Gold.xml`。ASAP 与 MAMS-ACSA 是方面类别任务，不支持本基线的数据读取器。

## 单独评估测试集

```bash
python -m bert_baseline evaluate \
  --checkpoint runs/bert_restaurant_seed42/best \
  --test Database/SemEval2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml \
  --output runs/bert_restaurant_evaluation
```

## 单条预测

```bash
python -m bert_baseline predict \
  --checkpoint runs/bert_restaurant_seed42/best \
  --text "The food was excellent, but the service was slow." \
  --aspect service
```

方面词需为原文准确子串；出现多次时通过 `--aspect-start` 指定字符起点。评估和预测默认沿用 checkpoint 上层 `run_config.json` 中的最大长度；移动模型目录时也应保留该文件或显式传 `--max-length`。

## 输出

- `best/`：最佳模型及 tokenizer。
- `run_config.json`、`split_manifest.json`：参数、标签计数、划分 ID。
- `history.json`：各 epoch 训练损失和验证指标。
- `training_summary.json`：训练时长与最佳验证 Macro-F1。
- `test_metrics.json`：Accuracy、固定标签集合上的 Macro-F1、各类 P/R/F1、混淆矩阵（行=真实、列=预测）。
- `predictions.jsonl`：原句、方面位置、真实标签、预测标签及概率，方便错误分析。

## 测试范围

测试真实下载 XML 的解析与字符跨度、conflict 过滤、按句划分及重复检测、已知混淆矩阵、保留目标的截断、微型 BERT 的前向/反向传播与保存加载，以及命令行训练→测试→预测的完整链路。微型模型由本地小词表和随机权重构造，不需联网，也不代表 BERT-base 实际效果。

正式结果请分别运行 seed 42、43、44，使用不同输出目录，报告均值和标准差。不要根据测试结果选择 epoch、标签协议或超参数。
