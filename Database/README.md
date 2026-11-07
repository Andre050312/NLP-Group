# 餐厅三分类数据集合

本目录只保留餐厅领域的 positive、neutral、negative 三分类数据。所有文件均为方面级情感，不是整句情感标签。

| 目录 | 类型 | 训练方面目标 | 验证方面目标 | 测试方面目标 |
|---|---|---:|---:|---:|
| Restaurant3 | SemEval-2014 显式方面派生版 | 3602 | 从训练集划分 | 1120 |
| Restaurant2015 | SemEval-2015 显式方面派生版 | 1188 | 从训练集划分 | 525 |
| Restaurant2016 | SemEval-2016 显式方面派生版 | 1714 | 从训练集划分 | 601 |
| MAMS/data/MAMS-ATSA/raw | 多方面情感，原始三分类 | 11186 | 1332 | 1336 |
| MAMS/data/MAMS-ACSA/raw | 方面类别情感，原始三分类 | 7090 | 888 | 901 |

数量由实际文件解析，指方面目标而非句子。MAMS-ACSA 没有方面字符位置，不支持当前方面词模型入口。

## 来源与转换

- Restaurant3：来自 SemEval-2014 Restaurant，仅移除 conflict 标注；原始混合标签源文件已移出项目，不保留副本。
- Restaurant2015/2016：从官方任务的英文餐厅标注镜像获取，转换为统一 aspectTerms XML。排除 NULL 隐式对象、OutOfScope 句子、无效位置、同跨度不同情感及重叠跨度；相同跨度相同情感的多条意见折叠为一个方面。原文和有效字符偏移不改写。
- MAMS：来自作者仓库，本身即为三分类，原始训练/验证/测试划分与文件字节保持不变。
- Restaurant3 的分类测试目标仍为 1,120 个；由于去掉了 conflict 方面，joint 的抽取目标由旧原始版的 1,134 个变为 1,120 个。新旧抽取/端到端分数的协议不同，不直接混比。
- 各数据目录的 manifest 和 `scripts/restaurant_sources.json` 记录源版本、来源地址、原始及派生哈希、过滤统计。
- 这些派生版本不应描述为未经处理的官方原始评测集。使用数据须遵循原发布方说明并引用论文。

## 下载与恢复

将项目的 `resources/restaurant_datasets_3class.zip` 一并复制到目标电脑，在项目根目录运行：

```powershell
python scripts/download_datasets.py --offline
python scripts/download_datasets.py --check
```

离线恢复和检查应完成 12 个 XML。自动恢复标准目录；已有匹配哈希的文件跳过。若要获得 JSONL 与目录说明，可将同一个 ZIP 解压到 Database（目标目录不存在时操作）。

联网下载也只会安装三分类版本，原始混合标签响应仅用于内存转换，不保存为数据集：

```powershell
python scripts/download_datasets.py
```

可选 `--dataset restaurant`、`restaurant2015`、`restaurant2016`、`mams`。目录或 ZIP 导入使用 `--import-from`，超时与重试用 `--timeout`、`--retries`。

## 清理旧数据

只有在新的 12 个 XML 全部通过校验后，清理脚本才允许移除已知旧路径 ASAP、SemEval2014、IMDB 和原混合离线包。它不删除模型代码或 runs：

```powershell
python scripts/clean_non_threeclass_data.py
python scripts/clean_non_threeclass_data.py --apply
```

第一条预览，第二条执行；记录见 cleanup_report.json。

## 禁止直接合并基准

`overlap_report.json` 记录按原句完全匹配的跨数据集重合。尤其是 Restaurant2016 的训练集与 Restaurant2015 的测试集有 662 句重合。Restaurant3 与 MAMS 也存在训练/测试交叉重合。默认各自独立训练；如果合并，须先排除与全部留出测试及验证文本重复的训练样本，并记录新的数据协议，不能把重合数据当成独立新样本。

## 来源

- SemEval-2014：https://alt.qcri.org/semeval2014/task4/
- SemEval-2015：https://alt.qcri.org/semeval2015/task12/
- SemEval-2016：https://alt.qcri.org/semeval2016/task5/
- XML 镜像：https://huggingface.co/alexcadillon
- MAMS：https://github.com/siat-nlp/MAMS-for-ABSA
