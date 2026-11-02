# 情感分析数据集

本目录保存前面讨论的四个数据集：SemEval-2014 Restaurant、SemEval-2014 Laptop、MAMS、ASAP。保留下载仓库的原始结构、说明及许可证，没有修改标签或重新划分数据。

## 目录与规模

| 数据 | 路径 | 训练 / 验证 / 测试（句子或评论） |
|---|---|---|
| SemEval Restaurant | `SemEval2014/` 下训练 XML 和 `ABSA_Gold_TestData/Restaurants_Test_Gold.xml` | 3041 / 无 / 800 |
| SemEval Laptop | `SemEval2014/` 下训练 XML 和 `ABSA_Gold_TestData/Laptops_Test_Gold.xml` | 3045 / 无 / 800 |
| MAMS ATSA（方面词） | `MAMS/data/MAMS-ATSA/raw/` | 4297 / 500 / 500 |
| MAMS ACSA（方面类别） | `MAMS/data/MAMS-ACSA/raw/` | 3149 / 400 / 400 |
| ASAP（中文方面类别） | `ASAP/data/{train,dev,test}.csv` | 36850 / 4940 / 4940 |

样本数由下载文件实际解析统计。一句可能包含多个方面，因此句子数不等于分类样本数。ASAP 的 `*_sample.csv` 是示例子集，不能重复计入完整数据。

## 来源

- SemEval 官方说明：https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
- SemEval XML 下载镜像：https://huggingface.co/datasets/alexcadillon/SemEval2014Task4 （第三方镜像，并非官方托管）
- MAMS 作者仓库：https://github.com/siat-nlp/MAMS-for-ABSA
- ASAP 作者仓库：https://github.com/Meituan-Dianping/asap

各仓库的精确 commit、每个 XML/CSV 的 SHA-256、大小和解析统计见 `manifest.json`。数据使用与论文引用应遵循原发布方说明；镜像可下载不代表额外授予数据使用权限。

## 使用建议

先使用 SemEval Restaurant 训练与测试，在原训练集内部按句子划分验证集。相同句子的不同方面不能分到不同集合。MAMS 使用原有 train/val/test 划分。ASAP 使用原有 train/dev/test 划分。

SemEval 含 positive、negative、neutral、conflict，需明确四分类或过滤 conflict 的协议。ASAP 标签为 1=正面、0=中性、-1=负面、-2=未提及；未提及不能直接当中性。

MAMS 和 ASAP 同时保留了作者提供的代码及文档，仅下载，未安装依赖或执行仓库代码。
