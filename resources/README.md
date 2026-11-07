# 餐厅三分类离线包

- `restaurant_datasets_3class.zip`：全量集合，包含 Restaurant3、Restaurant2015、Restaurant2016、MAMS-ATSA 与 MAMS-ACSA，仅含 negative、neutral、positive。
- `restaurant_3class.zip`：仅 SemEval-2014 Restaurant 的三分类派生版本。

从项目根目录运行 `python scripts/download_datasets.py --offline` 可按固定哈希恢复全量集合的 12 个 XML，不需要联网。同步项目时不要漏掉脚本、restaurant_sources.json、restaurant_format.py 和这个 ZIP。

各目录的 manifest 和包内说明记录来源与转换。保留官方训练/测试归属，但2014过滤 conflict；2015/2016 过滤隐式/不适用/冲突方面，属于明确记录的派生版本。MAMS 原始三分类字节不变。

这些基准之间存在大量重复文本，不直接混合训练。参见 Database/overlap_report.json。使用遵循原发布方说明并引用论文；打包不改变原数据的使用条件。
