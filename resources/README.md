# 原始 XML 离线包

`absa_xml_bundle.zip` 包含本项目已获取的 10 个原始 XML，供网络不可用时恢复数据：

- SemEval-2014 Restaurant/Laptop 各自的训练与测试 XML，共 4 个。
- MAMS ATSA/ACSA 的 train、val、test，共 6 个。
- MAMS 作者仓库的 README 和 LICENSE，保留来源说明。

从项目根目录运行 `python scripts/download_datasets.py --offline`，脚本只读取需要的 XML 并核对内置 SHA-256，不执行 ZIP 内的代码，不直接解压任意 ZIP 路径。已有有效文件保留。离线包没有修改标签、字符位置或数据划分。

来源与使用条件：

- SemEval 官方任务与数据说明：https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
- SemEval 下载镜像：https://huggingface.co/datasets/alexcadillon/SemEval2014Task4 ，版本 `3fd43769112ef6517c263e303fbb7dc3002c35df`。
- MAMS 作者仓库：https://github.com/siat-nlp/MAMS-for-ABSA ，版本 `cddcdb0f423b3fdb2c76b70744737abed3a00d17`。

使用数据时遵循原发布方说明并引用对应论文。打包不改变原始数据的使用条件。
