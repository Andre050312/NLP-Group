# 本地验证记录

日期：2026-11-11。基线提交：`41fd25405c00b7bf00dcd6512da48e6c22dcad84`。
本地分支：`analysis/neutral-scope-diagnostics`。

## 环境

Windows，Python 3.12.10，PyTorch 2.14.1+cpu，Transformers 4.57.1，
tokenizers 0.22.2，scikit-learn 1.7.2，NumPy 2.5.3。CUDA 不可用。

项目内 `.venv` 使用 system-site-packages 复用 CPU PyTorch，并在该环境内安装项目固定的 Transformers 4.57.1 和 scikit-learn 1.7.2。
没有替换全局安装。此环境不是 requirements 中的 Windows CUDA 2.10.0 构建，因此不声称完成 GPU、混合精度或显存验证。
初始全局 Transformers 5.22.1 缺少旧 tokenizer 接口，不用修改项目去兼容该非目标版本。

## 已完成的检查

- UTF-8 环境下全量回归：**56 项通过**，耗时236.851秒，最终退出码0；涵盖原仓库与新增测试。
- `test_neutral_diagnostics.py`：7 项通过，包含已知答案的 AP/并列分数、分母、输入校验、置信度、分组 bootstrap、报告导出和跨结果覆盖检查。
- `test_cls_ablation.py`：4 项通过，包含 joint/pair 的梯度、新旧 checkpoint 还原、三种模型的成对推理，以及本地微型模型的命令行训练和 CV 参数转发。
- 40 对候选样例结构检查通过：30 对中性目标、5 对正面目标、5 对负面目标，全部初始 pending。
- 真实 Restaurant3 数据的两组3折计划均成功生成，fold_assignment.json 完全相同；排除1个与官方测试重复的句子，池内3040句、3601个方面目标、3034个独立文本组。
- 两组 fold_assignment.json 的 SHA-256：`7308ebd613d76535ab6b1b061fb2e9231d2c3e9fca4e629a2bda1ed3395e427f`。
- 编译检查与 `git diff --check` 通过。

原始 clone 没有填充 MAMS gitlink 的数据目录。为运行原仓库数据检查，从已提交离线 ZIP 恢复6个 MAMS XML，逐个匹配内置哈希；没有网络获取新数据或改写数据 manifest，这些数据不打进贡献包。

## 重现测试

激活项目环境。若在当前机器使用已准备好的隔离环境，可将命令中的 `python` 改为 `.\.venv\Scripts\python.exe`。

```powershell
New-Item -ItemType Directory -Path runs/test_tmp -Force | Out-Null
$env:TEMP = Join-Path $PWD.Path 'runs/test_tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest discover -s tests -v
```

设置工作区 TEMP 是当前受限测试环境的需要；PYTHONUTF8 确保中文目录下父子进程编码一致。
第一次未统一子进程 UTF-8 时，原 baseline CLI 测试发生日志解码错误；不通过改动基线代码掩盖这个环境问题。
全量重跑记录位于本地忽略的 `runs/full_tests_utf8.log`。

## 尚未验证的研究结论

没有组内真实 checkpoint、真实逐条预测或 bias/gate 原始曲线，不能重算之前给出的45%、0.963或中性 F1。
模型测试采用随机初始化的微型 BERT 与合成小数据，只验证代码流程。
本次没有训练真实 DeBERTa、没有运行6次正式消融训练、没有测量改进效果，也没有人工批准候选诊断标签。
报告不得将这些尚未完成的研究结果写成已验证事实。
