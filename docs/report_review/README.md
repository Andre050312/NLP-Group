# 报告修订记录与复现入口

日期：2026-11-17。组别：Group 9。成员与学号按本人提供的信息填写；
具体分工按用户要求稍后确认，Word 中保留明确占位，不编造个人贡献。

## 文件

- `../../reports/Group_9_ABSA_Report_Review.docx`：11 页英文 Word 审阅稿。
- `../report_draft.md`：正文源文件；`references.md`：11 项一手资料阅读记录。
- `recorded_results.json`：从组员结果记录人工转录的数值，供图表与复核使用。
- `evidence_audit.json`：XML 计数、真实 gold 案例、来源文件 SHA256 与算术复核。
- `figures/`：3 张原创图，不包含未经确认许可的论文截图。
- `../../scripts/build_report.py`：Word 和图表生成器；使用托管文档运行时。
- `../../scripts/audit_report_evidence.py`：不需要模型的证据检查。

## 修正了什么

1. 把“10/10 提升”改成 8 次严格提升、2 次持平。符号检验剔除 ties，
   不把两个配置加五个共享 seed 当作十个独立实验，不作 pooled 显著性结论。
2. 核查出旧 ordinal loss 和 inference 不一致。保留旧 checkpoint 数学行为，
   新增独立 `ordered` 配置并验证训练、保存、加载；没有给它虚构实验成绩。
3. 新的 offset/CV bootstrap 改为成组抽取 exact-text 句子，两个规则共用抽样。
   旧结果不冒充新算法重算的区间，报告不复用未经重算的旧 aspect-row 区间。
4. 明确 argmax 不等于 P(neutral)>0.5；概率阈值扫描与 logit-offset 使用的
   排序可以不同。“同一测试标签上的最佳阈值”不是数据集上限。
5. 不将 BERT 与 DeBERTa 的预算、选模规则差异忽略掉；不以单 seed 否定一类方法；
   不把不显著当作等价，不把 CV 分数更低解释成已证明官方测试更容易。
6. 将邻居标签匹配写成关联证据；40 对候选样本全部待审核；clause 的机械不变性
   不当作理解能力。0.354→0.257 是约 27% 的下降，不是“一半”。
7. conflict 排除数为 train 91 个 aspect term、test 14 个；286 混入了 category。
   区分训练句子记录 1682 与唯一文本 1680。
8. 真实 XML 案例用于标注讨论，未伪装成模型的逐条预测或成功案例。

原始 `docs/neutral_results.md` 保持不变作为历史记录；其结论若与本复核冲突，
以本目录和修订正文的限定为准。数据文件在 Windows 的工作区可能带 CRLF，
evidence_audit 的 SHA256 明确针对当前原始文件字节，不等同于 manifest 中
下载/转换时 LF 文件的 hash；版本定位另由 source_commit 记录。

## 检查结果

- 全套离线测试：90 项通过，216.099 秒，含四种 head 的 tiny-model CLI roundtrip。
  使用测试夹具验证软件行为，不是 SemEval 成绩复现。
- 新的统计集成回归检查另见 `verification.md`。
- XML 标签数、两个 gold 案例、共享 seed 的均值/SD 和 ties 数已独立复核。
- Word 通过本机 Microsoft Word 隐藏实例实际导出并逐页查看。托管运行时没有
  LibreOffice，原版 render_docx.py 报 soffice.exe missing；仅转换步骤改用
  Word PDF，后续沿用技能 renderer 的 PDF→PNG 流程。没有安装替代运行时。
- 临时 PDF、页图和测试大日志位于 ignored `runs/`，不作为最终交付或训练证据。

## 如何复现

代码检查（项目 Python）：

```powershell
.venv/Scripts/python.exe scripts/audit_report_evidence.py
.venv/Scripts/python.exe -m unittest tests.test_evaluation_stats -v
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Word 构建必须使用 Codex workspace dependency loader 指定的托管 Python。
本机解析不到 loader 工具，已从运行时 runtime.json 确认 bundle 26.1112.11217：

```powershell
& 'C:/Users/Admin1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 scripts/build_report.py
```

构建器依赖 python-docx、Pillow，生成 Letter 纸张、可编辑正文和原生 Word 公式。
渲染优先使用 Documents skill 的 render_docx.py；无 LibreOffice 时先由 Word
导出当前 DOCX 为 PDF，再向 scripts/render_report_qa.py 显式提供该 PDF。
其他机器上的路径应从其托管 runtime.json 解析，不应照搬用户名路径。

## 提交前还需要小组补齐

- 逐人确认具体分工，替换 Author Contributions 的审阅占位。
- 上传轻量配置和原始预测证据：run_config.json/settings.json、history.json、
  test_metrics.json、validation/test 预测 JSONL、CV 汇总。权重不必塞进 Git。
  仓库 runs/ 被 ignore，建议把精简且核验过的证据放进一个明确的 evidence/ 目录，
  记录其所属 commit、seed 和 split；不要为凑完整而合成这些文件。
- 据此补三类完整 P/R/F1、真实模型成功与错误案例，并重算 cluster-bootstrap。
- 两位同学独立审核候选对，使用已有 review sheet/merge 工具汇总。
- 最终课程提交要求是报告 PDF 与 slides PDF 放入 Group_9.zip，截止
  2026-11-20 17:00；当前交付是用户要求的 Word 审阅稿，尚不是提交包。
