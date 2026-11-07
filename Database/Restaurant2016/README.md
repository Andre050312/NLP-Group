# SemEval-2016 Restaurant 三分类

方面级 negative/neutral/positive；保留原始训练/测试归属。
原始数据来源、版本、哈希和转换统计见 manifest.json。
2014 过滤 conflict；2015/2016 仅保留有明确字符跨度的方面，排除 NULL、无效位置、同跨度情感冲突与重叠标注；同跨度同情感按一个方面计数。未修改原文或合成标签。
验证集从训练来源划分，不能使用测试集调参。不能直接混合不同年份，详见上级 overlap_report.json。
官方任务：https://alt.qcri.org/semeval2016/task5/
