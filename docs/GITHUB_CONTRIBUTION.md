# 把这部分工作提交给小组

本地仓库位置：`Group project/NLP-Group`。
当前工作分支：`analysis/neutral-scope-diagnostics`，基于 `41fd254`。
提交和推送状态以 `git status`、`git log` 和远程分支为准；提交时使用自己的真实 Git 身份。
连接器的仓库读取权限与个人账号的 Git 推送权限可能不同。

## 推荐：个人分支 + Pull Request

如果组长已邀请你的 GitHub 账号为 collaborator，并且你接受了邀请，可直接把个人分支推到小组仓库。
Pull Request 的目标选 `Andre050312/NLP-Group:main`，来源选你的分支，让同学审核后合并。
这样可以清楚展示个人贡献，并避免覆盖同学当前主分支。

在本地仓库目录运行：

```powershell
git branch --show-current
git status --short
git diff --check
git config user.name
git config user.email
```

若显示的姓名/邮箱不是你的，使用 `git config user.name "你的姓名"` 和 `git config user.email "你的GitHub关联邮箱或noreply邮箱"`，
只设置本仓库；不要冒用同学的身份，也无需把邮箱写进报告。

先运行文档列出的检查，再暂存本贡献的明确路径：

```powershell
git add README.md absa_enhanced/cli.py absa_enhanced/cross_validation.py absa_enhanced/model.py absa_enhanced/pair.py neutral_diagnostics data/diagnostics docs/neutral_scope_study.md docs/GITHUB_CONTRIBUTION.md docs/pull_request_neutral_scope.md docs/neutral_scope_validation.md tests/test_neutral_diagnostics.py tests/test_cls_ablation.py
git diff --cached --stat
git commit -m "Add neutral sentiment diagnostics and CLS feature ablation"
git push -u origin analysis/neutral-scope-diagnostics
```

推送后在 GitHub 点 Compare & pull request，标题可用：
`Add neutral sentiment diagnostics and CLS feature ablation`。
正文已准备在 `docs/pull_request_neutral_scope.md`；检查测试状态与实际结果后再粘贴。
可以先建 Draft PR 表示候选样例仍需复核、真实模型结果待补齐，不需要等待所有实验跑完才让同学看到代码。

## 如果没有小组仓库的写权限

先在 GitHub 网页 Fork `Andre050312/NLP-Group` 到自己的账号，保留已有 origin，添加一个指向个人 fork 的远程：

```powershell
git remote add personal https://github.com/你的用户名/NLP-Group.git
git push -u personal analysis/neutral-scope-diagnostics
```

再从个人 fork 的此分支向小组仓库的 main 发 PR。无需改动或重建当前本地仓库。
若 personal 已存在，用 `git remote -v` 检查，不能直接重复添加或覆盖他人远程。
不要在缺少权限时反复尝试推送 main。

## 上传哪些内容

- 提交代码、候选诊断 CSV、复核记录、方法说明、测试与最终真实结果摘要。
- `runs/` 已被忽略。模型权重、缓存、虚拟环境不进入普通 Git 提交。
- 需要向报告提供真实结果时，将经过核对的小型指标表和说明单独复制到 `results/neutral_scope/`，注明命令、数据指纹、种子和模型目录。
- 预测原文若来自当前公开课程数据，仍应保留来源说明；不要把组内私有数据随便复制到公开仓库。
- PR 的 Files changed、提交记录和贡献说明是个人工作证据；贡献应按实际分工描述。

本目录的 patch/源码压缩包可作为无法推送时的备份，但不能代替以个人账号提交的贡献记录。

GitHub 官方流程：https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project
