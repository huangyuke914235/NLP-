# 政治问答清晰度与跨场景迁移实验

本项目由 QEvasion 与 QNLI 原始公开数据重新构建，输出固定测试集上的预测、结果表可用的指标、图、错误核查及独立复跑证据。本地 Word 文档按实验报告和考试答题纸分开交付。代码和可公开的实验结果对应 [课程仓库](https://github.com/huangyuke914235/NLP-)。

## 关键结果

| 任务和固定评估 | 方案 | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| QEvasion 严格分组，308 条 | 多数类 | 0.6688 | 0.2672 |
| QEvasion 严格分组，308 条 | 验证选定 TF-IDF + LR | 0.5779 | 0.4242 |
| QEvasion 严格分组，308 条 | 冻结 MiniLM + LR | 0.5844 | 0.3849 |
| QEvasion 严格分组，308 条 | MiniLM 句对联合微调，保留回答开头 | 0.6461 | 0.3936 |
| QEvasion 严格分组，308 条 | MiniLM 句对联合微调，等长首尾 | 0.6786 | 0.4576 |
| QNLI 固定开发集，5,463 条 | 零目标监督直接映射 | 0.4979 | 0.4825 |
| QNLI 固定开发集，5,463 条 | 目标域 TF-IDF + LR | 0.5949 | 0.5946 |
| QNLI 固定开发集，5,463 条 | 冻结 MiniLM 加源分数适配 | 0.5790 | 0.5782 |
| QNLI 固定开发集，5,463 条 | 目标域 MiniLM 句对联合微调 | 0.8256 | 0.8255 |

所有主表从保存的样本预测复算。完整预测包含稳定 ID、金标、硬预测及按标签名排列的概率。源域等长首尾方案以验证集选定，在原联合微调对照中改善了测试 Accuracy 和 Macro-F1。其与 TF-IDF 的固定测试差异区间跨零，验证选定的源域主报告模型仍为 TF-IDF。QNLI 联合微调提高两类 F1。

## 隔离依赖

实跑平台为 Windows 11、Python 3.12.13、RTX 5060 Laptop GPU、PyTorch 2.14.1+cu130、scikit-learn 1.6.1 与 Transformers 4.53.3。`requirements-run.txt` 固定除 PyTorch 外的已安装依赖。以下从项目目录执行。GPU 版与同设备 BF16 决定本次复跑口径；使用 `--device cpu` 可以运行，其批量和浮点路径不同。

```powershell
uv venv --python 3.12.13 .venv
uv pip install --python .venv/Scripts/python.exe --index https://download.pytorch.org/whl/cu130 'torch==2.14.1+cu130'
uv pip install --python .venv/Scripts/python.exe -r requirements-run.txt
$py = Resolve-Path .venv/Scripts/python.exe
$env:PYTHONUTF8 = '1'
$env:PYTHONHASHSEED = '20251125'
```

PyTorch 也可配合当前设备选择兼容的安装包。MiniLM 采用固定修订 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，实际权重 SHA-256 为 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`。

## 标准复跑与完整课程附件

`data_sources.json` 提供五个原始文件的 SHA-256、QEvasion 修订和下载地址。`prepare_data.py` 从固定发布地址恢复缺失文件，逐份核对 SHA-256。已有经过校验的原始文件时可加 `--offline`。如下载终止后留下校验不符的本地文件，应以原发布地址的完整文件替换再运行。模型代码和必要配置均在仓库。以下命令从仓库根目录执行，输出目录使用新的空目录。

```powershell
& $py prepare_data.py
& $py evidence/check_dataset_loader.py
& $py src/run_experiments.py --output results/my_review_1 --device auto
& $py src/run_pair_finetuning.py --output results/my_pair_1 --device auto
& $py src/run_source_truncation.py --output results/my_truncation_1 --device auto
```

上面的标准复跑涵盖所有基础模型、迁移、层级、标注效率、联合微调和等长首尾对照。GitHub 发布版的 `results/review_run_1/counterfactual_public.csv` 包含 ID、原金标、改问预期、变体类别概率与模型输出，`counterfactual.json` 提供行为汇总。无文本复算可运行 `& $py evidence/verify_reviewed_experiments.py --out evidence/my_review_audit.json`。

**完整文本级核验。** 从课程附件 `黄宇科-2025150266-反事实核验附件.zip` 取出 `configs/counterfactual_reviewed.csv`，放到克隆后的同名相对路径。文件包括 19 条经复核的等义问题、异主题／可从原回答取证／同范围拒答的改问、预期标签与逐条理由。完整实际预测也在附件内的 `results/review_run_1/counterfactual.csv` 和复跑同名文件。主程序按 ID 从原始数据恢复固定回答，用下列命令同样可以明确指定题集。

```powershell
& $py src/run_experiments.py --counterfactual configs/counterfactual_reviewed.csv --output results/my_review_with_cases --device auto
```

基线审计归档 `results/run_1` 与 `results/run_2` 各 32 个预测文件，联合微调 `results/pair_run_1` 和 `results/pair_run_2` 各 2 个。当前迁移重构与扩充题集采用 `results/review_run_1` 和 `results/review_run_2`，均有 33 个预测文件，其中 32 个与旧归档逐字节一致，新增包容映射 1 个。首尾对照 `results/truncation_run_1` 和 `results/truncation_run_2` 每轮有 2 个候选验证预测及 1 个验证选定的测试预测。实测各轮硬预测一致，保存概率绝对差为 0。标准工作流与完整反事实汇总均可核验。

```powershell
& $py prepare_data.py --data-dir replay_data
& $py src/run_experiments.py --data replay_data/processed --output results/my_review_2 --device auto
& $py src/run_pair_finetuning.py --data replay_data/processed --output results/my_pair_2 --device auto
& $py src/run_source_truncation.py --data replay_data/processed --output results/my_truncation_2 --device auto
& $py evidence/verify_predictions.py --first results/my_review_1 --second results/my_review_2 --out evidence/my_prediction_audit.json
& $py evidence/verify_predictions.py --first results/my_pair_1 --second results/my_pair_2 --out evidence/my_pair_audit.json
& $py evidence/verify_source_truncation.py --primary results/my_truncation_1 --replay results/my_truncation_2 --baseline results/my_pair_1 --output evidence/my_truncation_audit.json
```

## 复核证据

- `evidence/reviewed_experiment_audit.json` 包含配置重构前后 32 份预测 SHA-256、包容映射复跑、新版反事实行为复算。`evidence/review_prediction_audit.json` 独立重算新版各模型指标。`evidence/replay_chain.json` 保留首次完整运行的五个输入、十个派生文件及扩展输出的复跑记录。
- `evidence/source_truncation_audit.json` 验证截断策略与原生 `only_second` 的一致性、验证候选、固定测试及两轮复跑。`results/truncation_run_1/slices.json` 提供按真实 tokenizer 截断情况分组的支持数、召回与混淆矩阵。
- `evidence/paired_group_bootstrap.json` 是固定预测的 10,000 次配对来源组重抽样。QEvasion 按采访 URL，QNLI 按文章组。该区间刻画固定模型对评估来源的探索性不确定性，训练抽样波动另见数据效率表。
- `evidence/tfidf_error_contributions.json` 记录 0119、0240、0000 两类 logits 差的截距与词项分解，独立重建与基线预测核对。`evidence/figures/` 中的终端输出图附原始 `.log` 和 `.vt`，`capture_review_terminal.py` 的额外依赖为 Windows 的 `pywinpty==2.0.15` 和 `pyte==0.8.2`。
- [论文作者公开计分组件复现](paper_reproduction/README.md)固定公开代码与输入修订，记录 31 项注入样本与 36 个匹配子问题的分母区别。

## 评估口径与文件

- `prepare_data.py` 及 `data_sources.json` 校验原始输入、字段、缺失、采访或文章组、完全相同和高度相似问答。QEvasion 的 `official` 与 `grouped` 共用固定 308 个测试 ID；按采访 URL 和重复连通分量排除相关来源后，使用 2,274 条训练和 584 条验证。QNLI 用有标签官方 dev 作本地测试，根据 SQuAD 文章来源隔离训练验证，最终训练 12,000 条、验证 1,999 条、评估 5,463 条。100% 标注预算指该 12,000 条实验训练池。
- `configs/adapter.json` 是主程序读取的唯一概率映射入口，并随运行保存为 `adapter_used.json`。默认规则将 Clear Reply 对应 entailment、另两类聚合为 not_entailment；额外包容规则仅用于敏感性检验。`configs/experiment.json`、`configs/pair_finetuning.json`、`configs/source_truncation.json` 记录选参、训练与类序。划分种子为 2026；模型种子为 20251125；数据效率抽样种子为 20251125、20251126、20251127。
- `results/review_run_1/metrics.json`、`predictions/` 与 `selection.json` 提供新版冻结系列、控制变量与扩展。`data_efficiency.json`、`hierarchy.json` 及抽样 ID 文件记录二项扩展，`counterfactual.json` 提供各行为计数。细粒度直接评估只在有公开细标签的验证侧，粗类聚合仍在同一固定测试侧。`results/pair_run_1/` 提供联合句对模型的逐轮验证、测试预测和环境信息。
- `data/processed/*_stats.json`、`*_manifest.json` 和 `data/figures/` 提供数据分析证据。`evidence/` 包括独立指标复算、分层统计、人工误判验真、复跑证据和主要图表。评分项和报告落点见本地 `评分项与交付物对应表.md`。
- QEvasion 数据卡标明 CC BY-NC-ND 4.0。GLUE 数据集卡指向各子任务条款，QNLI 的来源 SQuAD 1.1 标明 CC BY-SA 4.0。数据来源及修订保存在配置中，下载由材料使用方从原始发布地址完成。MiniLM 模型卡标明 Apache 2.0。
