# 政治访谈回答清晰度与跨场景迁移

深圳大学自然语言处理课程。实验使用 QEvasion 和 GLUE QNLI，按采访与维基文章来源隔离数据，比较词汇、预训练句向量、联合微调及跨场景适配。

## 复现

从仓库目录执行。环境为 Python 3.12.13，依赖版本见 `requirements-run.txt`。实测使用 CUDA 13.0 与 RTX 5060 Laptop GPU。`data_sources.json` 固定原始数据发布地址和校验值，`configs/adapter.json` 记录字段和直接迁移的类别映射。

```powershell
uv venv --python 3.12.13 .venv
$py = Resolve-Path .venv/Scripts/python.exe
uv pip install --python $py --index https://download.pytorch.org/whl/cu130 'torch==2.14.1+cu130'
uv pip install --python $py -r requirements-run.txt
& $py prepare_data.py
```

下载超时时，可从 `data_sources.json` 对应的发布地址获取原文件，按清单文件名放入 `data/raw/`，再运行 `& $py prepare_data.py --offline`，由脚本核对 SHA-256 并重建划分。

训练前先从课程附件 `黄宇科-2025150266-反事实核验附件.zip` 解出 `configs/counterfactual_reviewed.csv`，放到项目同名路径。随后执行训练和逐样本比较。每条比较命令会重算两边的指标，核对样本 ID 与金标，并记录硬预测变化数、Macro-F1 最大绝对差和类别概率最大绝对差。

```powershell
& $py src/run_experiments.py --output results/reproduced_main --device auto
& $py evidence/verify_predictions.py --first results/review_run_1 --second results/reproduced_main --out results/reproduced_main/prediction_audit.json
& $py src/run_pair_finetuning.py --output results/reproduced_joint --device auto
& $py evidence/verify_predictions.py --first results/pair_run_1 --second results/reproduced_joint --out results/reproduced_joint/prediction_audit.json
& $py src/run_source_truncation.py --output results/reproduced_truncation --device auto
& $py evidence/verify_predictions.py --first results/truncation_run_1 --second results/reproduced_truncation --out results/reproduced_truncation/prediction_audit.json
& $py evidence/verify_counterfactual.py --first results/review_run_1/counterfactual_public.csv --second results/reproduced_main/counterfactual.csv --out results/reproduced_main/counterfactual_audit.json
```

三组分类对照分别覆盖 33、2、3 组预测。反事实单独核对 19 条题目、四个模型、类别支持数、计数和逐条预测，并核对等义问题与改变后的问题是否来自所附题集，以及回答是否与原始样本一致。相同环境的对照以硬预测、Macro-F1 和概率均零差异为参照，其他环境以核验 JSON 的实测差值为准。

单独复算随仓结果可直接执行下一条命令。它读取随仓结果的三个目录，适合核查报告表格。

```powershell
& $py evidence/verify_submission.py
```

明确回答的类权重对照使用相同的分组验证集和首尾输入。运行下列命令可复核两个种子、两种权重的四组预测。

```powershell
& $py src/run_source_class_weighting.py --output results/reproduced_weighting --device auto
& $py evidence/verify_predictions.py --first results/weighting_run_1 --second results/reproduced_weighting --out results/reproduced_weighting/prediction_audit.json
```

## 查看结果

| 内容 | 路径 |
| --- | --- |
| 基线、迁移、数据效率、层级与反事实汇总 | `results/review_run_1/` |
| 源域和目标域联合微调 | `results/pair_run_1/` |
| 源域等长首尾截断对照 | `results/truncation_run_1/` |
| 源域明确回答类权重对照（验证集） | `results/weighting_run_1/` |
| 分组配对区间、词项贡献和结果核验 | `evidence/` |
| 干净检出复跑记录 | `evidence/clean_clone_reproduction.json` |
| 数据统计与文件校验 | `data/processed/` |

`review_run_1` 是报告引用的主实验。`revision_run_1` 是较早一轮的本地过程记录，复现比较以 `review_run_1` 为准。各实验目录含 `metrics.json` 和 `predictions/`。预测由样本 ID 连接金标、预测标签和各类概率。论文作者的公开计分复跑见 `paper_reproduction/README.md`。
