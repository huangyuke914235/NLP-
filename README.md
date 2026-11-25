# 政治访谈回答清晰度与跨场景迁移

深圳大学自然语言处理课程。实验使用 QEvasion 和 GLUE QNLI，按采访与维基文章来源隔离数据，比较词汇、预训练句向量、联合微调及跨场景适配。

## 复现

从仓库目录执行。环境为 Python 3.12.13，依赖版本见 `requirements-run.txt`。实测使用 CUDA 13.0 与 RTX 5060 Laptop GPU。

```powershell
uv venv --python 3.12.13 .venv
$py = Resolve-Path .venv/Scripts/python.exe
uv pip install --python $py --index https://download.pytorch.org/whl/cu130 'torch==2.14.1+cu130'
uv pip install --python $py -r requirements-run.txt
& $py prepare_data.py
& $py src/run_experiments.py --output results/reproduced_main --device auto
& $py src/run_pair_finetuning.py --output results/reproduced_joint --device auto
& $py src/run_source_truncation.py --output results/reproduced_truncation --device auto
& $py evidence/verify_submission.py
```

`data_sources.json` 固定发布地址和输入校验值，`configs/adapter.json` 记录字段与直接迁移类别映射。完整反事实题集在课程附件 `黄宇科-2025150266-反事实核验附件.zip`，将其中 `configs/counterfactual_reviewed.csv` 放到项目同名路径即可运行文本级核验。

## 查看结果

| 内容 | 路径 |
| --- | --- |
| 基线、迁移、数据效率、层级与反事实汇总 | `results/review_run_1/` |
| 源域和目标域联合微调 | `results/pair_run_1/` |
| 源域等长首尾截断对照 | `results/truncation_run_1/` |
| 分组配对区间、词项贡献和结果核验 | `evidence/` |
| 数据统计与文件校验 | `data/processed/` |

各实验目录含 `metrics.json` 和 `predictions/`。预测由样本 ID 连接金标、预测标签和各类概率。论文作者的公开计分复跑见 `paper_reproduction/README.md`。
