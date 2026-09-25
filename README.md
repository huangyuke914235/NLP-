# 对话问答回避检测与跨场景迁移分析

自然语言处理期末大作业的代码、结果与实验报告。项目以 QEvasion 的问答清晰度三分类为主任务，比较多数类、TF-IDF + 逻辑回归和冻结的 MiniLM 表示 + 逻辑回归；另在 QNLI 上测试流程适配与源域直接迁移，并检查九类细粒度标签到三类粗标签的层级映射。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `自然语言处理期末大作业_按模板完成.docx` | 按课程模板撰写的 15 页实验报告 |
| `requirements.txt`、`DATA_SOURCES.md` | Python 依赖清单及数据来源、快照校验信息 |
| `code/common.py` | 数据适配、评估与结果保存的公共函数 |
| `code/01_eda.py`～`code/06_direct_transfer.py` | 数据分析、基线、语义表示、QNLI 适配、层级扩展与直接迁移实验 |
| `code/configs/` | QEvasion 与 QNLI 的字段、标签配置 |
| `code/data/` | 实验使用的 QEvasion 与 QNLI 原始数据本地副本 |
| `code/deps/` | 项目内第三方 Python 依赖副本；适用于本机 Windows/Python 3.13，不等于完整 Conda 环境 |
| `code/figures/` | 报告使用的图 |
| `code/outputs/` | 已运行得到的指标 JSON、预测 CSV 和 `st_embeddings.npz` |
| `code/outputs/hierarchy/` | 标签层级与源域直接迁移的指标和预测 |

`st_embeddings.npz` 约 20 MB，供 `05_hierarchy.py` 直接复用。删除它后，可先运行 `03_plm_lr.py` 重新生成。

## 环境和数据

在项目使用的 Anaconda `pytorch` 环境中运行。所需 Python 包列于 `requirements.txt`。附带的 `code/deps/` 是项目本地依赖副本，实验脚本会优先搜索它；其中的编译扩展面向 Windows/Python 3.13，不能代替跨平台安装，也不是整个 Anaconda 环境。实测 NumPy、pandas、scikit-learn 与 datasets 可从包内加载；matplotlib 和 PyTorch 仍由当前 Anaconda 环境提供，`sentence-transformers` 在检查时尚未安装。从头运行 `03_plm_lr.py` 前须先按清单安装所缺的包。首次运行语义模型时，还需要从 [MiniLM 模型页面](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)下载权重；模型权重没有打包。

原始数据已放在 `code/data/`。来源、样本量、本地文件记录时间和校验值见 `DATA_SOURCES.md`。QEvasion 来自[发布者的数据集页面](https://huggingface.co/datasets/ailsntua/QEvasion)；QNLI 的 `train.tsv` 和 `dev.tsv` 来自脚本中记录的 GLUE 下载地址。

以下命令在 PowerShell 中执行；从仓库根目录进入 `code` 后再运行，以便 `configs/qevasion.json` 中的相对数据路径生效：

```powershell
conda activate pytorch
python -m pip install -r requirements.txt
cd code
```

包内已有 `code/data/qevasion_raw/` 和 `code/data/qnli_raw/`，通常不必重新下载。若本地数据缺失，可在 `code` 目录重新获取 QEvasion：

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
python -c "from datasets import load_dataset; load_dataset('ailsntua/QEvasion').save_to_disk('data/qevasion_raw')"
```

`05_hierarchy.py` 和 `06_direct_transfer.py` 必须能从该目录读取 QEvasion。如果远程数据版本或顺序变化导致缓存断言失败，重新运行 `03_plm_lr.py` 生成对应的 `st_embeddings.npz` 和预测文件，再运行 `05_hierarchy.py`。

## 运行顺序

在 `code` 目录下依次执行：

```powershell
python 01_eda.py
python 02_baselines.py
python 03_plm_lr.py
python 04_transfer_qnli.py
python 05_hierarchy.py
python 06_direct_transfer.py
```

`03_plm_lr.py` 会在 CPU 上加载 MiniLM，首次运行可能需要下载模型。`04_transfer_qnli.py` 会尝试下载 QNLI 数据；若下载地址不可用，请自行取得官方 GLUE QNLI 的 `train.tsv`、`dev.tsv`，放在 `code/data/qnli_raw/` 后重试。`05_hierarchy.py` 依赖 `03_plm_lr.py` 的向量缓存和预测；`06_direct_transfer.py` 依赖 `04_transfer_qnli.py` 的预测 CSV。

## 结果定位与说明

主要结果记录在 `code/outputs/metrics_*.json` 与 `code/outputs/hierarchy/metrics_*.json`，逐样本预测在同目录的 CSV 中。报告正文中的数字统一保留四位小数。当前保存的一次运行中，QEvasion 上的 TF-IDF + LR Macro-F1 为 0.4351，MiniLM + LR 为 0.4855；九类细标签概率汇总后的 Macro-F1 为 0.3504。QNLI 上目标域训练的 TF-IDF + LR Macro-F1 为 0.5741，仅源域训练并固定投影标签的结果为 0.3364。

这些是一次固定随机种子运行的结果，不是多次实验的均值。QEvasion 官方测试集缺少细粒度 `evasion_label`，因此报告没有声称测得测试集的九分类 F1。细粒度标签 `Implicit` 映射到粗类 `Ambivalent`，而非 `Clear Reply`。

## 发布前检查

报告封面的个人信息及第五节 GitHub 仓库链接尚未填写，请提交前补齐。提交包现已包含原始数据和项目内第三方依赖，但不包含完整 Conda 环境、PyTorch 安装包或 MiniLM 模型权重。公开发布原始数据、预测 CSV 和第三方包前，请核对各来源的许可与课程要求。整个包约 500 MB、文件数较多，建议通过 Git 命令行推送；[GitHub 文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)列有文件与仓库大小限制。
