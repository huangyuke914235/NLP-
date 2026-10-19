# 对话问答回避检测与跨场景迁移分析

自然语言处理期末大作业的代码、结果与实验报告。项目以 QEvasion 的问答清晰度三分类为主任务，比较多数类、TF-IDF + 逻辑回归和冻结的 MiniLM 表示 + 逻辑回归；另在 QNLI 上测试流程适配与源域直接迁移，并检查九类细粒度标签到三类粗标签的层级映射。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `自然语言处理期末大作业.docx` | 期末大作业正文 |
| `实验报告_对话问答回避检测与跨场景迁移分析_提交版.docx` | 按实验报告提交模板编排的报告 |
| `requirements.txt`、`requirements-model-verified.txt`、`DATA_SOURCES.md` | 基本依赖、本次验证的模型组件版本及数据快照说明 |
| `code/common.py` | 数据适配、评估与结果保存的公共函数 |
| `code/01_eda.py`～`code/06_direct_transfer.py` | 数据分析、基线、语义表示、QNLI 适配、层级扩展与直接迁移实验 |
| `code/07_hierarchy_sensitivity.py`、`code/08_bootstrap_uncertainty.py` | 层级方案敏感性分析与测试集配对重抽样 |
| `code/09_review_validation.py` | 指标重算、来源 URL 审计、URL 分组留出及来源簇区间 |
| `code/10_check_environment.py` | 环境与模型离线预检；运行实验前先执行 |
| `code/configs/` | QEvasion 与 QNLI 的字段、标签配置 |
| `code/data/` | 实验使用的 QEvasion 与 QNLI 原始数据本地副本 |
| `code/deps/` | 项目内第三方 Python 依赖副本；适用于本机 Windows/Python 3.13，不等于完整 Conda 环境 |
| `code/model_deps/` | 本次验证的独立模型依赖副本，保留原 `deps/` 并优先加载模型组件 |
| `code/figures/` | 报告使用的图及补充方法示意图 |
| `code/outputs/` | 已运行得到的指标 JSON、预测 CSV 和 `st_embeddings.npz` |
| `code/outputs/hierarchy/` | 标签层级与源域直接迁移的指标和预测 |
| `code/outputs/review/` | 独立复核的 JSON 和日志，包括 URL 分组留出与来源簇 bootstrap |
| `code/outputs/review/replay_evidence/` | 01—06 完整复现的真实日志、环境、进程记录及新旧输出比较 |
| `code/outputs/review/output/playwright/` | 实验报告图3的日志展示页、真实浏览器截图及来源校验清单；保留采集时的目录名，不是待清理缓存 |

`st_embeddings.npz` 约 20 MB，供 `05_hierarchy.py` 直接复用。删除它后，可先运行 `03_plm_lr.py` 重新生成。

`code/figures/fig4_frozen_lr_pipeline.png` 是补充方法示意图，未纳入两份提交报告；文件名中的 `fig4` 不对应报告中的图4。实验报告图3的截图与展示页保留在上表所列证据目录，原始运行日志位于 `replay_evidence/`；不改名以免破坏既有来源记录。

## 环境和数据

本次复核使用 Windows 64 位、Python 3.13.15 和 Anaconda `pytorch` 环境；03 脚本在 CPU 上运行，不要求 CUDA GPU。所需基本包见根目录 `requirements.txt`，该文件引用 `requirements-model-verified.txt` 中六个已验证的模型组件版本：sentence-transformers 5.1.1、transformers 4.57.1、tokenizers 0.22.1、safetensors 0.6.2、huggingface-hub 0.36.0、regex 2025.9.18。这些是 2026-10-18 复核环境的记录，不是原始实验的完整历史锁文件。

保留的 `code/deps/` 和新增 `code/model_deps/` 是本机依赖副本，含平台相关二进制包，适用范围为已验证的 Windows/Python 3.13 环境。`common.py` 优先复用解释器中已可正常导入的 NumPy，再登记 `deps/` 并把 `model_deps/` 置于搜索路径首位；若解释器没有安装 NumPy，则在登记本地依赖后回退加载项目副本。这样既保留现有环境使用的 NumPy，也避免缺少 NumPy 时在登记依赖前中断。其余尚未导入的依赖按 `model_deps/`、`deps/`、解释器环境的顺序查找，防止旧目录中的同名模型组件覆盖验证版本。其他平台应安装适配平台的依赖，不能直接套用 Windows 二进制副本。

MiniLM 权重未包含在仓库中，来源为 [MiniLM 模型页面](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)。本次复现离线读取的缓存修订为 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，环境预检也显式检查该修订。首次准备模型需能访问模型发布方；默认预检不会下载，只有显式加入 `--allow-download` 才允许下载。全新安装时的其他包版本与上游数据、模型状态仍可能变化，仓库中的快照、特征、预测和比较记录用于核对本次报告结果。

原始数据已放在 `code/data/`。来源、样本量、本地文件记录时间和校验值见 `DATA_SOURCES.md`。QEvasion 来自[发布者的数据集页面](https://huggingface.co/datasets/ailsntua/QEvasion)；QNLI 的 `train.tsv` 和 `dev.tsv` 来自脚本中记录的 GLUE 下载地址。

QEvasion 训练集包含 287 个来源 URL、175 个不同的标题文本；175 不是访谈数量。测试集保留 56 个完整来源 URL，其中 39 个与训练集重合，对应 118/308 条测试样本（38.31%）。因此，官方测试成绩不应被描述为完全来源独立的泛化结果；测试集虽缺 `title`、`president`，仍可按 `url` 复核来源。

以下命令在 PowerShell 中执行；从仓库根目录进入 `code` 后再运行，以便 `configs/qevasion.json` 中的相对数据路径生效：

```powershell
conda activate pytorch
cd code
python 10_check_environment.py
```

若当前环境缺少基本依赖，可在仓库根目录执行 `python -m pip install -r requirements.txt`，再进入 `code` 运行预检。已有可用 PyTorch 可复用，不必安装本机相同的 CUDA 构建。模型尚未缓存时，运行 `python 10_check_environment.py --allow-download`；下载能否完成取决于网络与发布方服务。预检结果写入 `code/outputs/environment_verified.json`。仓库保留一次实际离线检查记录，其时间和导入位置仅描述记录中的本机环境，不代替其他机器重新运行预检，也不是历史实验锁文件。

包内已有 `code/data/qevasion_raw/` 和 `code/data/qnli_raw/`，复核保存结果应优先使用这些快照。若本地数据缺失，可在 `code` 目录尝试重新获取 QEvasion；此命令取得的是上游当前版本，不保证与本次快照相同：

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
python -c "from datasets import load_dataset; load_dataset('ailsntua/QEvasion').save_to_disk('data/qevasion_raw')"
```

`05_hierarchy.py`、`06_direct_transfer.py` 和 `09_review_validation.py` 必须能读取本地 QEvasion。若采用重新下载的数据，应先核对样本、顺序和标签，再在独立结果目录生成对应的特征与预测，避免把不同数据版本的新旧结果混用。

## 运行顺序

先执行编号 10 的环境预检；通过后在 `code` 目录按编号顺序执行 01—09：

```powershell
python 01_eda.py
python 02_baselines.py
python 03_plm_lr.py
python 04_transfer_qnli.py
python 05_hierarchy.py
python 06_direct_transfer.py
python 07_hierarchy_sensitivity.py
python 08_bootstrap_uncertainty.py
python 09_review_validation.py
```

`03_plm_lr.py` 在 CPU 上加载 MiniLM 并重新编码，不会直接跳过编码而复用旧缓存。`04_transfer_qnli.py` 仅在本地 QNLI TSV 缺失时尝试下载；若下载地址不可用，请按原始来源取得 `train.tsv`、`dev.tsv` 并核验快照。`05_hierarchy.py` 依赖 `03_plm_lr.py` 的向量缓存和预测；`06_direct_transfer.py` 依赖 `04_transfer_qnli.py` 的预测 CSV。01—08 的默认运行会重新写入各自输出，保留提交快照时宜在项目副本运行；09 将新增分析写入独立的 `outputs/review/`。

`07_hierarchy_sensitivity.py` 复用冻结句向量，比较九分类模型的类别权重与粗类决策规则；其中“按子类数平均”只是启发式得分，不是粗类概率。`08_bootstrap_uncertainty.py` 对相同测试样本配对重抽样，并按相同回答文本聚类重抽样。两项均为事后诊断，不使用官方测试集选择模型或调参。

`09_review_validation.py` 按原始来源 URL 补充审计：以同一来源的全部问题为簇进行配对 bootstrap，并在训练集内按 URL 做固定种子分组留出。留出训练/验证为 2,785/663 条、229/58 个 URL，URL 无重叠；直接三分类与三种细类汇总设置均固定 C=0.1。该分析用于后验敏感性复核，不依据官方测试集重新选择方案。它也复算已保存预测的逐类指标、混淆矩阵、原始划分及数据统计。

## 结果定位与说明

主要结果记录在 `code/outputs/metrics_*.json` 与 `code/outputs/hierarchy/metrics_*.json`，逐样本预测在同目录的 CSV 中。报告正文中的数字统一保留四位小数。当前保存的一次运行中，QEvasion 上的 TF-IDF + LR Macro-F1 为 0.4351，MiniLM + LR 为 0.4855；九类细标签概率汇总后的 Macro-F1 为 0.3504。QNLI 上目标域训练的 TF-IDF + LR Macro-F1 为 0.5741，仅源域训练并固定投影标签的结果为 0.3364。

这些是一次固定随机种子运行的结果，不是多次实验的均值。QEvasion 官方测试集缺少细粒度 `evasion_label`，因此报告没有声称测得测试集的九分类 F1。细粒度标签 `Implicit` 映射到粗类 `Ambivalent`，而非 `Clear Reply`。

敏感性分析结果保存在 `code/outputs/hierarchy/metrics_sensitivity.json`；样本/回答文本簇的区间保存在 `code/outputs/metrics_uncertainty.json`；按来源 URL 的审计、分组留出、来源子组与区间保存在 `code/outputs/review/review_validation.json`。09 使用种子 42、NumPy PCG64 和 10,000 次配对来源簇重抽样：MiniLM−TF-IDF 的 Macro-F1 差值为 +0.0504，95% 区间为 [−0.0222, 0.1498]；原层级方案−MiniLM 为 −0.1351，区间为 [−0.2268, −0.0637]。这些区间不涵盖训练种子波动，也不会消除训练/测试来源重合，不能据此宣称不同场景下的稳定优势。

## 完整复现记录

2026-10-18 将代码和原始数据复制到隔离目录，以六个新 Python 进程依次运行 01—06；03 完整重新编码，六步均成功退出。当前机器的进程总耗时依次为 3.84、17.02、44.04、9.79、5.15、7.19 秒，六步骤经过时间约 88 秒，不含依赖安装及前置环境检查。耗时仅描述本次机器上的单次运行。

所有测试硬预测、最终指标和混淆矩阵与保存结果一致；新生成的 MiniLM 特征缓存与原缓存逐元素及 SHA-256 完全一致。02、04、06 的指标和预测文件也逐字节一致。仍有明确的小差异：03 未选中的 C=10 验证 Macro-F1 从 0.546842 变为 0.540682，最终选择仍为 C=0.1；MiniLM 与层级预测文件的概率列最大绝对差约为 1.4×10⁻⁵、2.02×10⁻⁶。因此，复现结论是测试硬预测和主指标一致，不是所有输出文件完全相同。

完整证据见 `code/outputs/review/replay_evidence/`：`environment.json` 记录实际包版本与导入位置，`run_records.json` 记录各进程及耗时，`01_eda.log`—`06_direct_transfer.log` 保留真实运行输出，`output_comparison.json` 给出逐文件差异，`replay_summary.json` 提供汇总。另有 `environment_with_vendored_deps.json` 记录原 `deps/` 与新增 `model_deps/` 并存时的导入及离线模型加载预检。

## 数据和第三方许可

原始数据快照、依赖副本与其来源说明予以保留。数据和模型的使用范围以各发布方页面的许可与条款为准；第三方包随附的 LICENSE、NOTICE 等文件保留在相应依赖目录。本仓库未对这些第三方资源另行授予许可。

