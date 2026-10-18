# 数据来源与本地快照核验

原始数据的本地副本位于 `code/data/`。下表记录生成实验结果时使用的数据文件，核验日期为 2026-09-25。文件时间为本地快照的记录时间，不代表上游发布日期。

| 数据 | 来源 | 使用量 | 本地文件记录时间（UTC） |
| --- | --- | --- | --- |
| QEvasion | [发布者数据集](https://huggingface.co/datasets/ailsntua/QEvasion) | train 3,448；test 308 | 2026-09-04 14:45:23 |
| QNLI | [GLUE QNLI](https://huggingface.co/datasets/nyu-mll/glue/tree/main/qnli)；脚本使用 GLUE TSV 下载地址 | 训练集固定种子抽样 12,000；dev 全量 5,463 | 2026-09-11 05:12:27 |

本地原始文件的 SHA-256：

| 本地相对路径 | SHA-256 |
| --- | --- |
| `code/data/qevasion_raw/train/data-00000-of-00001.arrow` | `636D1CA556F8E614E219767ECABA07B0202F74CB43733F271B01AE96EBBBA84E` |
| `code/data/qevasion_raw/test/data-00000-of-00001.arrow` | `6D5765D1F7B75A7882D88BD493943EFD797E05184A7516D5B27E43049842C949` |
| `code/data/qnli_raw/train.tsv` | `3BB7026735A0D4CD439DC1CA6EA162E0C1CC8F962F52DF9CC50EE8F8278E0C0E` |
| `code/data/qnli_raw/dev.tsv` | `AC19409DCB04631EAAA094BDEBDCF621086BC352503DFB6A1A648B614986F4FD` |

由 `datasets.save_to_disk` 生成的 Arrow 文件可能因库版本或序列化方式而出现不同的文件哈希；这并不单独证明样本内容不同。请结合样本量、字段、标签分布以及 `code/outputs/eda_stats.json` 核查。若另用上游新快照，应在独立项目副本重新运行流程，不与旧快照的缓存、预测和指标混用。

## 来源单位复核（2026-10-18）

本次复核未改变上表的原始数据快照。QEvasion 中的 `url` 字段在训练集与测试集均完整，能够作为可核查的来源分组单位；测试集 `title`、`president` 的缺失不等于来源不可辨识。

| 来源统计 | 数量 |
| --- | ---: |
| 训练集唯一来源 URL | 287 |
| 训练集唯一标题文本 | 175 |
| 测试集唯一来源 URL | 56 |
| 训练/测试共享来源 URL | 39 |
| 测试样本中来源 URL 已见于训练集 | 118 / 308（38.31%） |
| 测试样本中来源 URL 未见于训练集 | 190 / 308 |

175 是不同标题字符串的数量，不应写为 175 次访谈。训练集和测试集存在来源重合，官方测试结果不是完全来源独立的泛化评价。补充审计按 URL 构造训练集内分组留出及测试集来源簇重抽样，结果见 `code/outputs/review/review_validation.json`；不据此重新选择官方测试方案。

## 第三方资源许可

数据和模型的使用范围以各发布方页面的许可与条款为准；原始数据引用及第三方依赖随附的 LICENSE、NOTICE 等文件予以保留。本项目提供的本地副本与快照校验不改变这些资源的权利归属，也不另行授予第三方资源许可。
