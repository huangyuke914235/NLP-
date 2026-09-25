# 数据来源与本地快照核验

提交包内附有原始数据的本地副本，位于 `code/data/`。以下信息记录了生成仓库内结果时使用的本地数据文件；核验日期为 2026-09-25。文件修改时间仅是本地记录时间，不能当作上游发布日期。原始数据的上游提交修订号未保存在旧实验记录中，因此不在这里臆测。公开上传这些数据前，请核对数据集发布页的许可和课程要求。

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

由 `datasets.save_to_disk` 生成的 Arrow 文件可能因库版本或序列化方式而出现不同的文件哈希；这并不单独证明样本内容不同。请结合样本量、字段、标签分布以及 `code/outputs/eda_stats.json` 核查。若远端数据顺序或内容改变，先重跑 `03_plm_lr.py` 再运行依赖其缓存的 `05_hierarchy.py`。
