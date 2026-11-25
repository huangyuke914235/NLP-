# QEvasion 原论文公开材料的部件复现

主论文为 Thomas、Filandrianos、Lymperaiou、Zerva 和 Stamou 的 “I Never Said That”，发表于 Findings of EMNLP 2024，5204–5233 页。论文先拆分政治访谈的多重提问，经人工校验后分别判断完整回答有无提供各子问题所需的信息。它比较直接预测三类清晰度与预测九类回避策略再上卷两种路径，使用编码器微调、生成模型提示与 LoRA 调优。本目录重跑作者公开的反事实摘要标注检验，并审计其模型评估脚本的输入。课程项目的模型训练、问句改写实验和预测另由主流程生成。[论文原文](https://aclanthology.org/2024.findings-emnlp.300/) · [作者版本](https://github.com/konstantinosftw/Question-Evasion/tree/ff78482fe3ee63b76ce427ea4b8ee2b690fe8896)

## 运行

需要 Git 和 Python 3.10 或更高版本。以下命令从 `coursework_nlp` 目录在 PowerShell 运行。首次执行将作者库克隆到临时目录；脚本始终从指定 Git 对象读取数据，不依赖当前检出的分支或工作树换行。

```powershell
$paperRepo = Join-Path ([System.IO.Path]::GetTempPath()) 'szu-qevasion-author'
git clone https://github.com/konstantinosftw/Question-Evasion $paperRepo
python .\paper_reproduction\reproduce_thomas.py --repo $paperRepo --output .\paper_reproduction\author_artifacts.json
```

复现使用作者的 `find_start_position`、`extract_original_label`、`extract_counterfactual_label` 和 `find_disagreement` 原函数。运行器从哈希校验通过的固定源文件加载这四个纯函数，把仅用于计算精确匹配比例的 `accuracy_score` 替换为 Python 逐条计数，统计分子与分母。原 `utils.py` 导入了与这四个函数无关的模糊匹配库和句子编码器，隔离函数可避免额外依赖与模型下载。复现器以受限反序列化读取作者的预测索引，保留逐文件 SHA-256 与缺项检查。

## 实测与范围

| 标注员 | 作者 JSON 记录 | 作者代码匹配并评价的子问题 | 选择诱导标签 | 选择率 | 作者命令行输出的不同意率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 31 | 36 | 4 | 11.11% | 88.89% |
| 2 | 31 | 36 | 3 | 8.33% | 91.67% |
| 3 | 31 | 36 | 3 | 8.33% | 91.67% |

作者附录 A.3 解释，其反事实摘要会在标注员看到的分解文本中注入与原访谈不符的信息，以检查标注员是否仅凭摘要判类。论文报告选择诱导标签的比例均不高于 0.08。当前作者库的三个 JSON 各含 31 项，原函数以匹配成功的子问题计分，复跑分母为 36，所得选择率见表。公开材料无法确认论文数字对应的进一步筛选或计分口径，本次保留差异和输入校验值。两次隔离 Python 进程输出的 `author_artifacts.json` 逐字节相同，SHA-256 为 `44676d6420dcc6fbff8110fccdde9939e7360e0224321bcf164b6249df591d`。该哈希包括环境版本字段，其他环境可按 JSON 比较计数和输入哈希。[附录 A.3](https://aclanthology.org/2024.findings-emnlp.300.pdf) · [作者计分代码](https://github.com/konstantinosftw/Question-Evasion/blob/ff78482fe3ee63b76ce427ea4b8ee2b690fe8896/scripts/counterfactual_summaries_analysis.py)

模型评估方面，论文第 4–5 节说明其评估使用 317 个共同标注问答，多数投票后由专家裁决无多数的个案。发布库中当前原始 CSV 也含 317 行，但 `results.py` 的映射没有覆盖 11 次 `2.9 Diffusion` 和 1 次 `2.5 Contradictory` 原标注。15 个已发布预测文件中 3 个各缺 1–2 个原始测试行索引，代码在缺项时未重置当前预测，还重复列出一个预测文件。复现工作核查了这些输入与路径，因此论文表 3 的训练与推理结果只在引用处按论文口径陈述。课程项目使用的 Hugging Face 发布版官方测试集为 308 条，另建采访来源分组评估，随项目模型、输入表示和切分也均不同。[作者模型结果脚本](https://github.com/konstantinosftw/Question-Evasion/blob/ff78482fe3ee63b76ce427ea4b8ee2b690fe8896/scripts/results.py) · [Hugging Face 数据卡](https://huggingface.co/datasets/ailsntua/QEvasion)

本项目固定回答的反事实诊断面向模型行为。等义改写保留问题的信息需求，依 CheckList 的不变性思想评价。经语义核对的不相关替换借鉴 CAT 的单侧替换思路，同时依任务含义考虑 `Ambivalent` 与明确拒答的边界。CAT 原作者以自动替换和任务默认关系标签评测多个公开基准，本项目使用人工逐条核对的小型题集以控制语义。查文时，CAT 论文给出的代码地址返回 404，课程诊断据论文描述独立实现。[Ribeiro 等，ACL 2020](https://aclanthology.org/2020.acl-main.442/) · [Elazar 等，Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.205/)
