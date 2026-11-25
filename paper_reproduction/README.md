# QEvasion 作者的公开计分复跑

原论文 https://aclanthology.org/2024.findings-emnlp.300/ 。作者源码使用固定提交 `ff78482fe3ee63b76ce427ea4b8ee2b690fe8896`。

```powershell
$upstream = Join-Path $env:TEMP 'question-evasion-author'
git clone https://github.com/konstantinosftw/Question-Evasion $upstream
& $py paper_reproduction/reproduce_thomas.py --repo $upstream --output paper_reproduction/author_reproduced.json
```

公开输入每位标注员有 31 个注入项目，匹配并计分后每位为 36 个子问题。诱导标签选择计数为 4、3、3。发布材料、输入校验值与运行结果见 `author_artifacts.json`。
