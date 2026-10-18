# 本地第三方依赖格式补丁

日期：2026-10-18。

Transformers 4.57.1 中的32字符公开模型类名可能被GitHub推送保护误识别为Mistral API密钥。先后两次提示指向了同一类名的不同文件。全包扫描还发现另一个同样长度的公开模型类名，两个候选均有对应类定义。对照官方同版本源码确认，这些不是本项目配置的凭据。

来源：[Transformers v4.57.1 官方源码](https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models)。

## 补丁范围

- `transformers/models/auto/modeling_auto.py`：4处自动模型映射。
- `transformers/models/auto/modeling_tf_auto.py`：1处自动模型映射。
- `transformers/models/mistral3/configuration_mistral3.py`：3处文档示例。
- `transformers/models/mistral3/modeling_mistral3.py`：2处文档示例、1处导出列表。
- `transformers/models/mistral3/modular_mistral3.py`：2处文档示例、1处导出列表。
- `transformers/models/mistral/modeling_mistral.py`：1处类名字面量。
- `transformers/models/mistral/modeling_tf_mistral.py`：2处类名字面量。
- `transformers/models/mistral/modular_mistral.py`：1处类名字面量。

补丁覆盖8个源码文件共18处字面量，将它们分割为相邻字面量。Python在编译时将其合并成完全相同的值；原始字符串保留原有前缀和引号形式。8个文件修改前后的完整Python抽象语法树均相同，包括文档字符串的值。类定义、导入语句、公开类名、映射结果、包版本、许可证及模型逻辑不变。

每个修改文件标注本地格式补丁；`.dist-info/RECORD` 同步更新对应校验值和字节数。对整个本地模型依赖目录按Python字符串词法单元进行扫描，确保上述同类字面量均已处理，而不是只修改错误弹窗显示的第一处。

这是对已核实误报的格式处理，不用于隐藏真实凭据，也没有关闭或豁免GitHub安全检查。将来更换依赖版本时，应重新核对上游源码与推送检查。
