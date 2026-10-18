# 实际日志页面截图说明

这两张图片是本地只读 HTML 证据查看页的真实浏览器截图，不是终端截图，也没有模拟终端界面。

| 文件 | 尺寸 | 展示内容 |
|---|---|---|
| reproduction_log.png | 1000 × 620 | 独立进程运行 01—06 的实际耗时和退出码；从02、03、04真实日志读取的关键指标（四位小数）；缓存逐元素一致和预测差异为0；复现差异说明 |
| reproduction_full.png | 1600 × 800 | 先前完整日志查看页截图，保留02、03、04原始输出逐字节选 |
| data_validation.png | 1600 × 940 | 01 的数据读取原始日志、实际配置标签映射、09 的来源审计和检查结果 |

实际截图浏览器为 Microsoft Edge 155.0.4283.62，使用 `--headless=new`，由 PowerShell 以 `-WindowStyle Hidden` 启动；最终截图均退出 0。渲染输入是同目录 `focus_replay.html`、`reproduction_log.html` 和 `data_validation.html`。截图未经图像文字重绘或伪终端加工。页面生成脚本为 `../../build_evidence_pages.py`。

为在报告5.3英寸图宽下清晰打印，`reproduction_log.png` 改用1000像素宽关键输出页，表格字号24像素，打印等效约9.2磅；完整1600像素截图另存 `reproduction_full.png`。关键页明确标注“实际日志摘要、指标保留四位小数”，未把四舍五入后的摘要当作逐字原始输出。

Playwright CLI 未缓存在 npm 中；尝试获取时 `registry.npmjs.org` 域名解析失败（ENOTFOUND）。本机缓存 Chromium 启动退出 21，未生成截图，因此按允许的后备路径使用系统 Edge。对应错误记录保留于同目录日志。

完整运行命令和时间来自 `../../replay_evidence/run_records.json`，页面中可展开逐字命令数组；模型输出节选来自相应 `.log`。原始数据标签名与展示名不同的事实已直接展示。`evidence_page_manifest.json` 保存了页面读取的证据文件 SHA-256。

关键披露：独立复现环境中 MiniLM 未入选的 C=10 验证点由 0.546842 变为 0.540682；最终选中的 C=0.1、官方测试分类指标和所有五份预测文件的样本身份及类别预测一致。截图和报告均不应写成“所有浮点输出逐字相同”。
