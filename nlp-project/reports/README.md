# Reports

保存可复核的统计摘要、图表、人工检查表和实验结论。大型逐行输出不要直接保存在 Jupyter 单元格中。

当前主要入口：

- `tokenizer_experiments.html`：NLLB/HY-MT2/CS336 分词与 fertility 的离线可视化汇总；
- `fertility_audit.json`、`fertility_audit_interpretation.md`：统一 fertility 原始结果与解释；
- `nllb_tokenizer_audit.json`、`nllb_tokenizer_interpretation.md`：NLLB tokenizer 专项审计；
- `nllb_cpu_smoke.json`：FLORES dev 10 条 NLLB CPU 推理输出，仅作链路验证；
- `mitra_v2_filtering.json`、`mitra_v2_assessment.md`：MITRA 过滤与数据质量说明；
- `dataset_download_record.json`、`dataset_download_manifest_update.json`：2026-09-24 新数据下载记录；
- `environment_setup.md`：CPU 环境、依赖版本与评测工具说明。

项目整体状态和下一步以 `docs/PROJECT_HANDOFF.md` 为准。
