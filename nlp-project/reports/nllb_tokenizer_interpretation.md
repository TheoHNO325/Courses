# NLLB tokenizer audit interpretation

审计对象：`facebook/nllb-200-distilled-600M` 的本地 tokenizer（`NllbTokenizerFast`）。
模型权重未下载；统计只使用 tokenizer。

## 结果

- FLORES 藏文源句最大长度为 131 tokens，MITRA 验证集最大长度为 119 tokens。
- 所有样本均低于 NLLB 训练阶段采用的 512-token 参考上限，因此第一版不需要长句切分。
- 藏文端 unknown token 数量很低，主要是少量罕见的藏文组合/附加符号。
- 中文端的 unknown token 主要来自罕见佛典汉字、梵语音译字和部分中文标点，而不是藏文编码错误。
- MITRA 验证集共有 11,073 个中文 unknown token，约占中文 token 总数的 2.45%；出现于 6,791/23,219 条样本，包含约 842 种不同的未知片段。高频例子包括 `惱`、`訶`、`囉`、`嚩`、`拏`、`唵`、`吽`、`蘊`、`羯` 等。
- 将目标语言代码从 `zho_Hant` 改为 `zho_Hans` 不会改变 tokenizer 的词表或 unknown 统计；语言代码不能自动完成繁简转换。

## 当前决策

1. 暂不训练新的藏文 tokenizer，也不做藏文语言学分词；继续使用 NLLB 原生 tokenizer。
2. 暂不删除或替换 MITRA 中的罕见佛典汉字，因为它们可能正是领域风格的重要组成部分。
3. 在微调前增加一个小型对照：原始繁体目标 vs. 繁简/异体字规范化目标，比较 unknown 比例和翻译风格损失。
4. 只有当保留传统佛典字形导致生成结果大量出现 `<unk>` 时，才考虑扩充词表；这会涉及新增 token、初始化嵌入和重新微调，不作为第一版方案。

完整机器统计见 `reports/nllb_tokenizer_audit.json`。
