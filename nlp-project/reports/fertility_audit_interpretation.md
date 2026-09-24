# 统一 fertility 实验结果

## 1. 实验范围

- MITRA 训练集使用固定 seed=336 的流式抽样 10,000 条；
- 使用 MITRA validation 全部 23,219 条；
- 使用 FLORES dev 997 条和 devtest 1,012 条；
- NLLB 和 HY-MT2 统计时不计入特殊 token，NLLB 目标语言按数据集使用 `zho_Hans`（FLORES）或 `zho_Hant`（MITRA）；
- CS336 使用 assignment1 的 BPE 实现，分别训练 Raw Unicode 和 Syllable-aware 两种版本；
- fertility 同时按 Unicode code point 和近似藏文音节统计。音节仅按 `་`/`་` 边界近似，不等价于词语分词。

完整机器结果见 `reports/fertility_audit.json`。CS336 词表文件见 `outputs/cs336-bpe-fertility/`。

## 2. 藏文源文本的 tokens / Tibetan syllable

| 数据集 | NLLB | HY-MT2 | CS336 Raw | Tsheg-boundary | Mark-aware |
|---|---:|---:|---:|---:|---:|
| FLORES dev | 1.196 | 4.264 | 3.099 | 3.120 | 2.239 |
| FLORES devtest | 1.203 | 4.251 | 3.095 | 3.116 | 2.242 |
| MITRA validation | 1.200 | 4.065 | 2.825 | 2.840 | 2.073 |
| MITRA train sample | 1.175 | 4.068 | 2.844 | 2.862 | 2.057 |

初步结论：

1. **NLLB 的藏文编码最紧凑**，约 1.17–1.20 token/音节。
2. **HY-MT2 的 token 序列显著更长**，约 4.06–4.26 token/音节；这会增加推理上下文和微调显存开销，但它没有出现 unknown token。
3. 当前 Tsheg-boundary 版本约 2.8–3.1 token/音节，明显长于 NLLB；插入空格本身会带来边界开销。
4. Mark-aware 版本把 `\p{L}` 改为 `[\p{L}\p{M}]`，使基字与组合记号进入同一预分词块，fertility 降至约 2.06–2.24，说明预分词边界确实会影响藏文编码效率。
5. Mark-aware 的改善仍不是“翻译质量改善”的证明，只说明输入 token 统计更紧凑；需要后续用 CS336 Transformer 训练或下游任务验证。
6. MITRA 文本的 fertility 略低于 FLORES，说明当前样本中二者的字符/术语分布不同，后续应继续按领域报告，而不能只汇报总平均。

## 3. 序列长度和 unknown

在当前测试范围内：

- NLLB 藏文最大长度：FLORES dev 116、devtest 129、MITRA validation 117、训练抽样 120；
- HY-MT2 藏文最大长度：FLORES dev 382、devtest 459、MITRA validation 253、训练抽样 269；
- 所有当前样本都没有超过 512 token；
- HY-MT2 tokenizer 没有 unknown token；
- NLLB 藏文端 unknown 很少，主要集中在少量组合/附加符号；
- NLLB 中文端 unknown 比例约为：FLORES dev 1.74%、devtest 2.00%、MITRA validation 2.73%、训练抽样 2.89%。MITRA 的佛典中文长尾更明显。

这里的 unknown 比例严格对应本次“去除特殊 token、MITRA 使用 `zho_Hant`”的统计协议，不能和旧报告中使用不同设置的数字直接比较。

## 4. NFC/NFD 观察

- FLORES dev 中有 5 条样本在 NFC 下发生变化、6 条在 NFD 下发生变化；
- FLORES devtest 中各有 5 条发生变化；
- MITRA validation 和 10k MITRA 抽样中没有发现变化；
- 当前没有观察到 normalization 导致 NLLB 长度上限问题，但 FLORES 中的少量变化仍需在数据预处理时留痕，不能覆盖原始文本。

## 5. CS336 词表限制

虽然命令请求了 8k 和 16k 词表，但在当前 10k 条藏文源文本和 assignment1 的预分词规则下，pair 表提前耗尽：

- Raw BPE 实际词表约 1,283；
- Tsheg-boundary BPE 实际词表约 1,397；
- Mark-aware BPE 实际词表约 3,805；
- 因此 8k/16k 两个请求在本次源文本实验中得到相同的有效 merge 规模。

这不是训练失败，而是一个需要记录的实验限制：藏文源文本的可合并 byte pair 类型有限。Mark-aware 能达到更大的有效词表，正是因为它没有在基字和组合记号之间提前切断。后续若要研究真正的 4k/8k/16k 词表缩放，应使用更大的藏文训练样本或源文+中文目标的共享语料。

## 6. 对后续项目的决策

1. 第一版 NLLB 继续使用原生 tokenizer，不人为插入音节空格。
2. NLLB 的 256 token 初始长度设置是安全的；若微调时使用 512 上限，当前数据不会造成大量截断。
3. HY-MT2 适合做本地参考模型，但必须预留更长的序列和更高显存，不能直接按 NLLB 的 batch 配置运行。
4. CS336 的价值主要是解释 tokenizer 和预分词影响；当前 Mark-aware 结果支持继续研究组合记号边界，但不应期待它在当前语料规模上超过 NLLB。
5. 下一步先冻结这套统计口径，再做 10–50 条 NLLB CPU 推理冒烟测试；不要因为 fertility 结果就立即修改预训练词表。
