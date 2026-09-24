# 藏文—中文翻译项目：计划与进度交接文档

> 面向后续 Codex/Agent session 的项目状态说明。更新日期：2026-09-24。

## 1. 项目目标

课程项目主题为“少数民族语言翻译——以藏语与东巴文为例”。当前主线是藏文到中文翻译，计划按以下顺序推进：

1. 数据集核验、质量分析和训练/验证/测试划分；
2. 基于开源多语翻译模型的藏中微调；
3. 藏文 OCR 与 OCR 噪声下的翻译；
4. 语言类型识别、图片输入和中文输出平台；
5. 有余力时加入东巴文象形字识别和图像—中文辅助翻译。

当前只进行 CPU 侧准备，不进行正式 GPU 训练。

## 2. 资源和工作约束

- 项目目录：`D:\Courses\nlp-project`
- CPU：12 核
- 内存：约 15 GB，总空闲约 2 GB
- 磁盘：D 盘约 1 TB 可用
- 未来可能使用最多 4 张 4090/5090，但目前没有 GPU 资源
- 不依赖外部翻译 API；模型尽量本地运行
- 原始数据只读，清洗和划分结果写入新文件
- 内存紧张时必须使用流式读取或小样本；不要一次性加载完整 MITRA

## 3. 已确定的模型和 tokenizer 方案

### 主模型

暂定主模型为：

```text
facebook/nllb-200-distilled-600M
```

理由：它是翻译专用的多语模型，支持低资源语言，并包含藏文语言代码 `bod_Tibt`。中文目标可使用：

- `zho_Hans`：简体中文
- `zho_Hant`：繁体中文

模型本体约 600M 参数，Hub 文件约 2.5 GB，未来有单张 24 GB GPU 时可进行小规模全参数或参数高效微调。模型许可为 CC-BY-NC-4.0，只适合本课程研究/非商业场景，不能默认用于商业部署。

### tokenizer

使用 NLLB 自带的 SentencePiece tokenizer，不另行训练 tokenizer：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M"
)
tokenizer.src_lang = "bod_Tibt"
tokenizer.tgt_lang = "zho_Hans"  # 或 zho_Hant
```

输入使用藏文 Unicode，不使用 Wylie。MITRA 的 Wylie 已在处理脚本中转换为藏文 Unicode。

### 关于藏文分词

第一版不做传统语言学意义上的藏文分词，也不人为插入音节空格。保留藏文 `་`、`།` 等符号，让 NLLB SentencePiece 完成子词切分。之后可将“原始 Unicode”和“按音节预切分”作为消融实验，但不是当前主流程的前置条件。

### 不作为主模型的候选

- mBART-50 官方语言列表不包含藏文；
- mT5-small 是通用 text-to-text 预训练模型，官方语言列表也没有藏文；
- ByT5-small 可作为后期 OCR 噪声鲁棒性消融，但字节序列更长、速度更慢，且不是翻译专用模型。

## 4. 已完成的数据工作

### FLORES-200

- 本地目录：`data/raw/flores200_20260529/`
- 藏文：`bod_Tibt`
- 中文：`zho_Hans`
- 已下载并完成本地完整检查，共 2009 条对齐句对（dev/devtest）
- 仅作固定评测集，不进入训练或超参数选择
- 下载记录：`reports/flores200_download_record.md`
- 检查报告：`reports/flores200_local_full_check.json`
- 人工检查样本：`reports/flores200_manual_check_20.csv`

### MITRA v2 官方评测集

- 原始目录：`data/raw/mitra_v2_eval/`
- `bo2zh.tsv`：2000 条，传统佛教/佛典中文
- `lotsawahouse.tsv`：1660 条，现代宗教、文学和讲解性文本
- 两者均作为固定评测或质量观察，不直接混入训练
- 说明：`reports/mitra_v2_assessment.md`

### MITRA v2 全量训练匹配

- 原始文件：`data/raw/mitra_v2/bo-zh_matches.ndjson.gz`
- 固定 commit：`bf7b340cd4f75bda3089479a6367ad94a8cbee10`
- SHA-256：`9179AD2F491A77FCEB06A52FDD93E1D2AECE3AD8623EF0BAA18A08E6373DF897`
- 记录数：836,559
- 原始藏文源文本是 Wylie，不是 Unicode；已使用 `pyewts` 转换

### MITRA 质量样本

- 样本：200 条，按内部 score 分位数和源文本长度分层
- JSONL：`reports/mitra_v2_quality_sample_200.jsonl`
- 统计：`reports/mitra_v2_full_scan.json`
- 可选人工检查工作簿：`outputs/mitra-v2-quality-review/mitra_v2_quality_review_200.xlsx`

用户不懂藏文，因此已决定不把人工语义忠实度标注作为当前阻塞步骤。工作簿保留给未来藏语合作者使用；当前先做结构性过滤、自动统计和固定评测。

### 2026-09-24 新增原始资源

以下资源已经下载并登记到 `data/manifests/datasets.csv`，但“下载完成”不等于“已经清洗或可以直接训练”：

| 数据集 | 本地状态 | 当前可用性判断 |
|---|---|---|
| OpenPecha C0A2DD042 | 4,618 个文件，约 84 MB；含 1,893 个藏文文件、78 个中文文件 | 需先确定藏中成对文档及切句方式；不能直接按清单中的 2,573 当作藏中句对 |
| Modern Tibetan Corpus | 8,924 个文件，约 82 MB；含 12 个 CoNLL-U 文件 | 可做现代藏文风格、词汇和分词分析；不是藏中平行训练集 |
| Shajiu PublicCorpus samples | 12 个文件，约 24 MB；登记约 100 条样例 | 仅限质量观察；许可证是自定义研究条款，不进入正式训练集 |
| CUTE Tibetan subset | 2 个藏文文件，共约 24.61 GB | 只下载了 `bo.txt` 与 `n-bo.txt`，没有中文对应侧；且属于机器翻译/合成来源，目前不能用于藏中平行微调或黄金评测 |
| DongbaMIE | 10 个文件，约 1.92 GB；句子级 23,530，段落级 2,539 | 官方切分和 JSON 标注齐全，图像位于两个 ZIP 中；可进入东巴图像语义理解的数据结构核验，原始数据不得重新分发 |
| Dongba1800 | 1,800 张图像及 1,800 个标注；train/test 为 1,440/360 | 可作为东巴字符/区域检测候选；许可证文字有冲突，未解决前不作为主实验依赖 |
| BDRC Tibetan OCR benchmark | 472 条，约 245 MB | 可作为固定藏文 OCR 评测集，不进入 OCR 训练 |

下载与核验依据：`reports/dataset_download_record.json`、`reports/dataset_download_manifest_update.json`、`outputs/download_spotcheck.txt` 和 `outputs/manifest_validation.txt`。这些是自动结构核验，不代表人工语义质量已经确认。

## 5. 已完成的 MITRA 结构性过滤和划分

配置：`configs/mitra_v2_filter.json`

当前过滤规则包括：

- 只保留一对一句对；
- 源 Wylie 长度 6–285；
- 中文长度 3–86；
- 中文/源文本字符长度比 0.03–3.81；
- 必须包含 CJK；
- 排除 ASCII 数字；
- 排除 EWTS 转换警告；
- 转换后必须确实包含藏文字符；
- 按文件名/文档划分验证集，避免同文档泄漏。

输出目录：`data/processed/mitra_v2_conservative/`

| split | 记录数 | 文档数 | SHA-256 |
|---|---:|---:|---|
| train | 427,916 | 638 | `D353875914F7239E7431AB380454E8F2C96C2C9A986A133194A8CC233B7E7FDC` |
| validation | 23,219 | 38 | `014422A7206DA546CF33750EBA43840EF676EEF894AB9038C69F7CBA857E4D76` |

过滤统计：

- 原始记录：836,559
- 接受记录：451,135（53.927%）
- 训练/验证文档无重叠
- 目标句并非总是一对一：原始数据只有约 56.4% 的目标为单段
- MITRA 内部 score 只是排序信号，不能当作概率或严格相似度

详细报告：`reports/mitra_v2_filtering.json`。处理脚本：`src/prepare_mitra.py`。

## 6. 当前实验状态与下一步

### 已完成：NLLB tokenizer CPU 审计

在下载模型权重前先独立完成了 tokenizer 统计；权重现已下载，但以下审计仍是纯 tokenizer 实验：

1. 藏文源句 token 长度分布；
2. 中文目标 token 长度分布；
3. 512 token 上限下的截断比例；
4. `unk` 或异常字符比例；
5. 藏文标点、tsheg 和组合字符是否被稳定编码；
6. `zho_Hans` 与 `zho_Hant` 的目标编码差异；
7. FLORES、MITRA 佛典、MITRA 现代文本之间的长度差异。

结果文件：

- `reports/nllb_tokenizer_audit.json`
- `reports/nllb_tokenizer_interpretation.md`
- `outputs/nllb-tokenizer/`

关键结果：FLORES 和 MITRA validation 均未出现超过 512 token 的样本；藏文端 unknown 很少；MITRA 中文端约 2.45% 的 token 属于未知 token，主要是罕见佛典汉字、梵语音译字和标点；`zho_Hans` 与 `zho_Hant` 不会改变 tokenizer 词表，也不会自动完成繁简转换。

### 环境状态

`requirements-cpu.txt` 已加入：

```text
transformers==4.46.3
sentencepiece==0.2.0
torch==2.4.1+cpu
```

原因是当前项目虚拟环境为 Python 3.9，不能直接使用 Transformers 5.x。当前 `.venv` 已安装 CPU 版 PyTorch，可运行本地 NLLB 推理。评测和报表依赖 `sacrebleu==2.6.0`、`pandas`、`openpyxl`、`matplotlib`、`tabulate` 也已安装并固定在 `requirements-cpu.txt`；环境核验见 `reports/environment_setup.md`。

NLLB 权重已完整下载到 `outputs/nllb-200-distilled-600M/pytorch_model.bin`：文件大小 2,460,457,927 字节，SHA-256 为 `C266C2CFD19758B6D09C1FC31ECDF1E485509035F6B51DFE84F1ADA83EEFCC42`，与 Hub 元数据一致。

CPU 冒烟测试脚本 `src/smoke_nllb_cpu.py` 已在 FLORES dev 的前 10 条样本上成功运行，使用 `bod_Tibt -> zho_Hans`，结果为：

- 模型加载：3.362 秒；
- 生成：253.912 秒；
- 平均：25.391 秒/条（4 个 CPU 线程）；
- 无空输出，但第 3 条出现严重的“号号号……”重复退化；
- 对这 10 条临时计算 chrF++ 为 11.60、中文分词 BLEU 为 13.60。

这组指标只用于诊断链路和发现解码问题，样本数过小且不是完整测试集，不能作为正式 zero-shot 分数。完整输出见 `reports/nllb_cpu_smoke.json`。复现命令：

```powershell
& .venv\Scripts\python.exe src\smoke_nllb_cpu.py --split dev --limit 10 --threads 4 --num-beams 2
```

运行模型前仍应尽量释放至少约 5 GB 可用内存。`download_probe_1mb_v2.bin` 是早期连通性探针，不是模型组成部分。

### 已完成：Tokenizer HTML 汇总

`reports/tokenizer_experiments.html` 汇总了 NLLB、HY-MT2、CS336 Raw/Tsheg-boundary/Mark-aware 的 fertility、长度、截断、unknown、Unicode 规范化和词表耗竭结果。页面约 106 KB，数据完全内嵌，可离线打开。`outputs/tokenizer_report_validation.txt` 已确认 payload JSON 和内联 JavaScript 语法通过；尚未把页面本身当作新的实验结果，数值来源仍以两个 JSON 报告为准。

### 当前下一步事项

1. **已完成统一 fertility 统计**：10k MITRA train sample、全部 validation 和 FLORES dev/devtest 均已比较 NLLB、HY-MT2、CS336 Raw/Tsheg-boundary/Mark-aware；结果见 `reports/fertility_audit.json` 和 `reports/fertility_audit_interpretation.md`；
2. 已覆盖源文本 NFC/NFD、tsheg、组合字符、unknown 和 256/512 token 上限检查；
3. 当前 CS336 Raw/Tsheg-boundary 的 8k/16k 请求在藏文源语料上因 pair 类型提前耗尽，实际词表约 1.3k–1.4k；Mark-aware 实际约 3.8k；若要做词表缩放，后续应扩大语料或改为共享藏中语料；
4. **已完成 NLLB 权重下载与 10 条 CPU 冒烟测试**；下一步先分析重复退化，再做 50–100 条分层 zero-shot 诊断；按当前约 25.4 秒/条估算，完整 2,009 条 FLORES 在 CPU 上需约 14 小时，留到资源更合适时执行；
5. 冻结统一评测实现：FLORES、MITRA `bo2zh`、Lotsawa 分域，主指标 chrF++，辅助 BLEU；SacreBLEU 2.6.0 的签名应在 metric 对象完成 `corpus_score` 后调用 `get_signature()`；
6. 对 OpenPecha 的 78 个中文文档、CUTE 的单语文件和 DongbaMIE ZIP/JSON 做正式 schema/profile，产出“可训练/仅分析/仅评测”清单；
7. 有 GPU 后先做 NLLB 10k pilot，再做 100k/全量微调；HY-MT2 只做本地推理和可选 LoRA；
8. 藏文 OCR 阶段以 BDRC 472 条作固定评测；翻译主线稳定后再串联 OCR；
9. 东巴文先核验 DongbaMIE 图像—JSON 对应和字段分布，再决定 Qwen2.5-VL LoRA；Dongba1800 许可证冲突解决前只做本地结构研究。

## 7. 后续 Agent 操作原则

1. 先阅读本文件、`README.md`、`docs/cpu-experiment-plan.md` 和相关 `reports/`。
2. 不覆盖 `data/raw/` 原始文件。
3. 不把 FLORES 或 MITRA 官方评测集混入训练。
4. 处理完整 MITRA 时使用流式读取；当前内存只有约 2 GB 空闲。
5. 不因“看起来像藏文”就声称语义质量正确；没有藏语人工核验时，只报告结构性和自动指标。
6. 任何大模型下载、全量训练、数据规则改变或新增外部数据源，都先说明目的、磁盘/内存代价和影响范围。
7. 重大代码改动前先向用户解释；优先添加小型、可复现的脚本和报告。

## 8. 常用入口

- 项目说明：`README.md`
- CPU 计划：`docs/cpu-experiment-plan.md`
- 藏文基础知识：`docs/藏语知识讲义.md`
- 分词基础：`docs/分词实验-基础概念.md`
- FLORES 检查：`src/validate_flores.py`
- MITRA 抽样：`src/sample_mitra.py`
- MITRA 过滤/划分：`src/prepare_mitra.py`
- 数据登记：`data/manifests/datasets.csv`
- Tokenizer 可视化：`reports/tokenizer_experiments.html`
- NLLB CPU 冒烟输出：`reports/nllb_cpu_smoke.json`
- 新数据下载核验：`outputs/download_spotcheck.txt`
- 环境配置记录：`reports/environment_setup.md`
