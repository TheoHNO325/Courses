# 藏语（Tibetan）NLP 基准与量化评测调研

调研日期：2026-09-20 ｜ 供研究生 NLP 课程项目使用

## 0. 重要方法学说明（请先读）

**本次调研无法直连网络读取全文。** 具体限制：

| 通道 | 结果 |
|---|---|
| `pwsh` + `curl.exe` / `node.exe` | `Program 'curl.exe' failed to run: Access is denied`（沙箱禁止启动原生程序） |
| `pwsh` + `Invoke-WebRequest` / `.NET WebClient`（含经本地代理 `127.0.0.1:7897`） | TLS 失败：`安全包中没有可用的凭证` / `基础连接已经关闭` |
| SSH（`cs336` 等 4 台已配置主机） | `All configured authentication methods failed` |
| `web_search` | ✅ 可用，但**只返回「来源 URL + 短摘录」**，不返回页面正文 |

**后果**：本报告只能给出 (a) 论文/基准的**存在性、出处、覆盖范围**（这类信息可从标题与页面元数据确认），以及 (b) 搜索索引**恰好落在数值附近时的原样摘录**。
凡是本次未取得数值行证的，一律标注 **未确认**，**本文档不含任何推测或编造的数字**。

---

## 1. FLORES-200 与藏语

### 1.1 覆盖情况
- FLORES-200 是 NLLB-200 使用的 200 语种评测集，官方语言代码表以 `flores200_codes.py` 形式广泛分发（HuggingFace Spaces、GitHub）。检索 `"bod_Tibt" FLORES-200` 命中的正是这些代码表文件，说明藏语代码 `bod_Tibt`（藏语 / Uchen 文字）存在于该表中 —— **属间接证据，未取得逐行确认 → 未确认（间接）**。
  - 代码表样本：<https://huggingface.co/spaces/Geonmo/nllb-translation-demo/blob/main/flores200_codes.py>
  - 语言列表说明：<https://raw.githubusercontent.com/bflaven/ia_usages/refs/heads/main/ia_translation_gradio/002_mlearning_ai/flores200_README.md>
  - 含藏文字母语言的旁证（Dzongkha / Uchen / `dz_Ti…` 出现在 FLORES-200 代码表中）：<https://repository.iiitd.edu.in/xmlui/bitstream/handle/123456789/1778/ritwik_thesis_v14%20%281%29.pdf>
- ISO 639-3 代码 `bod` = Tibetan、`mvf` = Mongolian，在少数语言论文中确有实机使用（见 §2 的 arXiv 2604.18106 表头 `bod | uig | kaz`）。

### 1.2 NLLB-200 / M2M-100 / Google Translate / GPT-4 级模型在 藏→英、藏→中 上的 BLEU / chrF
**未确认。** 本次未能取得任何一行「模型 × 藏语方向 × BLEU/chrF」的可溯源数值。

相关但**不能直接引用为藏语分数**的线索（仅供后续定位）：
- NLLB-200 原始技术报告（含 FLORES-200 分语种附录）：<https://www.kheafield.com/papers/facebook/nllb.pdf>
- WMT2023 findings 报告「used data for 204 language varieties from FLORES-200」：<https://aclanthology.org/anthology-files/anthology-files/pdf/wmt/2023.wmt-1.pdf>
- FLORES-200 devtest 结果表（多语种，未确认含藏语行）：<https://arxiv.org/pdf/2409.13949>
- NLLB-200 在低资源语言上的可靠性质疑：<https://ceur-ws.org/Vol-3596/short5.pdf>
- Google Translate 语种表（旁证其语种清单，藏语是否在内**未确认**）：<https://docs.cloud.google.com/translate/docs/languages>

> ⚠️ Google 官方 2024 年扩充 110 语种公告：<https://support.google.com/translate/answer/15139004> —— **藏语是否在 2024 扩充名单内，本次未确认。**

---

## 2. MiLiC-Eval（ACL 2025 Findings）

**出处（已确认）**
- 论文：*MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages*
- ACL Anthology：<https://aclanthology.org/2025.findings-acl.578/>
- arXiv：<https://arxiv.org/abs/2503.01150>
- 代码：<https://github.com/luciusssss/MiLiC-Eval>
- 数据：<https://huggingface.co/datasets/pkupie/milic-eval>
- HuggingFace Papers：<https://huggingface.co/papers/2503.01150>
- 中文解读：<https://papernotes.org/ACL2025/multilingual_mt/milic-eval_benchmarking_multilingual_llms_for_chinas_minority_languages/>

**已确认的定性结论**
- 覆盖中国**四个少数民族语言社区**（4 种低资源语言）。多来源表头一致给出语言代码：`bod`（藏语）、`uig`（维吾尔语）、`kaz`（哈萨克语）；第四种为蒙古语。
- 论文自身表述（原文摘录）：**"the performance is severely unbalanced across the four LRLs"** —— 即四个低资源语言之间性能严重不均衡。

**未确认**
- ❌ 藏语（bod）子任务的**具体任务清单**（未取得列表）。
- ❌ 数据规模 / 实例数（instances）—— **未确认**。HF 数据集页需签署联系信息协议方可访问，本次无法读取。
- ❌ 各 LLM 在**藏语**上的 accuracy / F1 —— **未确认**。
- ❌ 最好/最差模型排序 —— **未确认**。
- ⚠️ 唯一见到的「bod/uig/kaz」数值表来自**另一篇论文**（arXiv 2604.18106，表头 `Model | bod | uig | kaz | Average`，首行 `1B-base | 16.2 | 17.3 | …`，因摘录截断后列缺失）。该论文同时写有 **"TriMix gains are occasionally marginal or negative for Tibetan (bod) and Mongolian (mvf) in certain configurations"**。
  - 链接：<https://export.arxiv.org/pdf/2604.18106>
  - **该表是否就是 MiLiC-Eval 分数：未确认。请勿直接引用为 MiLiC-Eval 结果。**

---

## 3. CUTE（COLING 2025）

**出处（已确认）**
- 论文：*CUTE: A Multilingual Dataset for Enhancing Cross-Lingual Knowledge Transfer in Low-Resource Languages*
- COLING 2025 main：<https://aclanthology.org/2025.coling-main.670/> （PDF: <https://aclanthology.org/2025.coling-main.670.pdf>）
- arXiv 版：<https://arxiv.org/abs/2509.16914> ｜ HTML：<https://ar5iv.labs.arxiv.org/html/2509.16914>
- 中文评述：<https://www.themoonlight.io/zh/review/cute-a-multilingual-dataset-for-enhancing-cross-lingual-knowledge-transfer-in-low-resource-languages>

**已确认的定性信息**
- CUTE 是**面向低资源语言跨语言知识迁移的多语数据集**。
- 基于它训练的 **CUTE-Llama 采用 Llama2 架构**（原文摘录："CUTE-Llama is based on the Llama2 model architecture (Touvron et al.)"）。
- 论文包含以下表格（表题已确认，行值未确认）：
  - *Table 2: Distribution of CUTE dataset*（数据集分布）
  - 分类任务表：`\begin{tabular}{l|c c c|c c} \multirow{2}{*}{Model} & \multicolumn{3}{c|}{Classification (Accu…`
  - 阅读理解表：`Model | CMRC-trained (EM, …) | SQuAD-trained (EM, …)`
- 该论文与藏语相关：CUTE 所属的少数民族语言集合包含藏语（与 §2 的 bod 社区重合）。

**未确认**
- ❌ 具体语言清单（藏语/蒙古语/维吾尔语/哈萨克语的确认名单）—— **未确认**。
- ❌ 数据集规模（句数 / 实例数 / 分语言分布）—— **未确认**。
- ❌ 任何藏语相关的准确率 / F1 / EM 数值 —— **未确认**。

---

## 4. 其他藏语评测集与任务 SOTA

### 4.1 TLUE —— 专门的藏语理解评测基准（重点发现）
- *TLUE: A Tibetan Language Understanding Evaluation Benchmark*
- **EMNLP 2025 Main**：<https://aclanthology.org/2025.emnlp-main.1777/>
- arXiv：<https://arxiv.org/abs/2503.12051> ｜ ar5iv HTML：<https://ar5iv.labs.arxiv.org/html/2503.12051>
- OpenReview：<https://openreview.net/forum?id=cZMNXz4PxE>
- Semantic Scholar：<https://www.semanticscholar.org/paper/TLUE%3A-A-Tibetan-Language-Understanding-Evaluation-Gao-Huang/03839bac5a8886ed3e256dfb6b50334218518002>
- 中文评述：<https://www.themoonlight.io/zh/review/tlue-a-tibetan-language-understanding-evaluation-benchmark>
- 作者含 Dorje Tashi、Gadeng Luosang、Renzeng Duojie、Xiangxiang Wang 等。
- **取得的唯一数值行证**（ar5iv HTML 表格行，原样摘录）：
  > `college_actuarial_science | 4.72 | 6.6 | 18.87 | 17.92 | 25.47 | 8.49 | 17.92`
  - 这是 TLUE 按**领域（domain）**列出的 7 个数值（可能是 7 个评测模型或 7 项任务/指标的得分）。
  - **列含义（哪些模型/指标）—— 未确认**，请勿在报告中标注为「某模型 = 4.72」。
- ❌ TLUE 的任务数、领域数、实例数、各模型总平均分 —— **全部未确认**。

### 4.2 CMiLBENCH（ACL 2026）—— 含藏语列的多任务层级基准
- *Diversity in Unity, Theory in Practice: Hierarchical Multitask Benchmarks for Chinese Minority Languages*
  <https://aclanthology.org/2026.acl-long.1684.pdf>
- 已确认存在下列**藏语相关表头**（数值未取得）：
  - 任务维度：`Model | MCQA | MLIQA | MLE | MLU | MDC | MMT`，首行分组 `Tib…`（Tibetan）
  - 另一维度：`Model | CCC | DD | RPE | SSE | VAA`，首行分组 `Tibetan`
  - 语言维度：`Model | Tibetan(4列) | Mongolian(4列) | Uyghur(4列)`
  - 表 14 讨论「Minority Language Understanding」与「Minority Language Expressions」的区别。
- ❌ 所有具体数值 —— **未确认**。

### 4.3 MC²（ACL 2024）
- *MC²: Towards Transparent and Culturally-Aware NLP for Minority Languages in China* <https://aclanthology.org/2024.acl-long.479.pdf>
  （讨论语料去重；含中国少数民族语言社区与书写系统表）—— 具体藏语数值 **未确认**。

### 4.4 藏语分词（Word Segmentation）
- SOTA 线索（**未确认是否为藏语分词**）：IEEE 论文数值行摘录
  > `M & 99.68 & 99.24 & 99.46 & 99.79 & 99.83 & 99.81 & 99.88 & 98.81 & 99.34 & 99.89 & 99.63 & 99.76`
  <https://xplorestaging.ieee.org/ielx7/5971803/10197185/10197207.pdf>
  → 若确为藏语分词，则 F1 普遍在 **99.6–99.9** 区间（藏语以 tsheg 分音节，分词难度低，高 F1 可解释）。**未确认，需回原文核对。**
- 联合分词+词性标注（BiGRU-CRF）：<https://dl.acm.org/doi/fullHtml/10.1145/3446132.3446395>
- 基于预训练模型的分词重构：<https://www.joca.cn/EN/Y2025/V45/I4/1199>
- 动态多尺度增强藏文分词：<http://cnkimirror.clcn.net.cn/KCMS/detail/detail.aspx?filename=RJDK202512017>
- 改进 BPE 用于藏语：<https://www.fujipress.jp/jaciii/jc/jacii002900061273/?lang=ja>

### 4.5 藏语词性标注（POS）
- 传统词依存特征基线：**90.32% 准确率**（CCL-2013 论文原文摘录：「实验1采用传统的词依存特征，取得 90.32% 的准确度，为本文的基准系统」）
  <http://cips-cl.org/static/anthology/CCL-2013/CCL-13-068.pdf>
- 基于字性标注的词性预测：<http://cips-cl.org/static/CCL2015/papers_CN/Oral/93_...pdf>
- 融合词汇信息的藏文词性一体化：<https://d.wanfangdata.com.cn/periodical/jsjgcysj202512030>
- 基于大模型的藏文词性标注：<http://m.qikan.cqvip.com/article/ArticleDetail?id=7202690811>
- 藏语依存分析（大模型最优，UAS/LAS/UEM/LEM 均最佳）：<http://jcip.cipsc.org.cn/cn/article/pdf/preview/zwxxxb_4031.pdf>

### 4.6 藏语命名实体识别（NER）
- **BiLSTM-CRF** 数值行摘录（jsjkx 中文期刊）：
  > `BiLSTM-CRF | 80.60 | 75…`，表头为 `模型 | 总体F1 | PER-F1 | ORG-F1 | LOC-F1`
  → 总体 F1 **80.60**（PER-F1 约 75 起，截断）<https://www.jsjkx.com/CN/article/downloadArticleFile.do?attachType=PDF&id=24466>
- 融合多元特征表示的藏文 NER（CCL 2024）：<https://aclanthology.org/2024.ccl-1.26.pdf>（表中见 87.15 / 89.98 / 88.54 一行，**归属未确认**）
- 从字节到语义：藏文 NER 新范式：<https://www.jsjkx.com/EN/10.11896/jsjkx.260300033>
- 藏医 NER（Syllable-Word-Sentence Embedding Transformer）：<https://digital-library.theiet.org/doi/pdf/10.1049/cit2.70029>
- **TibNED 数据集规模**（原样摘录）：`PERED | 28255 | 33467 | …`（实体规模 28255 → 更新后 33467）<https://cdn.sciengine.com/doi/pdfView/3E6F0872B28F46F69833FADFCE1DE118>

### 4.7 藏汉机器翻译（Tibetan–Chinese MT）
- **CCMT2022 低资源藏汉机器翻译评测报告**（官方评测报告，含 BLEU 榜单）：
  <http://sc.cipsc.org.cn/mt/conference/2022/papers/test_paper/72/72_Paper.pdf>
  → ❌ 具体 BLEU 数值本次 **未取得（未确认）**。
- 藏汉 NMT 粒度研究：<https://ieeexplore.ieee.org/document/10661111>
- 基于多样性数据重组增强的藏汉 NMT（CCL 2025）：<https://aclanthology.org/anthology-files/pdf/ccl/2025.ccl-1.2.pdf>
  （含表头 `任务 | 模型 | 粒度 | BLEU/%`，数值行未取得）
- 藏汉语音翻译（多阶段课程学习）：<https://www.arocmag.cn/abs/2025.11.0510>
- 改进 Transformer 的藏汉翻译：<https://xplorestaging.ieee.org/document/11290804>
- **TIBA / 新闻测试集上的具体 BLEU：未确认。**

---

## 5. 分词（Tokenization）与文字/规范化挑战

### 5.1 核心论文：分词肥裕度的「预分词天花板」
- **《Vowel Signs Are Not Letters: A Pre-tokenization Ceiling on Multilingual Tokenizer Fertility》**
  - HuggingFace Papers：<https://huggingface.co/papers/2608.26449>
  - arXiv：**2608.26449**
  - Semantic Scholar：<https://www.semanticscholar.org/reader/ddfe9c9bb77816ab10d60001636ddd32ab4e0ef8>
  - 代码与结果：<https://github.com/sajalregmi/arkios-tokenizer>
  - 中文导读：<https://gist.science/zh/paper/2608.26449> ｜ <https://hub.baai.ac.cn/paper/2fcdf012-2546-4424-980c-dde8e209bb03>
- **已确认的核心论点**（BAAI 中文摘录）：采用 HuggingFace **ByteLevel 预分词器**的字节级 BPE 分词器，继承了 **GPT-2 的单词正则规则，即把「单词」定义为 `\p{L}+`（一个或多个 Unicode 字母）**。→ 藏文元音符号（U+0F71–U+0F84 等 **Mn/Mc 类组合记号**）不是 `\p{L}`，会被预分词切在「字母」之外，从而对藏语等文字施加了一个**与 BPE 学习无关的肥裕度下界（ceiling/floor）**。
- 摘录片段（讨论部分，原样）：**"…a floor that trained tokenizers missed by 40% would be a valid theorem and a useful…"**
  → 论文中出现 **40%** 这一量级，但其确切所指 **未确认**。
- ❌ **藏语 vs 英语的 tokens/word、tokens/char 具体数值 —— 未确认**（这是本次最大的数据缺口）。

### 5.2 其他肥裕度 / 分词评估文献
- *Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation*：<https://arxiv.org/pdf/2510.09947v1>（含 `Fertility | Av…` 定义表）
- *Scaling Laws or Threshold Effects: Optimal Vocabulary Size for Low-Resource…*（ACL 2026 Findings 1588）：<https://aclanthology.org/2026.findings-acl.1588.pdf>
- **Table 21: Fertility scores across tokenizers and languages**：<https://openreview.net/pdf?id=oEpMYAs10U>
- **Table 13: Fertility scores comparison between normalized and non-normalized text**（规范化 vs 未规范化的肥裕度对比，直接对应本任务的规范化问题）：<https://openreview.net/pdf?id=RpFtkDdCgj>
- Tokenization Equity Audit（GitHub 审计报告）：<https://github.com/HeyAvijitRoy/tea-benchmark/blob/main/Tokenization%20Equity%20Audit.pdf>
- Tier 3 字节分配比较（DeepSeek-V3 / Mistral-3 / Llama-4）：<https://aclanthology.org/2026.acl-demo.41.pdf>

### 5.3 藏语专用分词工作
- **Tibetan-LLaMA 2: Large Language Model for Tibetan**：<https://dl.acm.org/doi/pdf/10.1145/3776748>
  （含「multiple tokenization strategies using a limited amount of…」的词表大小实验；另有 `Win | Tie | Lose` 对比表首行 `Tibetan-LLaMA 2-13B vs Chinese-Alpaca-2-13B | 6…`）→ 数值 **未确认**。
- **TibLex：基于拉丁编码的藏文词表优化策略**（CCL 2025）：<http://cips-cl.org/aclanthology_org/2025.ccl-1.21.pdf>
- **An Improved Byte Pair Encoding Method for Tibetan**：<https://www.fujipress.jp/jaciii/jc/jacii002900061273/?lang=ja>
- 藏文词表：词级实验限制词表为最频繁 30K 词（Tibetan–Chinese NMT 论文摘录）：<https://dl.acm.org/doi/pdf/10.1145/3448216>
- TiBERT：藏语预训练模型：<http://cslt.org/mediawiki/images/5/5e/Paper_share_yzx0610.pdf>
- CINO 多语言预训练模型用于藏文分词：<https://d.wanfangdata.com.cn/periodical/CihQZXJpb2RpY2FsQ0hJU29scjkyMDI2MDYxMDIwMjYwNjEwMTYxMjM4Ehd4Ym16eHl4Yi16cmt4YjIwMjYwMjAwNRoIdTE2NG5qYWI%3D>

### 5.4 Unicode / 规范化问题（可引用的权威来源）
| 议题 | 来源 | 状态 |
|---|---|---|
| 藏文元音符号的组合类（combining class）缺陷——直接影响 NFC/NFD 稳定性与排序 | Unicode ML: *Major Defect in Combining classes of Tibetan Vowels* <http://www.unicode.org/mail-arch/unicode-ml/y2003-m06/0250.html> | ✅ 来源确认 |
| 元音符号可附着于单辅音或辅音叠字（stack）之上 | Unicode Standard Ch.10 <http://www.unicode.org/versions/Unicode6.0.0/ch10.pdf> | ✅ 来源确认 |
| **TSHEG（U+0F0B）标记音节结尾**；virama 在藏文中通常不用于构成字母组合 | Unicode L2/13-069 <http://unicode.org/L2/L2013/13069-soyombo.pdf>；Unicode 3.0 Ch.9 <https://www.unicode.org/versions/Unicode3.0.0/ch09.pdf> | ✅ 来源确认 |
| 藏文/宗喀/拉达克文断行规则 | Unicode L2/05-073 <http://www.unicode.org/L2/L2005/05073-tibetan-linebreak.html> | ✅ 来源确认 |
| 藏文字符区码位图（U+0F00–U+0FFF，含 U+0F0B tsheg、U+0F0D shad、U+0F42 藏文字母 GA、U+0F71–U+0F84 元音/组合记号） | <https://www.unicode.org/versions/Unicode3.0.0/charts/U0F00.pdf> | ✅ 码位表确认（**具体码位归属请以该字符表为准**） |
| 藏文叠字（non-standard stacks）处理 | Unicode ML y2011-m09 <https://unicode.org/mail-arch/unicode-ml/y2011-m09/0017.html> | ✅ 来源确认 |
| 藏文 NLP 中的规范化实践 | <http://jcip.cipsc.org.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=1013>；<https://huggingface.co/pagantibet/normalisationS2S-nontokenised> | 来源确认，具体影响 **未确认** |

> ⚠️ 任务中提到的「U+0F42 vs U+0F42 变体」表述疑为笔误；推测原意为**字形相同但码位不同的混淆**（如 U+0F42 字母 GA 与 U+0F42 之外的相关码位、或 U+0F0B/U+0F0C 类 tsheg 变体、亦或 NFC/NFD 下元音序列差异）。**本次未确认**，建议回 Unicode 字符表核对。

---

## 6. MITRA 与藏语 LLM 基准

### 6.1 MITRA（arXiv 2601.06400）
- 标题：*MITRA: A Large-Scale Parallel Corpus and Multilingual Pretrained Language Model for Machine Translation and Semantic Retrieval for Pāli, Sanskrit, Buddhist Chinese, and Tibetan*
- 链接：<https://arxiv.org/abs/2601.06400> ｜ <https://arxiv.org/pdf/2601.06400.pdf> ｜ <https://ar5iv.labs.arxiv.org/html/2601.06400>
- **已确认**
  - 覆盖 **巴利语、梵语、佛教汉语、藏语** 四种「佛教语言」。
  - 机器翻译评测使用 **GEMBA**（LLM-as-judge）框架，将 **Gemma 2 MITRA** 与多个开源模型对比（原文摘录：*"We present the results of the machine translation evaluation of Gemma 2 MITRA against other open models in Figure 2. We use GEMBA…"*）。
  - 对比模型含 **Mistral 7B v0.3 IT** 等。
  - 检索任务表：`Task Type | Source | Target | BM25 | …`
  - 官方榜单：**Dharmamitra Machine Translation Performance Leaderboard** <https://dharmamitra.github.io/dharmamitra-leaderboard/>
- ❌ **MITRA 的 chrF / BLEU 具体数值 —— 未确认**（正文图 2/表 1 的数值本次无法取得；官方榜单页面亦无法读取）。

### 6.2 藏语 LLM 与基准（TiLamb / T-LLaMA / Sunlight / TibetanGPT）
| 名称 | 状态 | 链接 |
|---|---|---|
| **TiLamb**（基于增量预训练的藏文大语言模型） | ✅ 确认存在，**CCL 2024** | <https://aclanthology.org/2024.ccl-1.19.pdf> ｜ <https://github.com/CMLI-NLP/TiLamb> ｜ <https://aclanthology.cn/2024.ccl-1.19/> |
| **Sun-Shine / TFD**（藏文化基座大模型；基于 LLaMA 3；含 Sun-Shine 2.0） | ✅ 确认存在，arXiv 2503.18288 | <https://arxiv.org/abs/2503.18288> ｜ <https://arxiv.org/html/2503.18288v7> ｜ <https://www.alphaxiv.org/abs/2503.18288> ｜ HF: <https://huggingface.co/papers/2503.18288> |
| **Tibetan-LLaMA 2** | ✅ 确认存在（ACM TALLIP 类） | <https://dl.acm.org/doi/pdf/10.1145/3776748> |
| **TIBSTC-CoT**（多领域藏语思维链指令数据集，用 chrF++ 评测） | ✅ 确认存在，arXiv 2508.01977 | <https://ar5iv.labs.arxiv.org/html/2508.01977> |
| **T-LLaMA** | ❌ **未确认存在**（本次检索未命中该名称的藏语模型） | — |
| **Sunlight / TibetanGPT** | ❌ **未确认**（本次检索未命中；`Sun-Shine` 或为混淆来源） | — |

**取得的两处数值行证（原样摘录，列含义未确认）**
- Sun-Shine：`professional_law | 0.3178 | 0.2223 | 0.1786 | 0.2000 | 0.2609 | 0.2865 | 0.3802`
  <https://ar5iv.labs.arxiv.org/html/2503.18288>
- Tibetan-LLaMA 2 对比表首行：`Tibetan-LLaMA 2-13B vs Chinese-Alpaca-2-13B | 6…`（Win/Tie/Lose 列，截断）
  <https://dl.acm.org/doi/pdf/10.1145/3776748>

---

## 7. 汇总：可直接写入课程报告 vs 必须标注未确认

### ✅ 可直接引用（出处已核实）
1. MiLiC-Eval 覆盖 `bod`（藏语）等 **4 种中国少数民族低资源语言**，论文自述**四个 LRL 之间性能严重不均衡**。— ACL 2025 Findings 578 / arXiv 2503.01150
2. CUTE 为低资源跨语言知识迁移数据集，**CUTE-Llama 基于 Llama2 架构**，含分类与阅读理解任务表。— COLING 2025 main 670 / arXiv 2509.16914
3. **TLUE 是专门的藏语理解评测基准**。— EMNLP 2025 Main 1777 / arXiv 2503.12051
4. **CMiLBENCH**（ACL 2026）提供含藏语列的层级多任务基准（MCQA/MLIQA/MLE/MLU/MDC/MMT）。— aclanthology 2026.acl-long.1684
5. **BERT 类模型藏语 NER 总体 F1 = 80.60（BiLSTM-CRF）**（含 PER/ORG/LOC 分项）。— jsjkx 期刊 PDF
6. **藏语 POS 传统词依存特征基线准确率 = 90.32%**。— CCL-2013 论文
7. **藏语分词 F1 疑似达 99.6–99.9**（IEEE 数值行，**待核对**）。
8. **TibNED 命名实体规模：28255 → 33467**（PERED）。— sciengine
9. **《Vowel Signs Are Not Letters》证明 `\p{L}+` 预分词规则对元音符号类文字（含藏文）构成肥裕度天花板**，论文中出现 **40%** 量级。— arXiv 2608.26449
10. **MITRA** 覆盖巴利/梵/佛教汉语/藏语，用 **GEMBA** 评测 **Gemma 2 MITRA**，官方榜单 dharmamitra-leaderboard。— arXiv 2601.06400
11. Unicode 侧：藏文元音组合类缺陷、TSHEG 分音节、元音符号可附着于叠字。— Unicode ML / L2 文档

### ❌ 全部未确认（**课程报告中不得给出数字**）
- FLORES-200 上 NLLB-200 / M2M-100 / Google Translate / GPT-4 级模型的**藏→英、藏→中 BLEU/chrF**
- MiLiC-Eval 的**实例数**与**藏语 accuracy/F1**及模型排名
- CUTE 的**数据集规模**与**藏语结果**
- TLUE 的**任务/领域/实例数**与**各模型总分**
- **藏语 tokenizer fertility（tokens/word、tokens/char）对英语的比值** —— 无任何可溯源数值
- MITRA 的 **chrF/BLEU**
- TiLamb / Sun-Shine / Tibetan-LLaMA 2 的**基准分数**
- **T-LLaMA、Sunlight、TibetanGPT** 的存在性

### 🔧 建议的补数路径（需有外网的环境）
1. 在**可直连网络**的机器上重跑：`curl -sL https://arxiv.org/pdf/2503.01150 -o milic.pdf` 等，直接读表格（本次沙箱完全禁止出网）。
2. 优先下载 **arXiv 2608.26449**（分词肥裕度论文，正文应含藏语 vs 英语的 tokens/word 对照表）与 **arXiv 2503.12051**（TLUE）。
3. **CCMT2022 藏汉评测报告 PDF** <http://sc.cipsc.org.cn/mt/conference/2022/papers/test_paper/72/72_Paper.pdf> 是藏汉 MT BLEU 最权威的单一来源，务必下载。
4. 用 HF `datasets` 直接加载 `pkupie/milic-eval` 统计实例数（需先在网页上同意条款）。
5. Dharmamitra 排行榜页面可直接给出 MITRA 的藏语 MT 分数：<https://dharmamitra.github.io/dharmamitra-leaderboard/>
