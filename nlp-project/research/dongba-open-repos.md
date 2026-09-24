# 东巴文（Dongba / Naxi pictographic script）开放仓库检索结果

> 方法限制：本次仅使用 `web_search`（无页面抓取能力，shell 也无网络）。
> 因此「仓库存在」可由检索结果标题确认，但**目录结构、文件类型、精确样本数、许可证、最后更新时间**多数只能从片段推断，凡不能确证者一律标注 **未确认**。

## A. 已确认存在的仓库 / 数据集托管页

| 仓库 URL | 内容 | 规模 | 许可 | 备注 |
|---|---|---|---|---|
| https://github.com/thinklis/DongbaMIE | DongbaMIE：东巴象形字**多模态信息抽取数据集**（代码仓，标题写明 EMNLP 2025 Findings） | arXiv HTML 附录表格片段出现总数 `23,530`、`2,539` 及 `81,372 / 65,140` 等计数，**列含义未确认** | 未确认 | 论文：https://aclanthology.org/2025.findings-emnlp.51.pdf ；https://arxiv.org/html/2503.03644v1 ；HF 论文页 https://huggingface.co/papers/2503.03644 |
| https://huggingface.co/datasets/thinklis/DongbaMIE | 上述数据集的 HuggingFace 托管版（数据集页面已确认存在） | 未确认 | 未确认 | 检索片段只给出页面标题与论文计数表，无 dataset card 细节 |
| https://github.com/ma-yu-qi/STEF | STEF：基于 Swin Transformer 的增强特征金字塔融合模型，用于**东巴字符检测**（代码仓） | 未确认（仓库内含数据与否未确认） | 未确认 | 对应论文 DOI 10.1186/s40494-024-01321-2（npj Heritage Science，2024）：https://www.nature.com/articles/s40494-024-01321-2 ；同文提到数据集 DB200 (Ma et al)，但 DB200 与 STEF 仓库的对应关系**未确认** |
| https://github.com/mountain/bor | 仓库简介原文：`bor is a set of dongba characters in metafont`（东巴字形 Metafont 定义，属字形/字体类） | 未确认（字形数量无片段） | 未确认 | 目前唯一确认的「东巴字形库」类 GitHub 仓库 |
| https://github.com/AIGeeksGroup/OmniOCR | `OmniOCR: Generalist OCR for Ethnic Minority Languages`（少数民族语言通用 OCR 代码库） | 未确认 | 未确认 | 论文 https://arxiv.org/pdf/2602.21042v1 、https://huggingface.co/papers/2602.21042 ；**是否包含东巴文训练/评测分支未确认**（片段只出现 Tibetan 等语言列） |
| https://huggingface.co/Apeters247/naxi-qwen3-14b-v5 | 纳西语（Naxi）Qwen3-14B 模型仓，README 含方言/语法评分表 | 未确认 | 未确认 | https://huggingface.co/Apeters247/naxi-qwen3-14b-v5/blob/main/README.md ；**是否训练/涉及东巴文（象形文字）数据未确认** |
| https://springernature.figshare.com/articles/dataset/Dataset_for_Single_Character_Detection_in_Dongba_Manuscripts/26969755 | 东巴手稿**单字符检测数据集（Dongba1800）**，随 Scientific Data 论文发布（含 file=49083013） | 名称含 1800；同一论文 Table 1 称 "Metadata about the Dongba1800 dataset" | 未确认 | 论文 https://www.nature.com/articles/s41597-025-05434-6 ；元数据表 https://www.nature.com/articles/s41597-025-05434-6/tables/1 ；相关片段 "to 1012 different pictographic character classes" 出自 https://www.nature.com/articles/s40494-026-02874-0_reference.pdf ，**是否指该数据集未确认** |
| https://mzyy.muc.edu.cn/info/1102/2007.htm | DB1404：**手写东巴文单字数据集**，中央民族大学「民族语言智能分析与安全治理」教育部重点实验室面向社会公开 | 未确认 | 未确认（需提交《手写东巴文数据集DB1404使用申请表》） | 公告 https://mzyy.muc.edu.cn/info/1098/1997.htm 、https://xingong.muc.edu.cn/info/1042/4055.htm 、下载中心 https://mzyy.muc.edu.cn/xzzx.htm ；非 GitHub/HF 托管 |

## B. 论文级线索（数据/代码是否开源、链接均未确认）

| 来源 URL | 相关资产 |
|---|---|
| https://arxiv.org/html/2304.00746v4 | VGTS 论文中的 **DBH（Dongba Hieroglyphics）** 数据集："a new dataset featuring Dongba characters… Naxi minority"。发布渠道未确认 |
| https://link-proxy.springer.com/content/pdf/10.1007/s44443-025-00386-8.pdf | 片段 "and the DB200 (Ma et al"，即 **DB200** 数据集名称来源 |
| https://dl.acm.org/doi/pdf/10.1145/3721980 | Dongba→汉语**机器翻译 + 迁移学习**（DOI 10.1145/3721980）；平行语料是否公开未确认；机构页 https://pure.ecnu.edu.cn/en/publications/dongba-machine-translation-with-transfer-learning-leveraging-pre-/ |
| https://www.nature.com/articles/s40494-025-02096-w | 多模态上下文感知东巴文翻译（Qwen2.5-VL 东巴→中文）；代码/数据未确认 |
| https://www.nature.com/articles/s40494-026-02494-8 | 跨模态对齐构建东巴单字数据集；用 7 个经典模型验证；数据集是否开放未确认 |
| https://www.sciencedirect.com/science/article/abs/pii/S0957417425025618 | Lightweight Dongba character recognition baseline（DOI 10.1016/j.eswa.2025.128944）；代码未确认 |
| https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.70161 | EdgeCANet：手写东巴字符识别；代码/数据未确认 |
| https://www.nature.com/articles/s40494-026-02793-0_reference.pdf | LMAFNet：轻量多尺度注意力融合的手写东巴字符识别（标题出处） |
| https://www.scilit.com/publications/5cffb2306625a5565ca0430fb4eae45c | DongbaBPN：基于 Boundary GCN 的东巴字符检测（标题出处） |
| https://www.nature.com/articles/s40494-026-02874-0_reference.pdf | STMS-Net 与 HiST-Glyph：历史象形手稿智能分析 |
| https://www.sciencedirect.com/science/article/abs/pii/S0957417426022670 | GC-Fusion：字形-坐标融合的多模态 LLM 东巴手稿翻译 |
| https://www.ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/joe.2018.9177 | Intelligent classification on images of Dongba ancient books |
| https://bcpublication.org/index.php/SJISR/article/download/8999/8942/11988 | Unlocking Ancient Pictographs：多模态 LLM 理解东巴字符 |
| https://pure.aber.ac.uk/ws/portalfiles/portal/92454687/Article.pdf | CNN 东巴象形字分类（ResNet 最高） |
| https://www.ebiotrade.com/newsf/2025-7/20250702002316782.htm | 中文报道：Dongba1800 数据集构建与多模型验证 |

## C. 字形/字体与 Unicode 数据源（非 GitHub/HF 仓库）

| URL | 内容 |
|---|---|
| https://www.babelstone.co.uk/Fonts/ | BabelStone Unicode Fonts 主页；**BabelStone Naxi LLC** 字体（2017-05 发布，覆盖纳西东巴文）供下载；被论文引用见 http://www.fluxus-editions.fr/gla5-xudu.pdf |
| http://www.unicode.org/L2/L2017/17339-n4898-naxi-dongba-rev.pdf | UTC/ISO 提交的 Naxi Dongba 码位与字形清单（片段含 "63 DONGBA CHARACTER BBER ZEEQ … mosquito 蚊子"） |
| https://repos.ecosyste.ms/hosts/GitHub/topics/dongba | GitHub `dongba` topic 聚合页（确认存在该 topic）；其列出的完整仓库清单未确认 |

## D. 未确认（检索无法证实，禁止据此假设存在）

1. **STEF 仓库内部构成**：文件类型、是否随仓发布东巴检测数据集、样本数、许可证、最后更新时间——全部未确认。
2. **DongbaMIE 的许可证**、精确样本/标注规模与最后更新时间——未确认（仅确认论文为 EMNLP 2025 Findings）。
3. **除 `thinklis/DongbaMIE` 外，是否存在任何 HuggingFace 数据集/组织专门托管东巴图像**——未确认（多次不同措辞检索无结果）。
4. **任何东巴 OCR 模型权重**（HF model 仓）——未确认。
5. **任何东巴文字体/字形 GitHub 仓库**：除 `mountain/bor`（Metafont）外未确认；BabelStone Naxi 为官网下载字体，非代码仓。
6. **`mountain/bor` 的字形数量、许可证、更新时间**——未确认。
7. **`Apeters247/naxi-qwen3-14b-v5` 的许可证，及其是否含东巴文（象形字）数据**——未确认。
8. **DBH（VGTS）、DB200、Dongba1800(figshare) 的下载方式、许可证与样本量细节**——未确认（Dongba1800 仅确认 figshare 条目与论文存在）。
9. **东巴文平行语料/机器翻译语料的开源仓库**（含 DBLP/DOI 10.1145/3721980 配套数据）——未确认。
10. **上表 B 中所有论文的代码/数据开源链接（任何 github.com/huggingface.co URL）**——均未在检索结果中出现，未确认。
11. **任何仓库的「最后更新时间」**——检索片段未提供日期元数据；仅有论文年份（STEF 2024、Dongba1800/Scientific Data 2025、DongbaMIE 2025）。
12. **Kaggle 上的东巴文数据集**——未确认（专项检索无结果）。
