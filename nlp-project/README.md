# 少数民族语言翻译课程项目

本项目以藏文到中文翻译为核心，逐步扩展到藏文 OCR、文字类型识别，以及东巴文图像的中文辅助释读或多模态翻译。

## 当前阶段（更新于 2026-09-24）

CPU 侧的数据、Tokenizer 和最小推理链路已经基本打通：

1. FLORES、MITRA 及新增语料均已登记来源、许可、哈希和用途；
2. NLLB/HY-MT2/CS336 的统一 fertility 实验已经完成；
3. NLLB-200 distilled 600M 权重已经完整下载并通过哈希校验；
4. NLLB 已在 FLORES dev 的 10 条样本上完成 CPU 冒烟推理；
5. DongbaMIE、Dongba1800 和 BDRC 藏文 OCR benchmark 已下载，尚未进入正式训练；
6. 已生成可离线查看的 tokenizer 实验 HTML 报告。

当前仍不进行正式 GPU 训练。下一步是冻结统一评测脚本、先做 50–100 条分层 zero-shot 诊断、核验新下载数据的可训练单元，并准备 NLLB 10k pilot 配置；完整 FLORES 推理留到资源更合适时执行。

## 本机约束

- CPU：12 核；
- 内存：15 GB，总可用约 2 GB；
- 磁盘：D 盘约 1 TB 可用。

因此所有数据处理必须优先使用流式读取或分块处理。任何实验都应先在小样本上验证，再扩大规模。

## 仓库内容说明（重要）

本仓库**只包含代码、文档、配置和统计报告**。以下内容被 `.gitignore` 排除，需要按「数据集下载」一节自行获取：

| 类别 | 位置 | 本地体量 |
|---|---|---|
| 全部数据集 | `data/raw/`、`data/interim/`、`data/processed/` | 约 25 GB |
| 模型权重（NLLB-200） | `outputs/nllb-200-distilled-600M/` | 2.46 GB |
| tokenizer 文件 | `outputs/nllb-tokenizer/`、`outputs/hy-mt2-tokenizer/` | 约 27 MB |
| CS336 BPE 词表产物 | `outputs/cs336-bpe-fertility/` | 约 0.5 MB |
| HuggingFace 缓存 | `.hf-cache/` | — |
| 含逐句原始语料的报告 | 见「报告的许可边界」一节 | — |

数据登记表 `data/manifests/datasets.csv`、下载计划 `data/manifests/download_plan.json` 与统计报告 `reports/` 是跟踪的。

## 数据目录约定

- `data/raw/`：原始下载文件，只读，不进行原地修改；
- `data/interim/`：规范化、过滤或转换后的中间文件；
- `data/processed/`：经确认可用于训练、验证或测试的数据；
- `data/manifests/`：来源、许可、版本、哈希和用途记录；
- `notebooks/`：探索性实验；
- `reports/`：统计结果、图表和人工检查记录；
- `configs/`：后续实验配置；
- `src/`：稳定后再提取的正式实现；
- `tools/`：下载、校验与报告生成工具；
- `tests/`：正式实现的测试。

## 数据处理原则

- 保留原始文件，不覆盖、不重写；
- 清洗结果写入新文件，并记录规则和数量变化；
- 训练、验证和测试优先按文档划分，而不是随机拆句；
- 许可证不明确的数据不得进入正式训练集；
- FLORES 等基准测试数据不得进入训练或超参数选择；
- 合成数据必须显式标记，不能作为人工黄金测试集。

CPU 阶段的详细顺序和验收条件见 `docs/cpu-experiment-plan.md`。

NLLB tokenizer 的算法、藏文输入处理、目标语言标记和训练时的 `text_target` 用法见 `docs/NLLB-tokenizer-reference.md`。

项目完整实施路线、模型选择、分词/fertility 实验、数据集规模、微调指标和东巴文图文扩展方案见 `docs/课程项目大纲.md`。

统一 NLLB/HY-MT2/CS336 Raw、Tsheg-boundary、Mark-aware fertility 实验结果见 `reports/fertility_audit_interpretation.md`，机器可读明细见 `reports/fertility_audit.json`。

可交互浏览的 tokenizer 实验汇总见 `reports/tokenizer_experiments.html`。NLLB CPU 冒烟输出见 `reports/nllb_cpu_smoke.json`。

供后续 session/agent 接手的当前计划、进度、文件入口和未完成事项见 `docs/PROJECT_HANDOFF.md`。

---

## 1. 环境准备

```powershell
# 创建虚拟环境（需要 Python 3.9）
python -m venv .venv

# 安装 CPU 版依赖（torch 需从 PyTorch CPU 索引安装）
& .venv\Scripts\python.exe -m pip install -r requirements-cpu.txt

# 校验环境：确认 torch / transformers / numpy 版本与 NLLB 资产可用
& .venv\Scripts\python.exe tools\verify_env.py
```

验证结果写入 `outputs/environment_verification.txt`，详细记录见 `reports/environment_setup.md`。

**网络约定**：本机访问 HuggingFace 必须经过 HTTP 代理 `127.0.0.1:7897`（直连会被拒绝）；GitHub 与 PyPI 可直连。
下载脚本通过 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量读取该代理；若你的网络环境不同，请修改
`tools/download_datasets.py` 与 `tools/plan_downloads.py` 顶部的 `PROXY` 常量。

**gated 数据集**：FLORES-200 与 DongbaMIE 需要在 HuggingFace 申请并接受条款，然后本地登录：

```powershell
& .venv\Scripts\python.exe -m huggingface_hub.commands.huggingface_cli login
```

token 保存在 `%USERPROFILE%\.cache\huggingface\token`，下载脚本会自动读取。

## 2. 模型下载

### 2.1 NLLB-200 distilled 600M（主模型）

模型本体约 2.46 GB，权重文件校验值为 **2,460,457,927 字节**。

```powershell
# 前置：设置代理（HuggingFace 直连不可用）
$env:HTTP_PROXY  = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"

# 下载完整 checkpoint（含 tokenizer、sentencepiece 模型与权重）
& .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/nllb-200-distilled-600M', local_dir='outputs/nllb-200-distilled-600M')"
```

若代理链路中断，可用支持续传的 curl 单独续传权重：

```powershell
curl.exe -L --proxy http://127.0.0.1:7897 --retry 5 --retry-all-errors -C - `
  -o outputs\nllb-200-distilled-600M\pytorch_model.bin `
  "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/pytorch_model.bin?download=true"
```

下载完成后**先核对文件大小约 2,460,457,927 字节**，再运行冒烟测试：

```powershell
# FLORES dev 前 10 条，bod_Tibt -> zho_Hans 本地 CPU 推理
& .venv\Scripts\python.exe src\smoke_nllb_cpu.py --split dev --limit 10 --threads 4
```

结果写入 `reports/nllb_cpu_smoke.json`。注意：CPU 上约 25 秒/条，全量 FLORES dev（997 条）单侧约需 7 小时。

### 2.2 仅 tokenizer（用于编码审计，不下载权重）

```powershell
# NLLB tokenizer
& .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/nllb-200-distilled-600M', local_dir='outputs/nllb-tokenizer', allow_patterns=['tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model'])"

# HY-MT2-1.8B tokenizer（对照基线）
& .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/Hy-MT2-1.8B', local_dir='outputs/hy-mt2-tokenizer', allow_patterns=['tokenizer*','special_tokens_map.json','chat_template.jinja'])"
```

HY-MT2 为 1.8B 因果语言模型，CPU 侧只做极少量冒烟测试，不建议在本机加载权重。

## 3. 数据集下载

所有数据集由 `tools/download_datasets.py` 统一获取，支持**断点续传**：中断后重新执行即可从断点继续，已完整的文件会跳过并校验体积。

```powershell
# 1) 生成下载计划（查询各数据集实际文件清单与体积，写入 data/manifests/download_plan.json）
& .venv\Scripts\python.exe tools\plan_downloads.py

# 2) 查看计划中的全部目标
& .venv\Scripts\python.exe tools\download_datasets.py --list

# 3) 下载全部目标（约 25 GB；CUTE 单项 22.9 GB 耗时较长）
& .venv\Scripts\python.exe tools\download_datasets.py --all

# 或只下载指定目标（逗号分隔）
& .venv\Scripts\python.exe tools\download_datasets.py --only openpecha_c0a2dd042,modern_tibetan_corpus,shajiu_publiccorpus_samples

# 4) 抽检：用真实读取器逐个打开，确认可用而非仅仅存在
& .venv\Scripts\python.exe tools\spotcheck_downloads.py

# 5) 把下载结果登记回 data/manifests/datasets.csv
& .venv\Scripts\python.exe tools\update_dataset_manifest.py --apply
```

### 3.1 目标清单

| 目标 id | 数据集 | 体量 | 许可 | 获取方式 |
|---|---|---:|---|---|
| `openpecha_c0a2dd042` | OpenPecha 平行语料 | 80 MB | 仓库无 LICENSE 文件 | GitHub tarball |
| `modern_tibetan_corpus` | Modern Tibetan Corpus | 78 MB | MIT | GitHub tarball |
| `shajiu_publiccorpus_samples` | Shajiu/PublicCorpus 样例 | 23 MB | 自定义研究条款 | GitHub tarball |
| `dongbamie_code` | DongbaMIE 代码仓 | 3 MB | — | GitHub tarball |
| `dongbamie` | DongbaMIE 标注 + 图像 | 1.8 GB | CC-BY-NC-SA-4.0 | HF（**gated**） |
| `bdrc_tibetan_ocr_benchmark` | BDRC 藏文 OCR 基准 | 234 MB | CC0-1.0 | HF |
| `dongba1800` | Dongba1800 单字符检测 | 222 MB | 许可存在文本冲突 | figshare |
| `cute_bo_subset` | CUTE 藏文子集 | 22.9 GB | CC-BY-4.0 | HF |

```powershell
# 也可分批下载
& .venv\Scripts\python.exe tools\download_datasets.py --only dongbamie,bdrc_tibetan_ocr_benchmark,dongba1800
& .venv\Scripts\python.exe tools\download_datasets.py --only cute_bo_subset
```

### 3.2 关键数据集的精确来源

```text
FLORES-200（gated，仅作固定评测集，绝不参与训练或调参）
  https://huggingface.co/datasets/facebook/flores
  commit 71abf77d8b7beb5cfef59898d6b24d92ab7654fc
  本地：data/raw/flores200_20260529/  bod_Tibt dev 997 + devtest 1012

MITRA Parallel v2 全量训练匹配（836,559 条，Wylie 源文，需用 pyewts 转藏文 Unicode）
  https://github.com/dharmamitra/mitra-parallel
  commit bf7b340cd4f75bda3089479a6367ad94a8cbee10
  文件：v2/bo-zh_matches.ndjson.gz  ->  data/raw/mitra_v2/
  SHA-256: 9179AD2F491A77FCEB06A52FDD93E1D2AECE3AD8623EF0BAA18A08E6373DF897

MITRA 官方评测集（固定测试，不混入训练）
  v2/evaluation/bo-zh/bo2zh.tsv        2000 条，古典佛典中文
  v2/evaluation/bo-zh/lotsawahouse.tsv 1660 条，现代仪轨中文
  ->  data/raw/mitra_v2_eval/
```

其余数据集的精确 URL、commit 与 SHA-256 见 `data/manifests/datasets.csv`。
MITRA 与 FLORES 的下载记录分别见 `reports/mitra_v2_assessment.md` 与 `reports/flores200_download_record.md`。

### 3.3 CUTE 说明

CUTE-Datasets 全量 50.6 GB，本项目**只取藏文相关部分**：`parallel-corpus/bo.txt`（11.2 GB）与
`non-parallel-corpus/n-bo.txt`（11.7 GB）。CUTE 是机器翻译合成数据，按项目原则**只能做合成数据消融，
不得作为黄金评测集**。

## 4. 报告的许可边界

以下三个报告文件包含**逐句原始语料**，因此被 `.gitignore` 排除，只保留在本地：

| 文件 | 内容 | 排除原因 |
|---|---|---|
| `reports/flores200_manual_check_20.csv` | 20 条 FLORES-200 藏中句对 | FLORES-200 为 gated 数据集，公开再分发违反其使用条款 |
| `reports/nllb_cpu_smoke.json` | 10 条 FLORES-200 句对与模型输出 | 同上 |
| `reports/mitra_v2_quality_sample_200.jsonl` | 200 条 MITRA Parallel 记录 | MITRA 为 CC-BY-SA-4.0，整文件属于语料内容再分发 |

其余报告只包含统计量、未知字符片段（单个汉字）与解读文字，因此保留在仓库中。
需要这些样例时，按第 3 节下载对应数据集后本地重新生成即可。

## 5. 当前可复现实验

FLORES-200 藏中全量对齐检查已经可以独立运行：

```powershell
.\.venv\Scripts\python.exe .\src\validate_flores.py
```

脚本默认限制为 256 MB 内存和 2 个 CPU 线程，生成机器可读汇总与 20 条人工检查表。交互式说明见 `notebooks/01_flores_alignment_check.ipynb`。环境依赖固定在 `requirements-cpu.txt`。

### 5.1 重新生成 tokenizer 实验报告

```powershell
& .venv\Scripts\python.exe tools\make_tokenizer_report.py
& .venv\Scripts\python.exe tools\check_report.py   # 校验生成页面的内嵌 JSON 与 JS
```

产物为 `reports/tokenizer_experiments.html`，自包含、可离线打开。

### 5.2 工具脚本一览

| 脚本 | 用途 |
|---|---|
| `tools/plan_downloads.py` | 查询数据集清单，生成下载计划 |
| `tools/download_datasets.py` | 断点续传下载数据集 |
| `tools/spotcheck_downloads.py` | 抽检下载结果是否真正可读 |
| `tools/update_dataset_manifest.py` | 把下载结果登记回 `datasets.csv` |
| `tools/check_manifest.py` | 校验 `datasets.csv` 结构完整性 |
| `tools/verify_env.py` | 校验环境与 NLLB 资产 |
| `tools/fetch_wheels.py` | 用 urllib 取 wheel，绕过 pip 网络问题 |
| `tools/make_tokenizer_report.py` | 生成 tokenizer 实验 HTML 报告 |
| `tools/check_report.py` | 校验 HTML 报告的 JSON/JS 结构 |
| `tools/vendor_esprima.py` | 获取纯 Python 的 esprima 以校验 JS |
| `tools/secret_scan.py` | 上传前扫描凭据泄露 |
| `tools/inventory_upload.py` | 统计将被上传的文件范围 |

**受限沙箱提示**：PowerShell 的 `2>&1` 可能因命名管道限制导致子进程被静默终止；工作区外的可执行文件
（`git`/`curl`/`node`）可能被拒绝执行。排查方法与绕行方案见 `reports/environment_setup.md`。
