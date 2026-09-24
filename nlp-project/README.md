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

本仓库**只包含代码、文档、配置和统计报告**。以下内容被 `.gitignore` 排除，需要按第 3、4 节自行获取：

| 类别 | 位置 | 本地体量 |
|---|---|---|
| 全部数据集 | `data/raw/`、`data/interim/`、`data/processed/` | 约 27.6 GB |
| 模型权重（NLLB-200） | `outputs/nllb-200-distilled-600M/` | 2.46 GB |
| tokenizer 文件 | `outputs/nllb-tokenizer/`、`outputs/hy-mt2-tokenizer/` | 约 27 MB |
| CS336 BPE 词表产物 | `outputs/cs336-bpe-fertility/` | 约 0.5 MB |
| HuggingFace 缓存 | `.hf-cache/` | — |
| 含逐句原始语料的报告 | 见第 5 节 | — |

数据登记表 `data/manifests/datasets.csv`、下载计划 `data/manifests/download_plan.json`、
期望布局 `data/manifests/expected_layout.json` 与统计报告 `reports/` 是跟踪的。

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

所有数据集由 `tools/download_datasets.py` 统一获取，支持**断点续传**与**校验**：每个文件都对照计划中
固定的字节数校验，计划中带 SHA-256 的文件还会校验哈希；中断后重新执行会从断点继续，已完整的文件直接跳过。

```powershell
# 1) 生成下载计划（查询各数据集实际文件清单与体积，写入 data/manifests/download_plan.json）
& .venv\Scripts\python.exe tools\plan_downloads.py

# 2) 查看计划中的全部目标
& .venv\Scripts\python.exe tools\download_datasets.py --list

# 3) 下载全部目标（约 27.6 GB；CUTE 单项 22.9 GB 耗时较长，实测约 12 MB/s）
& .venv\Scripts\python.exe tools\download_datasets.py --all

# 4) 抽检：用真实读取器逐个打开，确认可用而非仅仅存在
& .venv\Scripts\python.exe tools\spotcheck_downloads.py

# 5) 把下载结果登记回 data/manifests/datasets.csv
& .venv\Scripts\python.exe tools\update_dataset_manifest.py --apply
```

`plan_downloads.py` 带完整性闸门：只要 11 个必需目标里有任何一个因网络抖动没取到，它就**不会**覆盖已有计划，
而是报错退出，避免把一个残缺的计划写进去。

### 3.1 目标清单（11 项）

| 目标 id | 数据集 | 体量 | 许可 | 获取方式 |
|---|---|---:|---|---|
| `flores200` | FLORES-200 `bod_Tibt`/`zho_Hans` dev+devtest | 547 KB | CC-BY-SA-4.0 | HF（**gated**） |
| `mitra_v2_full` | MITRA Parallel v2 全量匹配 | 80.5 MB | CC-BY-SA-4.0 | 固定 commit 的 HTTPS |
| `mitra_v2_eval` | MITRA 官方 `bo2zh` + `lotsawahouse` | 1.4 MB | 见仓库条款 | 固定 commit 的 HTTPS |
| `dongbamie` | DongbaMIE 标注 + 图像 | 1.8 GB | CC-BY-NC-SA-4.0 | HF（**gated**） |
| `bdrc_tibetan_ocr_benchmark` | BDRC 藏文 OCR 基准 | 234 MB | CC0-1.0 | HF |
| `cute_bo_subset` | CUTE 藏文子集 | 22.9 GB | CC-BY-4.0 | HF |
| `openpecha_c0a2dd042` | OpenPecha 平行语料 | 84 MB | 仓库无 LICENSE 文件 | GitHub tarball |
| `modern_tibetan_corpus` | Modern Tibetan Corpus | 82 MB | MIT | GitHub tarball |
| `shajiu_publiccorpus_samples` | Shajiu/PublicCorpus 样例 | 24 MB | 自定义研究条款 | GitHub tarball |
| `dongbamie_code` | DongbaMIE 代码仓（论文图与代码） | 3.6 MB | — | GitHub tarball |
| `dongba1800` | Dongba1800 单字符检测 | 222 MB | 许可存在文本冲突 | figshare |

```powershell
# 也可按需分批下载
& .venv\Scripts\python.exe tools\download_datasets.py --only flores200,mitra_v2_full,mitra_v2_eval
& .venv\Scripts\python.exe tools\download_datasets.py --only dongbamie,bdrc_tibetan_ocr_benchmark,dongba1800
& .venv\Scripts\python.exe tools\download_datasets.py --only cute_bo_subset
```

### 3.2 为什么下载后目录结构会与本机一致

三个可能造成路径错位的点都已显式处理：

1. **FLORES 的 HF 路径带前缀**。HuggingFace 上文件位于 `data/language/<lang>/<split>.parquet`，而本机布局
   是 `<lang>/<split>.parquet`。计划里对该目标设置 `strip_prefix="data/language/"`，下载后自动搬到正确位置。
2. **MITRA 用固定 commit**，而非分支名：
   `https://raw.githubusercontent.com/dharmamitra/mitra-parallel/bf7b340cd4f75bda3089479a6367ad94a8cbee10/...`，
   保证上游更新不会改变取到的内容。
3. **每个数据集的目标目录写死在计划里**（如 `data/raw/flores200_20260529`），不随工具或日期变化。

结果就是下载完成后 `data/raw/` 下的目录名、层级与文件名与本机完全相同——这一点由第 4 节的校验脚本自动确认。

CUTE-Datasets 全量 50.6 GB，本项目**只取藏文相关部分**：`parallel-corpus/bo.txt`（11.2 GB）与
`non-parallel-corpus/n-bo.txt`（11.7 GB）。它是机器翻译合成数据，按项目原则**只能做消融，
不得作为黄金评测集**。

## 4. 生成派生数据并校验布局

`data/processed/` 与 `data/interim/` 不是下载来的，而是由 `src/` 里的脚本从原始数据算出来的。
完整复现顺序如下：

```powershell
# 4.1 派生验证集划分与过滤结果 -> data/processed/mitra_v2_conservative/
& .venv\Scripts\python.exe src\prepare_mitra.py `
    --input data\raw\mitra_v2\bo-zh_matches.ndjson.gz `
    --config configs\mitra_v2_filter.json `
    --output-dir data\processed\mitra_v2_conservative `
    --report reports\mitra_v2_filtering.json

# 4.2 分词器 fertility 实验的中间抽样文本 -> data/interim/fertility/
& .venv\Scripts\python.exe src\audit_fertility.py
```

`src/audit_fertility.py` 需要 `outputs/nllb-tokenizer/` 与 `outputs/hy-mt2-tokenizer/`（见第 2.2 节）。

### 4.3 一键校验：确认你的数据与本机一致

```powershell
# 全量校验（含逐文件 SHA-256）
& .venv\Scripts\python.exe tools\verify_data_layout.py

# 快速校验（只比目录、文件数、字节数与目录摘要）
& .venv\Scripts\python.exe tools\verify_data_layout.py --quick

# 只校验某几个数据集
& .venv\Scripts\python.exe tools\verify_data_layout.py --only data/raw/flores200_20260529,data/raw/mitra_v2
```

判定依据是仓库中跟踪的 `data/manifests/expected_layout.json`，它记录了 15 个目录的
文件数、总字节数、目录摘要（relpath+size 的 SHA-256），以及体积较小文件的逐个 SHA-256。
校验通过时输出 `RESULT: data layout matches the canonical spec`，任一项不符则以非零码退出：

```text
data/raw/flores200_20260529     OK         4 files        0.56 MB
      verified 4 file hashes
data/raw/mitra_v2               OK         1 files       80.46 MB
data/raw/mitra_v2_eval          OK         2 files        1.37 MB
      verified 2 file hashes
...
RESULT: data layout matches the canonical spec
```

如果某个数据集的目录名或层级与本机不同，这一步会直接报 FAIL，不需要人工比对。

新增数据集后，用 `tools/snapshot_data_layout.py` 重新生成该规范文件：

```powershell
& .venv\Scripts\python.exe tools\snapshot_data_layout.py
```

> **说明**：`data/raw/flores200_bod_tibt/` 与 `data/raw/mitra_v2_probe/` 是早期实验留下的非规范目录
> （已被 `flores200_20260529/` 取代），不属于规范布局，脚本会提示它们是 legacy、可安全忽略或删除。

## 5. 报告的许可边界

以下三个报告文件包含**逐句原始语料**，因此被 `.gitignore` 排除，只保留在本地：

| 文件 | 内容 | 排除原因 |
|---|---|---|
| `reports/flores200_manual_check_20.csv` | 20 条 FLORES-200 藏中句对 | FLORES-200 为 gated 数据集，公开再分发违反其使用条款 |
| `reports/nllb_cpu_smoke.json` | 10 条 FLORES-200 句对与模型输出 | 同上 |
| `reports/mitra_v2_quality_sample_200.jsonl` | 200 条 MITRA Parallel 记录 | MITRA 为 CC-BY-SA-4.0，整文件属于语料内容再分发 |

其余报告只包含统计量、未知字符片段（单个汉字）与解读文字，因此保留在仓库中。
需要这些样例时，按第 3 节下载对应数据集后本地重新生成即可。

## 6. 当前可复现实验

FLORES-200 藏中全量对齐检查已经可以独立运行：

```powershell
.\.venv\Scripts\python.exe .\src\validate_flores.py
```

脚本默认限制为 256 MB 内存和 2 个 CPU 线程，生成机器可读汇总与 20 条人工检查表。交互式说明见 `notebooks/01_flores_alignment_check.ipynb`。环境依赖固定在 `requirements-cpu.txt`。

### 6.1 重新生成 tokenizer 实验报告

```powershell
& .venv\Scripts\python.exe tools\make_tokenizer_report.py
& .venv\Scripts\python.exe tools\check_report.py   # 校验生成页面的内嵌 JSON 与 JS
```

产物为 `reports/tokenizer_experiments.html`，自包含、可离线打开。

### 6.2 从零复现的完整顺序

```powershell
git clone https://github.com/TheoHNO325/Courses.git
cd Courses\nlp-project

python -m venv .venv
& .venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
& .venv\Scripts\python.exe tools\verify_env.py                 # 1. 环境

# 2. 模型
& .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/nllb-200-distilled-600M', local_dir='outputs/nllb-200-distilled-600M')"

# 3. 数据（约 27.6 GB）
& .venv\Scripts\python.exe tools\plan_downloads.py
& .venv\Scripts\python.exe tools\download_datasets.py --all
& .venv\Scripts\python.exe tools\spotcheck_downloads.py

# 4. 派生数据
& .venv\Scripts\python.exe src\prepare_mitra.py --input data\raw\mitra_v2\bo-zh_matches.ndjson.gz --config configs\mitra_v2_filter.json --output-dir data\processed\mitra_v2_conservative --report reports\mitra_v2_filtering.json
& .venv\Scripts\python.exe src\audit_fertility.py

# 5. 校验：应输出 RESULT: data layout matches the canonical spec
& .venv\Scripts\python.exe tools\verify_data_layout.py
```

### 6.3 工具脚本一览

| 脚本 | 用途 |
|---|---|
| `tools/plan_downloads.py` | 查询数据集清单、生成下载计划；缺目标时拒绝写出残缺计划 |
| `tools/download_datasets.py` | 断点续传 + 尺寸/SHA-256 校验下载数据集 |
| `tools/spotcheck_downloads.py` | 抽检下载结果是否真正可读 |
| `tools/update_dataset_manifest.py` | 把下载结果登记回 `datasets.csv` |
| `tools/check_manifest.py` | 校验 `datasets.csv` 结构完整性 |
| `tools/snapshot_data_layout.py` | 生成 `data/manifests/expected_layout.json` 规范 |
| `tools/verify_data_layout.py` | 校验本地数据布局是否与规范一致 |
| `tools/verify_env.py` | 校验环境与 NLLB 资产 |
| `tools/fetch_wheels.py` | 用 urllib 取 wheel，绕过 pip 网络问题 |
| `tools/make_tokenizer_report.py` | 生成 tokenizer 实验 HTML 报告 |
| `tools/check_report.py` | 校验 HTML 报告的 JSON/JS 结构 |
| `tools/vendor_esprima.py` | 获取纯 Python 的 esprima 以校验 JS |
| `tools/secret_scan.py` | 上传前扫描凭据泄露 |
| `tools/inventory_upload.py` | 统计将被上传的文件范围 |

**受限沙箱提示**：PowerShell 的 `2>&1` 可能因命名管道限制导致子进程被静默终止；工作区外的可执行文件
（`git`/`curl`/`node`）可能被拒绝执行。排查方法与绕行方案见 `reports/environment_setup.md`。
