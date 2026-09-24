# CPU 环境配置记录

> 更新日期：2026-09-24。对应任务“配好没配的环境”，仅新增依赖，未修改任何既有脚本。

## 1. 配置前状态

`requirements-cpu.txt` 只固定了 5 个包，实际缺失评测与报表所必需的依赖：

- 缺失：`sacrebleu`、`pandas`、`openpyxl`、`matplotlib`、`tabulate`；
- 已存在且**保持不变**：`torch==2.4.1+cpu`、`transformers==4.46.3`、`sentencepiece==0.2.0`、
  `numpy==2.0.2`、`duckdb==1.1.3`、`pyewts==1.0.0`、`huggingface_hub==0.36.2`、`tqdm==4.70.1`。

基线快照：`outputs/environment_baseline_pip_freeze.txt`。

## 2. 新增依赖

| 包 | 版本 | 用途 |
|---|---|---|
| sacrebleu | 2.6.0 | chrF++（主指标）与 BLEU（辅助指标） |
| pandas | 2.3.3 | 统计与表格处理 |
| openpyxl | 3.1.5 | 人工核查工作簿 |
| matplotlib | 3.9.4 | 图表生成 |
| tabulate | 0.9.0 | 纯文本表格输出 |

传递依赖同步固定：`lxml 6.1.3`、`portalocker 3.2.0`、`python-dateutil 2.9.0`、`pytz 2026.4`、
`tzdata 2026.4`、`et-xmlfile 2.0.0`、`six 1.17.0`、`contourpy 1.3.0`、`cycler 0.12.1`、
`fonttools 4.60.2`、`kiwisolver 1.4.7`、`pillow 11.3.0`、`pyparsing 3.3.3`、
`importlib-resources 6.5.2`、`zipp 3.23.1`、`pywin32 312`。

全部版本已写回 `requirements-cpu.txt`。

## 3. 本机环境的四个坑（重要，后续 session 会再遇到）

1. **沙箱禁止运行工作区外的可执行文件。** `git.exe`、`curl.exe`、`fsutil.exe`、`node.exe` 全部返回
   `Access is denied`。因此：
   - 下载 GitHub 仓库不能 `git clone`，改用 GitHub API tarball + Python `urllib`；
   - 下载 HF 文件不能 `curl`，改用 `huggingface_hub.hf_hub_download`；
   - JS 无法用 `node --check` 校验，改用 vendored 的纯 Python `esprima`。
2. **PowerShell 里禁止使用 `2>&1`。** 该语法会强制子进程经由命名管道，沙箱直接杀死进程，
   表现为“无输出 + 空退出码”，极易误判为脚本 bug。本会话中 `python -m pip` 的多次“假失败”
   全部源于此。**诊断子进程时不要合并 stderr。**
3. **pip 自身网络层会挂起。** 经代理时 pip 在取到元数据后停住（连接复用失效），而 `urllib`
   直连 / 代理都正常（已实测：wheel 直连与走代理均 HTTP 200）。解决方案见第 4 节。
4. **代理 `127.0.0.1:7897` 是访问 HuggingFace 的唯一通道**（HF 直连 TCP 被拒），
   而 GitHub 与 PyPI 可直连。实测带宽约 5 MB/s。

## 4. 可复现的离线安装流程

```powershell
# 1) 用 urllib 取 wheel（绕过 pip 的网络层）
& .venv\Scripts\python.exe tools\fetch_wheels.py --out outputs\_wheels `
    sacrebleu pandas openpyxl matplotlib tabulate portalocker lxml python-dateutil `
    pytz tzdata six et-xmlfile contourpy cycler fonttools kiwisolver pillow pyparsing `
    importlib-resources zipp pywin32

# 2) 让 pip 完全离线安装
& .venv\Scripts\python.exe -m pip install --no-index --find-links outputs\_wheels `
    sacrebleu pandas openpyxl matplotlib tabulate
```

依赖闭包是逐个报错补全的（依次缺 `importlib-resources` → `zipp` → `pywin32`），
第 1 步的包列表已包含全部闭包。wheel 缓存在 `outputs/_wheels/`（约 41 MB）。

## 5. 配置后验证

`outputs/environment_verification.txt`（脚本：`tools/verify_env.py`）：

- `torch 2.4.1+cpu`、`transformers 4.46.3`、`numpy 2.0.2`、`tokenizers 0.20.3`
  **均未变动**，既有 NLLB 管线未被破坏；
- `torch.cuda.is_available() == False`（CPU 版，符合预期）；
- NLLB checkpoint 仍是 2,460,457,927 字节，tokenizer 可加载，
  `བོད་ཡིག` 编码为 `[256047, 147019, 13275, 248284, 2]`，`zho_Hans` id = **256200**；
- sacrebleu 功能验证：chrF++ 可正常计算（示例 60.2057）。

## 6. SacreBLEU signature 的正确用法

此前把 `get_signature()` 调在 `BLEUScore`/`CHRFScore` 分数对象上，因此误判 sacrebleu 2.6.0 移除了该 API。实际接口仍然存在于 metric 对象上，并且必须先完成一次 `corpus_score`，让 metric 获得参考集数量：

```python
from sacrebleu.metrics import BLEU, CHRF

bleu = BLEU(tokenize="zh")
chrf = CHRF(word_order=2)
bleu_score = bleu.corpus_score(hypotheses, [references])
chrf_score = chrf.corpus_score(hypotheses, [references])
print(bleu.get_signature())
print(chrf.get_signature())
```

本机已验证可得到：

```text
BLEU  nrefs:1|case:mixed|eff:no|tok:zh|smooth:exp|version:2.6.0
chrF++ nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
```

因此无需降级 SacreBLEU。正式评测时同时保存分数、metric 构造参数和 `get_signature()` 输出。单句或极短测试集上的 BLEU 仍不稳定，正式报告必须使用完整测试集。
