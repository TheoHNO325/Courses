# NLLB tokenizer 参考说明

本文档记录本项目当前确定使用的 tokenizer 及其在藏中翻译任务中的处理流程。

## 1. 当前配置

```text
模型：facebook/nllb-200-distilled-600M
Tokenizer：NllbTokenizerFast
词表规模：约 256K 个 token
藏文语言代码：bod_Tibt
简体中文语言代码：zho_Hans
繁体中文语言代码：zho_Hant
SentencePiece 模型：sentencepiece.bpe.model
```

本地 tokenizer 文件位于：

```text
outputs/nllb-tokenizer/
```

模型权重和 tokenizer 是绑定的。微调时不能随意换成另一个 tokenizer，否则 token ID 与模型的 embedding/output 矩阵不再对应。

## 2. “文本转换为数字 ID”是什么意思

是的，真正送入 Transformer 的不是字符串，而是整数序列。但是我们不会把原始语料永久替换成 ID；原始文本仍然是数据的主记录，ID 通常在训练批次生成时动态产生，也可以作为缓存。

一条藏文到中文样本大致经过：

```text
原始藏文/中文字符串
        ↓
Unicode 与 tokenizer 内部规范化
        ↓
SentencePiece 预切分
        ↓
BPE 子词切分
        ↓
词表查找：token piece → token ID
        ↓
加入语言标记、结束标记并组成 batch
        ↓
Transformer 编码器/解码器
```

ID 本身没有语言学含义。例如 ID `147019` 不是“某个固定藏文词”的通用编号，它只是当前 NLLB 词表中的一个索引；它必须和同一 checkpoint 的 embedding 矩阵一起使用。

## 3. NLLB tokenizer 的主要算法

### 3.1 规范化

NLLB 的 tokenizer 文件包含一个预编译的 SentencePiece 字符规范化器，以及“连续多个普通空格合并为一个空格”的规则。它不是简单地逐字符查表，也不是我们自己定义的藏文规则。

因此本项目的原则是：

- 输入使用藏文 Unicode，不使用 Wylie；
- 不手动删除 `་`、`།` 等藏文分隔符和标点；
- 不在训练前额外插入所谓的“藏文单词边界”；
- 继续保留已有的数据 Unicode 检查和转换记录。

### 3.2 SentencePiece 与 BPE 子词

tokenizer 使用 SentencePiece 风格的子词模型。它会在空格边界使用可见的边界标记 `▁`，然后根据预训练时学到的 BPE 词表，把文本切成常见的字符片段、音节片段或更长的子词。

这不是传统的藏文分词：

- 不要求知道藏文词典；
- 不保证一个 token 对应一个词或一个音节；
- 同一字符序列在不同上下文中可能被合并成不同长度的子词；
- 罕见字符可能退化为 `<unk>`。

所以当前阶段不做独立的藏文语言学分词。之后如果要研究“原始 Unicode vs. 音节预切分”，应当作为消融实验，而不是默认预处理。

### 3.3 语言标记和特殊 token

常用特殊 token 的 ID 如下（来自当前本地 tokenizer）：

| token | 作用 | ID |
|---|---|---:|
| `<s>` | 序列开始/兼容字段 | 0 |
| `<pad>` | batch 补齐 | 1 |
| `</s>` | 序列结束 | 2 |
| `<unk>` | 词表无法表示的片段 | 3 |
| `bod_Tibt` | 藏文语言标记 | 256030 |
| `zho_Hans` | 简体中文语言标记 | 256200 |
| `zho_Hant` | 繁体中文语言标记 | 256201 |

语言标记也不是风格标签。`zho_Hant` 不会自动把简体转换为繁体，`zho_Hans` 也不会自动把佛典繁体转换为现代简体；目标文本本身仍决定模型学到的表达风格。

## 4. 藏文输入的具体处理

### 4.1 输入文本

MITRA 原始来源是 Wylie 转写，但已在 `data/processed/mitra_v2_conservative/` 中转换为藏文 Unicode。训练输入应使用字段：

```text
source_tibetan
```

不要把 `source_wylie` 直接送进当前 NLLB tokenizer，除非另行设计 Wylie 实验。

### 4.2 源语言编码

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "outputs/nllb-tokenizer"
)
tokenizer.src_lang = "bod_Tibt"

encoded = tokenizer(
    "བོད་ཡིག་གི་སྐད་ཡིག",
    add_special_tokens=True,
    truncation=False,
)
```

一个实际示例的 token 形式为：

```text
藏文原文：བོད་ཡིག་གི་སྐད་ཡིག
tokens：  bod_Tibt  ▁བོད་  ཡིག་  གི་  སྐད་  ཡི  ག  </s>
IDs：     256030    147019  68379  2451  26621  13275  248284  2
```

注意：`▁` 是 tokenizer 的边界标记，不是要写回藏文原文的字符。

### 4.3 目标中文编码

训练目标必须使用 `text_target=`，不能只设置 `tgt_lang` 后再次把中文作为普通源文本传入。正确写法是：

```python
tokenizer.src_lang = "bod_Tibt"
tokenizer.tgt_lang = "zho_Hans"

model_inputs = tokenizer(
    "བོད་ཡིག་གི་སྐད་ཡིག",
    add_special_tokens=True,
)

labels = tokenizer(
    text_target="藏文翻译示例。",
    add_special_tokens=True,
)
```

此时目标序列开头是 `zho_Hans`，而不是 `bod_Tibt`。繁体中文目标只需将 `tgt_lang` 和 `text_target` 对应配置改为 `zho_Hant`。

## 5. Batch、padding、attention mask 和 labels

不同样本长度通常不同，因此组成 batch 时会在右侧补 `<pad>`：

```text
较短序列：token1 token2 </s> <pad> <pad>
较长序列：token1 token2 token3 token4 </s>
```

同时生成 `attention_mask`：

```text
真实 token：1
padding：   0
```

训练 decoder 时，padding 部分不应产生损失。数据整理器通常会把 label 中的 padding ID 替换为 `-100`，PyTorch 的交叉熵损失会忽略这些位置。

## 6. 模型生成阶段

推理时，输入是藏文的 `input_ids` 和 `attention_mask`。为了明确中文输出语言，需要强制 decoder 的第一个语言 token：

```python
target_id = tokenizer.convert_tokens_to_ids("zho_Hans")
generated = model.generate(
    **model_inputs,
    forced_bos_token_id=target_id,
    max_new_tokens=128,
)
text = tokenizer.batch_decode(
    generated,
    skip_special_tokens=True,
)[0]
```

然后 tokenizer 按相反过程把 token ID 还原为文本：去掉语言标记和 `</s>`，移除内部的 `▁` 边界表示，并拼接子词。

## 7. `<unk>` 的含义和当前审计结果

`<unk>` 不是“模型不认识这句话”，而是某个字符或片段没有合适的词表表示。当前审计发现：

- 藏文端 unknown 数量较低，主要是罕见藏文附加符号；
- MITRA 中文端约 2.45% 的 token 是 unknown，主要是佛典罕见字和梵语音译字；
- 例如 `惱`、`訶`、`囉`、`嚩`、`拏`、`唵`、`吽` 等；
- 这不是藏文分词错误。

第一版不直接扩充词表。扩充词表需要调用 `add_tokens`、调整模型 embedding/output 矩阵并重新微调，属于后续对照实验。当前先比较原始传统中文目标和规范化中文目标的 unknown 比例及风格损失。

## 8. 长度策略

NLLB 模型的 tokenizer 上限配置为 1024，但官方训练参考上限是 512。本项目的 CPU 审计结果显示：

- FLORES 最大源句长度：131 tokens；
- MITRA 验证集最大源句长度：119 tokens；
- 当前样本没有超过 512 tokens 的句子。

因此第一版不启用静默截断。后续若加入 OCR 长文本或文档级输入，应先按句号、藏文终止符或段落边界切分，再进行 tokenization。

## 9. 对本项目的结论

当前推荐流程是：

```text
Wylie（仅原始记录）
    ↓ 已完成的转换
藏文 Unicode
    ↓ NLLB SentencePiece/BPE tokenizer
bod_Tibt + 藏文子词 ID 序列
    ↓ NLLB encoder-decoder
zho_Hans / zho_Hant 目标 ID 序列
    ↓ tokenizer.decode
中文字符串
```

原始文本、Unicode 转换结果、tokenizer 版本和模型 checkpoint 都必须保留记录，不能只保存最终数字 ID。这样后续才能比较 tokenizer、繁简规范化和藏文预切分等实验。
