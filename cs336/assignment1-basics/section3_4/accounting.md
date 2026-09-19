下面按 `adamw_accounting` 的 (a)–(d) 给出解答。记：

- \(B\)：batch size  
- \(T\)：context length  
- \(V\)：vocab size  
- \(L\)：num_layers  
- \(D\)：d_model  
- \(H\)：num_heads  
- \(d_{ff}\)：FFN 中间维度，题目假设 \(d_{ff}=\frac{8}{3}D\)  
- float32：每个元素 4 字节  

另外按你给的代码，`token_embeddings` 和 `lm_head` 是两组独立参数，不共享权重。SwiGLU 有 \(W_1,W_2,W_3\) 三个线性层，所以之前 AI 答案里只列 \(W_1,W_2\) 是漏了 \(W_3\)。

---

## (a) AdamW 峰值内存的代数表达式

### 1. 参数数量

每层参数：

- Attention：
  \[
  Q,K,V,O:4D^2
  \]
- SwiGLU：
  \[
  W_1:D\times d_{ff},\quad W_2:d_{ff}\times D,\quad W_3:D\times d_{ff}
  \]
  共：
  \[
  3Dd_{ff}
  \]
- 两个 RMSNorm：
  \[
  2D
  \]

所以每层参数：
\[
4D^2+3Dd_{ff}+2D
\]

总参数：
\[
P=2VD+L(4D^2+3Dd_{ff}+2D)+D
\]

若使用 \(d_{ff}=\frac{8}{3}D\)，则：
\[
P=2VD+L(12D^2+2D)+D
\]

参数内存、梯度内存、AdamW 优化器状态：

- 参数：\(4P\) 字节
- 梯度：\(4P\) 字节
- AdamW 状态：一阶矩 \(m\) 和二阶矩 \(v\)，各 4 字节：
  \[
  8P
  \]
- 非激活总内存：
  \[
  16P
  \]

---

### 2. 激活内存

按题目要求，只计算列出的组件。每层激活元素数：

Attention 子层：

- RMSNorm1：\(BTD\)
- QKV 投影输出：\(3BTD\)
- \(QK^\top\)：\(BHT^2\)
- softmax：\(BHT^2\)
- 加权和输出：\(BTD\)
- 输出投影：\(BTD\)

合计：
\[
6BTD+2BHT^2
\]

FFN 子层：

- RMSNorm2：\(BTD\)
- \(W_1\) 输出：\(BTd_{ff}\)
- \(W_3\) 输出：\(BTd_{ff}\)
- SiLU(gate)：\(BTd_{ff}\)
- element-wise product：\(BTd_{ff}\)
- \(W_2\) 输出：\(BTD\)

合计：
\[
2BTD+4BTd_{ff}
\]

所以每层激活元素数：
\[
8BTD+4BTd_{ff}+2BHT^2
\]

最终部分：

- final RMSNorm：\(BTD\)
- output embedding logits：\(BTV\)
- cross-entropy softmax：\(BTV\)

合计：
\[
BTD+2BTV
\]

因此总激活元素数：
\[
A=L(8BTD+4BTd_{ff}+2BHT^2)+BTD+2BTV
\]

激活内存字节：
\[
M_{\text{act}}=4A
\]

即：
\[
M_{\text{act}}
=
4\left[
L(8BTD+4BTd_{ff}+2BHT^2)+BTD+2BTV
\right]
\]

若 \(d_{ff}=\frac{8}{3}D\)，则：
\[
M_{\text{act}}
=
4\left[
L\left(\frac{56}{3}BTD+2BHT^2\right)+BTD+2BTV
\right]
\]

---

### 3. 总峰值内存

\[
M_{\text{total}}=16P+M_{\text{act}}
\]

即：
\[
M_{\text{total}}
=
16\left[2VD+L(4D^2+3Dd_{ff}+2D)+D\right]
+
4\left[
L(8BTD+4BTd_{ff}+2BHT^2)+BTD+2BTV
\right]
\]

---

## (b) GPT-2 XL 实例化与最大 batch size

GPT-2 XL 配置：

\[
V=50257,\quad T=1024,\quad L=48,\quad D=1600,\quad H=25,\quad d_{ff}=4288
\]

### 1. 参数量

每层：

- Attention：\(4D^2=4\times1600^2=10,240,000\)
- SwiGLU：\(3Dd_{ff}=3\times1600\times4288=20,582,400\)
- RMSNorm：\(2D=3,200\)

每层合计：
\[
10,240,000+20,582,400+3,200=30,825,600
\]

48 层：
\[
48\times30,825,600=1,479,628,800
\]

Token embedding：
\[
VD=50257\times1600=80,411,200
\]

LM head：
\[
VD=80,411,200
\]

final RMSNorm：
\[
D=1600
\]

总参数：
\[
P=1,479,628,800+80,411,200+80,411,200+1600
=1,640,452,800
\]

非激活内存：
\[
16P=16\times1,640,452,800=26,247,244,800\text{ 字节}
\approx 26.25\text{ GB}
\]

---

### 2. 激活内存

每层每 batch 激活元素数：

\[
8TD+4Td_{ff}+2HT^2
\]

代入：

\[
8\times1024\times1600=13,107,200
\]

\[
4\times1024\times4288=17,563,648
\]

\[
2\times25\times1024^2=52,428,800
\]

每层：
\[
13,107,200+17,563,648+52,428,800=83,099,648
\]

48 层：
\[
48\times83,099,648=3,988,783,104
\]

最终部分：

\[
TD=1024\times1600=1,638,400
\]

\[
2TV=2\times1024\times50257=102,926,336
\]

最终合计：
\[
1,638,400+102,926,336=104,564,736
\]

总激活元素每 batch：
\[
3,988,783,104+104,564,736=4,093,347,840
\]

激活内存每 batch：
\[
4\times4,093,347,840=16,373,391,360\text{ 字节}
\approx 16.37\text{ GB}
\]

所以总内存：
\[
M(B)=16,373,391,360\cdot B+26,247,244,800\text{ 字节}
\]

用 GB 表示：
\[
M(B)\approx 16.37\,B+26.25\text{ GB}
\]

---

### 3. 80GB 下最大 batch size

按 \(80\text{ GB}=80,000,000,000\) 字节：

\[
16,373,391,360B+26,247,244,800\le 80,000,000,000
\]

\[
B\le \frac{80,000,000,000-26,247,244,800}{16,373,391,360}
\approx 3.28
\]

所以最大整数 batch size：
\[
\boxed{B_{\max}=3}
\]

---

## (c) 一步 AdamW 的 FLOPs

只统计主要矩阵乘法。前向 FLOPs：

每层：

- QKV 投影：
  \[
  3\times2BTD^2=6BTD^2
  \]
- 输出投影：
  \[
  2BTD^2
  \]
- \(QK^\top\)：
  \[
  2BT^2D
  \]
- attention 加权和：
  \[
  2BT^2D
  \]
- SwiGLU：
  \[
  W_1:2BTDd_{ff},\quad W_2:2BTd_{ff}D,\quad W_3:2BTDd_{ff}
  \]
  共：
  \[
  6BTDd_{ff}
  \]

每层前向：
\[
8BTD^2+4BT^2D+6BTDd_{ff}
\]

最终 LM head：
\[
2BTDV
\]

所以前向总 FLOPs：
\[
F_{\text{forward}}
=
L(8BTD^2+4BT^2D+6BTDd_{ff})+2BTDV
\]

若 \(d_{ff}=\frac{8}{3}D\)：
\[
F_{\text{forward}}
=
L(24BTD^2+4BT^2D)+2BTDV
\]

反向约为前向的 2 倍，所以前向+反向：
\[
3F_{\text{forward}}
\]

AdamW 更新本身约每个参数 \(O(10)\) FLOPs，记为 \(10P\)。因此一步训练总 FLOPs：
\[
\boxed{
F_{\text{step}}
=
3\left[
L(8BTD^2+4BT^2D+6BTDd_{ff})+2BTDV
\right]
+10P
}
\]

其中
\[
P=2VD+L(4D^2+3Dd_{ff}+2D)+D
\]

若 \(d_{ff}=\frac{8}{3}D\)：
\[
\boxed{
F_{\text{step}}
=
3\left[
L(24BTD^2+4BT^2D)+2BTDV
\right]
+10P
}
\]

---

## (d) 单张 H100、50% MFU 下训练 GPT-2 XL 的时间

使用 GPT-2 XL 配置，且 \(B=1024\)。

前向 FLOPs 数值：

每层：
\[
8BTD^2=21,474,836,480,000
\]

\[
4BT^2D=6,871,947,673,600
\]

\[
6BTDd_{ff}=43,164,421,324,800
\]

每层合计：
\[
71,511,205,478,400
\]

48 层：
\[
3,432,537,862,963,200
\]

LM head：
\[
2BTDV=168,634,508,902,400
\]

前向总：
\[
F_{\text{forward}}\approx 3.601\times10^{15}\text{ FLOPs}
\]

一步训练，前向+反向：
\[
F_{\text{step}}\approx 3F_{\text{forward}}
\approx 1.080\times10^{16}\text{ FLOPs}
\]

400K 步：
\[
F_{\text{total}}
=
400000\times1.080\times10^{16}
\approx 4.321\times10^{21}\text{ FLOPs}
\]

H100 峰值：
\[
495\text{ TFLOP/s}=4.95\times10^{14}\text{ FLOP/s}
\]

50% MFU：
\[
0.5\times4.95\times10^{14}=2.475\times10^{14}\text{ FLOP/s}
\]

时间：
\[
\frac{4.321\times10^{21}}{2.475\times10^{14}}
\approx 1.746\times10^7\text{ 秒}
\]

换算成小时：
\[
\frac{1.746\times10^7}{3600}\approx 4850\text{ 小时}
\]

所以：
\[
\boxed{\text{约 }4850\text{ 小时，约 }202\text{ 天}}
\]

---

### 小结

- (a) 总峰值内存：
  \[
  16P+4\left[L(8BTD+4BTd_{ff}+2BHT^2)+BTD+2BTV\right]
  \]
- (b) GPT-2 XL：
  \[
  M(B)\approx 16.37B+26.25\text{ GB}
  \]
  80GB 下最大 batch size：
  \[
  \boxed{3}
  \]
- (c) 一步 AdamW 训练 FLOPs：
  \[
  3F_{\text{forward}}+10P
  \]
- (d) 单张 H100、50% MFU、400K steps、batch size 1024：
  \[
  \boxed{\approx 4850\text{ 小时}}
  \]

另外提醒：之前 AI 的 Transformer FLOPs 列表里，SwiGLU 的 \(W_3\) 矩阵乘被漏掉了，实际 FFN 有三个线性层 \(W_1,W_2,W_3\)。