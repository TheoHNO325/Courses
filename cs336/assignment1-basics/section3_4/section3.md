### 1. Transformer Block 内的 7 个 MatMul（形状与 FLOPs）

假设输入 `x` 的形状为 `(B, T, D)`，多头注意力头数为 `H`，每头维度 `d = D / H`。

| 序号 | 操作名称 | 左矩阵 Shape | 右矩阵 Shape | 结果 Shape | FLOPs 公式（`2mnp`） |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Q 投影** (`x W_q`) | `(B, T, D)` | `(D, D)` | `(B, T, D)` | `2 * B * T * D * D` |
| 2 | **K 投影** (`x W_k`) | `(B, T, D)` | `(D, D)` | `(B, T, D)` | `2 * B * T * D * D` |
| 3 | **V 投影** (`x W_v`) | `(B, T, D)` | `(D, D)` | `(B, T, D)` | `2 * B * T * D * D` |
| 4 | **Attention 分数** (`Q K^T`) | `(B, H, T, d)` | `(B, H, d, T)` | `(B, H, T, T)` | `2 * B * H * T * d * T` = `2 * B * T² * D` |
| 5 | **Attention 加权** (`score @ V`) | `(B, H, T, T)` | `(B, H, T, d)` | `(B, H, T, d)` | `2 * B * H * T * T * d` = `2 * B * T² * D` |
| 6 | **输出投影** (`attn_out W_o`) | `(B, T, D)` | `(D, D)` | `(B, T, D)` | `2 * B * T * D * D` |
| 7 | **MLP 第一层** (`x W_1` / `gate`) | `(B, T, D)` | `(D, D_ff)` | `(B, T, D_ff)` | `2 * B * T * D * D_ff` |
| 8 | **MLP 第二层** (`h W_2`) | `(B, T, D_ff)` | `(D_ff, D)` | `(B, T, D)` | `2 * B * T * D_ff * D` |

> **关于第 4、5 步的化简**：因为 `H * d = D`，所以 `2 * B * H * T² * d` 直接化简为 `2 * B * T² * D`，非常漂亮。

---

### 2. 整个 Transformer LM 的总和

假设有 `L` 层，最后的 LM Head 是一个额外的线性层（将维度 `D` 映射到词表大小 `V`）。

- **所有 Block 的总和**：把上面 **1~8 项** 全部加起来，再乘以 `L`。
- **最终 LM Head**：`(B, T, D)` 乘 `(D, V)`，FLOPs = `2 * B * T * D * V`。

所以**总公式**为：
\[
\text{Total FLOPs} = L \times \left( \sum_{i=1}^{8} \text{FLOPs}_i \right) + 2 \cdot B \cdot T \cdot D \cdot V
\]

---

### 3. 关于你提到的 `softmax` 和残差

- **Softmax**：只涉及指数、加法和除法，**不涉及矩阵乘法**，所以算 FLOPs 时通常忽略（或算作 `O(B*H*T²)` 的标量操作，相较 `2mnp` 可忽略不计）。
- **残差连接（Add）**和 **LayerNorm**：只做逐元素加减乘除，算力消耗远低于 MatMul，硬件上（FLOPs counting）通常不算入主要矩阵乘法项。

