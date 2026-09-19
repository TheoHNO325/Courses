import torch.nn as nn
from einops import rearrange, einsum
import torch

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.device = device
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        ).to(device)
        # 初始化
        std = (2 / (in_features + out_features)) ** 0.5   # 注意：std 应该是 sqrt(2/(...))，不是 2/(...)
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,      # 按照题意，a = -3 * sqrt(std) ? 这里是笔误？题目说 a=-3*std**0.5, b=3*std**0.5
            b=3 * std        # 但通常 trunc_normal 的边界是 mean ± 2*std，这里用 ±3*std 也没问题
        )

        
    def forward(self,x):
        
        y = einsum(x.to(self.device),self.weight,
        "... infit, outfit infit  -> ... outfit")
        return y

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.device = device
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
        ).to(device)

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1,
            a=-3,      # 按照题意，a = -3 * sqrt(std) ? 这里是笔误？题目说 a=-3*std**0.5, b=3*std**0.5
            b=3        # 但通常 trunc_normal 的边界是 mean ± 2*std，这里用 ±3*std 也没问题
        )
    
    def forward(self,x):
        
        return self.weight[x]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
    
        self.device = device
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.ones(d_model,  **factory_kwargs))

    def forward(self,x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x / ((torch.sum(x**2,dim=-1,keepdim=True) / self.d_model)**0.5 + self.eps) * self.weight
        return rms.to(in_dtype)

def SiLu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,device = None,dtype=None):
        super().__init__()

        self.w1 = Linear(d_model,d_ff,device,dtype)
        self.w2 = Linear(d_ff,d_model,device,dtype)
        self.w3 = Linear(d_model,d_ff,device,dtype)

        self.device = device

    def forward(self,x):
        return self.w2(SiLu(self.w1(x)) * self.w3(x))
    


# class RoPE(nn.Module):
#     def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
#         super().__init__()

#         theta_i = 1 / torch.tensor([theta ** ((2*k-2) / d_k) for k in range(1,d_k // 2 + 1)])

#         row = torch.zeros(max_seq_len, d_k,d_k)

#         for k in range(d_k // 2 ):
#             # print(torch.tensor([[[torch.cos(theta_i[k] * i), -torch.sin(theta_i[k] *i)],[torch.sin(theta_i[k] * i), torch.cos(theta_i[k] * i)]] for i in range(max_seq_len)]).shape)
#             # print( torch.tensor([[[torch.cos(theta_i[k] * i), -torch.sin(theta_i[k] *i)],[torch.sin(theta_i[k] * i), torch.cos(theta_i[k] * i)]] for i in range(max_seq_len)]))
#             row[:,2*k:2*k+2,2*k:2*k+2] = torch.tensor([[[torch.cos(theta_i[k] * i), -torch.sin(theta_i[k] *i)],[torch.sin(theta_i[k] * i), torch.cos(theta_i[k] * i)]] for i in range(max_seq_len)])
#             # print(row.shape)
#             # print(row[0,:8,:8])
#             # print(row[2,:8,:8])

#         self.rope = row.to(device)


#     def forward(self,x,token_positions):
        
#         row_selected = self.rope[token_positions,:,:]

#         res = einsum(x,row_selected,
#                      '... seq_len d_k, seq_len d_k1 d_k -> ... seq_len d_k1')
#         return res
    
# 更帅气的写法
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        theta = 1 / torch.tensor([theta ** ((2*k-2) / d_k) for k in range(1,d_k // 2 + 1)])
        positions = torch.tensor([i for i in range(max_seq_len)])

        angles = torch.outer(positions,theta) 
        
        cosine = torch.cos(angles).unsqueeze(dim=1)
        sine = torch.sin(angles).unsqueeze(dim=1)

        # print(self.cos.shape)

        self.register_buffer('cos',cosine,persistent=False)
        self.register_buffer('sin',sine,persistent=False)


    def forward(self,x,token_positions = None):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        y = torch.empty_like(x)
        if token_positions == None:
            token_positions = torch.arange(x.size(1), device=x.device)
        y[...,0::2] = self.cos[token_positions] * x_even - self.sin[token_positions] * x_odd
        y[...,1::2] = self.sin[token_positions] * x_even + self.cos[token_positions] * x_odd

        # 我敲！这也太帅了！
        return y
    
def softmax(x,dim):
    # 假设维度是batch seq d_k
    x_max = torch.max(x,dim=dim,keepdim=True).values
    a = torch.exp(x-x_max)

    return a / torch.sum(a,dim=dim,keepdim=True)


def scaled_dot_product_attention(queries,keys,values,mask):
    dut = einsum(queries,keys,'batch ... q_len d_k, batch ... k_len d_k -> batch ... q_len k_len')
    mask_inf = torch.zeros(mask.shape, device=dut.device, dtype=dut.dtype)
    mask_inf[~mask] = -torch.inf
    # print(mask_inf)
    # print(mask_inf.shape)

    # print(dut)
    dut = dut + mask_inf
    # print(dut)
    dut = dut / queries.shape[-1] ** 0.5
    
    print(dut.shape)
    
    attention_score = softmax(dut, -1)

    output = einsum(attention_score,values,'batch ... q_len k_len, batch ... k_len d_v -> batch ... q_len d_v')
    
    return output

class Causal_multi_head_attention(nn.Module):
    def __init__(self,d_model,num_heads,max_seq_len=2056,theta=10000,if_rope=True):
        super().__init__()

        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads

        self.q_proj = Linear(num_heads * self.d_k, d_model)
        self.k_proj = Linear(num_heads * self.d_k, d_model)
        self.v_proj = Linear(num_heads * self.d_v, d_model)

        self.output_proj = Linear(d_model, num_heads * self.d_v)
        self.if_rope = if_rope
        if self.if_rope:
            self.rope = RoPE(theta=theta,d_k=self.d_k,max_seq_len=max_seq_len)
        # print(self.rope.state_dict().keys())

    def forward(self,x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        query = self.q_proj(x).view(batch_size,seq_len,self.num_heads,self.d_k)
        key = self.k_proj(x).view(batch_size,seq_len,self.num_heads,self.d_k)
        value = self.v_proj(x).view(batch_size,seq_len,self.num_heads,self.d_v)

        # print(query.shape)
        # print(key.shape)
        if self.if_rope:
            query = self.rope(query)
            key = self.rope(key)

        query = rearrange(query,"batch seq head dim -> batch head seq dim")
        key = rearrange(key,"batch seq head dim -> batch head seq dim")
        value = rearrange(value,"batch seq head dim -> batch head seq dim")


        # print(query.shape)
        # print(key.shape)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        
        # 低水平实现
        # for i in range(self.num_heads):
        #     head.append(scaled_dot_product_attention(query[:, :, i:i+self.d_k],key[:,:,i:i+self.d_k],value[:,:,i:i+self.d_v],causal_mask))

        # 高水平的AI都用
        multi_head_results = scaled_dot_product_attention(query,key,value,causal_mask)
        # print("multi_resul", multi_head_results.shape)
        # multi_head_results = torch.concat(multi_head_results,dim=1)                                                  

        multi_head_results = multi_head_results.permute(0, 2, 1, 3)   # [batch, seq_len, num_heads, head_dim]
        merged = multi_head_results.reshape(batch_size, seq_len, -1)    
        output = self.output_proj(merged)
        return output


class transformer_block(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len,theta):
        super().__init__()
        self.attention = Causal_multi_head_attention(d_model,num_heads,max_seq_len,theta)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model,d_ff)


    def forward(self,x):
        output = self.attention(self.ln1(x))
        y = x + output
        output2 = self.ffn(self.ln2(y))
        y2 = y + output2
        return y2

class transformer_lm(nn.Module):
    def __init__(self,vocab_size,context_length,num_layers,d_model,num_heads,d_ff,max_seq_len,theta):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size,d_model)
        self.layers = nn.ModuleList([
            transformer_block(d_model, num_heads, d_ff,max_seq_len,theta) for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model,vocab_size)
        self.seq_len = context_length

    def forward(self,x):
        x =self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.lm_head(self.ln_final(x))
        return x 
    
'''
lm head 把 x的维度从d_model映射到词表大小的logits
交叉熵就是 -log p(target) 的平均。随机初始化的模型，对"下一个 token 是谁"没有任何信息，它的预测分布接近 vocab 上的均匀分布，于是 p(target) ≈ 1/V
'''




