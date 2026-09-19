import torch

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

    
def load_checkpoint(src,model,optimizer):
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

def cross_entropy(logits,targets):

    '''
    logits: batch_size vocab_size
    targets: batch_size
    '''

    max_val = torch.max(logits,dim=-1,keepdim=True)[0]

    shifted = logits-max_val

    loss = -shifted[torch.arange(logits.size(0)),targets] + torch.log(torch.sum(torch.exp(shifted),dim=-1,keepdim=True))
    return loss.mean() # / batch (/ seq if do)


'''
推一下公式 let's push the formula
seq = x0,x1,……x_i, x_i+1
- log p_theta(x_i+1 | x_i) = - log exp^{logits[x_i+1]} + log sum{exp^{logits[i] for i in seq}} 
= -logits[x_i+1] + log(sum(exp^{logits[i] for i in seq}))

but adding on shifted:

softmax = exp^{logits[x_i+1]} / sum(exp^{logits[i] for i in seq})
= exp^{logits[x_i+1] - logits.max()}  / sum(exp^{logis[i] - logits.max()} for i in seq)

let shifted = logits - logits.max() and put in that.

for details, see cs336\assignment1-basics\section3_4\section4.md.
'''

def perplexity(logits,targets):
    
    max_val = torch.max(logits,dim=-1,keepdim=True)[0]

    shifted = logits-max_val

    loss = -shifted[torch.arange(logits.size(0)),targets] + torch.log(torch.sum(torch.exp(shifted),dim=-1,keepdim=True))

    return torch.exp(loss.mean()) # for only sequence

from collections.abc import Callable
from typing import Optional
import torch
import math
# import matplotlib.pyplot as plt

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params , weight_decay, betas,  lr=1e-3,eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, 'weight_decay':weight_decay, 'beta1':betas[0], 'beta2': betas[1],'eps':eps }
        super().__init__(params, defaults)

    def step(self,closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            lam = group['weight_decay']
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                p.data *= (1 - lr * lam)

                t = state.get("t", 0)
                t = t+1
                m = state.get("m", torch.zeros(grad.shape))
                v = state.get("v", torch.zeros(grad.shape))
                alpha = lr * math.sqrt(1 - beta2 ** t) / (1-beta1 ** t)
                state['m'] = beta1 * m + (1-beta1)* grad
                state['v'] = beta2 * v + (1-beta2)* grad**2

                m = state.get("m", torch.zeros(grad.shape))
                v = state.get("v", torch.zeros(grad.shape))
                p.data -= alpha * m / (v**0.5 + eps)
                state['t']  = t

        return loss
'''
Adam optimizer infer

let learning rate = alpha 
let weight decay = lam
let beta1, beta2 control the updates to the moment edtimates

alpha_t = alpha * (……)
update theta : *= (1- alpha * lam)
update m = m,beta1,grad
update v = v,beta2,grad^2
update theat : -= alpha_t, m, v
'''
def scheduler(t, alpha_max,alpha_min,t_w,t_c):
    if t < t_w:
        return t/t_w * alpha_max
    elif t < t_c:
        return alpha_min + (1+math.cos((t-t_w)/(t_c-t_w)*math.pi))*(alpha_max-alpha_min) / 2
    else:
        return alpha_min

def gradient_clipping(parameters, max_l2_norm):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6) 
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= scale

# # ------------------------------------------------------------
# # 设置学习率列表和训练步数
# # ------------------------------------------------------------
# learning_rates = [1, 10, 100]   # 即 1e1, 1e2, 1e3
# num_steps = 100

# # 用于存储每个学习率下的 loss 历史（用于绘图）
# all_losses = {}

# # ------------------------------------------------------------
# # 对每个学习率分别训练
# # ------------------------------------------------------------
# for lr in learning_rates:
#     print(f"\n=== Training with lr = {lr} ===")
#     # 每次重新初始化权重（避免受到上次训练的影响）
#     weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
#     opt = SGD([weights], lr=lr)
    
#     loss_history = []
#     for t in range(num_steps):
#         opt.zero_grad()
#         loss = (weights ** 2).mean()
#         loss.backward()
#         opt.step()
        
#         # 记录 loss（使用 item() 提取标量）
#         loss_history.append(loss.item())
        
#         # 可选：每 1000 步打印一次当前 loss（避免输出太多）
#         if (t + 1) % 1 == 0:
#             print(f"Step {t+1:5d}, loss = {loss.item():.6f}")
    
#     all_losses[lr] = loss_history

# # ------------------------------------------------------------
# # 绘图：将三条 loss 曲线画在同一张图上（双轴图仅显示 loss）
# # ------------------------------------------------------------
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # 颜色列表
# colors = ['blue', 'green', 'orange','red']
# for idx, lr in enumerate(learning_rates):
#     ax1.plot(all_losses[lr], label=f'LR = {lr}', color=colors[idx], linewidth=1.5)

# ax1.set_xlabel('Iteration')
# ax1.set_ylabel('Loss')
# ax1.set_title('Loss Curves for Different Learning Rates')
# ax1.grid(True, alpha=0.3)
# ax1.legend(loc='upper right')

# # 保存图片
# plt.tight_layout()
# plt.savefig("loss_comparison_lr_10_100_1000.png", dpi=300, bbox_inches='tight')
# plt.show()