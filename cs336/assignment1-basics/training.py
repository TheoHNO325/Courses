import torch

def cross_entropy(logits,targets):

    '''
    logits: batch_size vocab_size
    targets: batch_size
    '''

    max_val = torch.max(logits,dim=-1,keepdim=True)[0]

    shifted = logits-max_val

    print(shifted[torch.arange(logits.size(0)),targets])
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

    print(shifted[torch.arange(logits.size(0)),targets])
    loss = -shifted[torch.arange(logits.size(0)),targets] + torch.log(torch.sum(torch.exp(shifted),dim=-1,keepdim=True))

    return torch.exp(loss.mean()) # for only sequence




