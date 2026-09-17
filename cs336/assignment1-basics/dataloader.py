import torch
import numpy as np

def get_batch(x:np.array,batch_size,context_length,device='cpu'):
    n = len(x)

    starts = np.random.randint(0, n - context_length, size=batch_size)

    inputs = np.stack([x[s:s + context_length] for s in starts])
    targets = np.stack([x[s + 1:s + 1 + context_length] for s in starts])

    inputs = torch.as_tensor(inputs, dtype=torch.long, device=device)
    targets = torch.as_tensor(targets, dtype=torch.long, device=device)

    return inputs, targets



    