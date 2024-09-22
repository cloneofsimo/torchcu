import torch

def linear(x : torch.Tensor, w : torch.Tensor, b : torch.Tensor):
    assert x.shape == (2, 4, 5)
    assert w.shape == (3, 5)
    assert b.shape == (3,)

    return x @ w.T + b

function_signature = {
    'name': 'linear',
    'inputs': [((2, 4, 5), torch.float32), ((3, 5), torch.float32), ((3,), torch.float32)],
    'outputs': [((2, 4, 3), torch.float32)],
}
