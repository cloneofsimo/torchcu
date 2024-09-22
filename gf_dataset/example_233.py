
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

def torch_se_cosine_embedding_loss_fp16(input1: torch.Tensor, input2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculates the cosine embedding loss with Squeeze-and-Excitation (SE) module applied to the input tensors.
    """
    input1 = input1.to(torch.float16)
    input2 = input2.to(torch.float16)
    se_module = SEBlock(input1.shape[1])
    input1 = se_module(input1).to(torch.float32)
    input2 = se_module(input2).to(torch.float32)
    loss = torch.nn.CosineEmbeddingLoss()(input1, input2, labels)
    return loss

function_signature = {
    "name": "torch_se_cosine_embedding_loss_fp16",
    "inputs": [
        ((1, 64, 16, 16), torch.float32),
        ((1, 64, 16, 16), torch.float32),
        ((1,), torch.int64)
    ],
    "outputs": [
        ((), torch.float32),
    ]
}
