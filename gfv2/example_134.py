
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        avg_out = self.fc1(avg_out)
        avg_out = self.relu1(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.fc1(max_out)
        max_out = self.relu1(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, target):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine * self.s
        target = target.view(-1, 1)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target, 1)
        output = (one_hot * (phi + self.m) + (1.0 - one_hot) * phi)
        output = F.log_softmax(output, dim=1)
        loss = -torch.mean(output * one_hot)
        return loss

def swiglu(x):
    """
    Sigmoid-weighted Linear Unit (Swish) activation function.
    """
    return x * torch.sigmoid(x)

def forward_function(input_tensor: torch.Tensor, weight_tensor: torch.Tensor, target: torch.Tensor, 
                    channel_attention_weight: torch.Tensor) -> torch.Tensor:
    """
    Performs forward pass of a simple neural network with bfloat16 precision, channel attention, and arcface loss.
    """
    # Convert to bfloat16
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight_tensor.to(torch.bfloat16)
    
    # Matrix multiplication
    output = torch.matmul(input_bf16, weight_bf16.t())

    # Swiglu activation
    output = swiglu(output).to(torch.float32)

    # Channel attention
    output = output * channel_attention_weight

    # ArcFace loss
    loss = ArcFaceLoss(output.size(1), 100)(output, target)

    return loss

function_signature = {
    "name": "forward_function",
    "inputs": [
        ((16, 128), torch.float32),  # input_tensor
        ((128, 100), torch.float32), # weight_tensor
        ((16,), torch.int64),      # target
        ((16, 1), torch.float32),  # channel_attention_weight
    ],
    "outputs": [
        ((1,), torch.float32),      # loss
    ]
}
