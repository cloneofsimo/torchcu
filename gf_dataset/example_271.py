
import torch
import torch.nn as nn
from torch.nn import functional as F

class SEBottleneck(nn.Module):
    def __init__(self, inplanes, planes, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(planes * 4, planes // reduction)
        self.fc2 = nn.Linear(planes // reduction, planes * 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        # Squeeze
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        # Excitation
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * residual

        return out


def torch_se_module(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies Squeeze-and-Excitation module to input tensor.
    """
    se = SEBottleneck(input_tensor.size(1), input_tensor.size(1))
    return se(input_tensor)

function_signature = {
    "name": "torch_se_module",
    "inputs": [
        ((32, 128, 14, 14), torch.float32)
    ],
    "outputs": [
        ((32, 128, 14, 14), torch.float32),
    ]
}
