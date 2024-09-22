
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import RandomResizedCrop
from typing import List, Tuple

class CutMix(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.random_resized_crop = RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1))

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input1.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(batch_size)
        
        # Randomly select a portion to cut from the second image
        cutmix_mask = self.random_resized_crop(torch.ones(batch_size, 1, 32, 32))
        cutmix_mask = cutmix_mask.expand_as(input1)

        # Apply cutmix to input and label
        mixed_input = lam * input1 + (1 - lam) * input2 * cutmix_mask
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_input, mixed_label

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def mixed_forward_int8_cutmix_adaptiveavgpool2d_gradient_clipping(input1: torch.Tensor, input2: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor) -> torch.Tensor:
    """
    Performs a mixed forward pass with int8 quantization, cutmix, adaptive average pooling, and gradient clipping.

    Args:
        input1: First input tensor for cutmix.
        input2: Second input tensor for cutmix.
        label1: First label tensor for cutmix.
        label2: Second label tensor for cutmix.

    Returns:
        The output tensor after the mixed forward pass.
    """
    model = Model()
    cutmix = CutMix()
    mixed_input, mixed_label = cutmix(input1, input2, label1, label2)
    mixed_input = mixed_input.to(torch.int8)

    # Forward pass through the model
    output = model(mixed_input)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    return output

function_signature = {
    "name": "mixed_forward_int8_cutmix_adaptiveavgpool2d_gradient_clipping",
    "inputs": [
        ((32, 3, 32, 32), torch.float32),
        ((32, 3, 32, 32), torch.float32),
        ((32,), torch.int64),
        ((32,), torch.int64),
    ],
    "outputs": [
        ((32, 10), torch.float32),
    ]
}
