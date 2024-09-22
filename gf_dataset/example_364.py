
import torch
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, Linear
from torch.cuda.amp import autocast
from cutlass import Conv2d, CutlassModule

class DepthwiseSeparableConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class FaceRecognitionModel(torch.nn.Module):
    def __init__(self, embedding_size=512):
        super(FaceRecognitionModel, self).__init__()
        self.conv1 = DepthwiseSeparableConv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = BatchNorm2d(64)
        self.conv2 = DepthwiseSeparableConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = BatchNorm2d(128)
        self.conv3 = DepthwiseSeparableConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = BatchNorm2d(256)
        self.conv4 = DepthwiseSeparableConv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = BatchNorm2d(512)
        self.fc = Linear(512 * 4 * 4, embedding_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def cosface_loss(embeddings, labels, s=64.0, m=0.35):
    """
    Cosine face loss with a margin.

    Args:
        embeddings: Batch of feature embeddings.
        labels: Batch of labels.
        s: Scale factor for the cosine similarity.
        m: Margin value for the cosine loss.

    Returns:
        Cosine face loss.
    """
    cosine = F.cosine_similarity(embeddings, embeddings[labels])
    # Calculate the margin for each class
    phi = cosine * s
    margin = m * s
    # Calculate the loss
    loss = F.cross_entropy(phi - margin, labels)
    return loss

def fading_out_function(face_embeddings: torch.Tensor, labels: torch.Tensor, epoch: int, total_epochs: int) -> torch.Tensor:
    """
    Applies a fading-out mechanism to the cosine face loss.

    Args:
        face_embeddings: Batch of feature embeddings.
        labels: Batch of labels.
        epoch: Current epoch.
        total_epochs: Total number of epochs.

    Returns:
        Cosine face loss with fading-out applied.
    """
    with autocast():
        loss = cosface_loss(face_embeddings, labels)
        # Apply fading-out mechanism
        fading_out_factor = 1 - (epoch / total_epochs)
        loss = loss * fading_out_factor
    return loss

function_signature = {
    "name": "fading_out_function",
    "inputs": [
        ((128, 512), torch.float32),
        ((128,), torch.int64),
        (1, torch.int64),
        (1, torch.int64)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
