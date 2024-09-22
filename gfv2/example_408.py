
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    From: https://arxiv.org/abs/1603.09382
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = (x / keep_prob) * random_tensor
        return output

class GatedLinearUnits(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = act_layer()
        self.gate = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.act(self.fc1(x))
        x2 = self.fc2(x)
        gate = self.sigmoid(self.gate(x))
        return x1 * gate + x2 * (1 - gate)

class CoordAttention(nn.Module):
    def __init__(self, dim, reduction=8, ksize=3, use_fp16=False):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.ksize = ksize
        self.use_fp16 = use_fp16

        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))

        self.conv1 = nn.Conv1d(dim, dim // reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(dim // reduction)

        self.conv2 = nn.Conv1d(dim, dim // reduction, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(dim // reduction)

        self.conv3 = nn.Conv1d(dim // reduction * 2, dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(dim)

    def forward(self, x):
        n, c, h, w = x.size()
        # [n, c, h, 1]
        h_att = self.pool_h(x)
        # [n, c, 1, w]
        w_att = self.pool_w(x)
        # [n, c, h, 1] -> [n, c, h, w]
        h_att = torch.sigmoid(self.bn1(self.conv1(h_att.squeeze(-1))).unsqueeze(-1).expand(-1, -1, -1, w))
        # [n, c, 1, w] -> [n, c, h, w]
        w_att = torch.sigmoid(self.bn2(self.conv2(w_att.squeeze(-2))).unsqueeze(-2).expand(-1, -1, h, -1))

        if self.use_fp16:
            x = x.to(torch.bfloat16)
        x_att = x * h_att * w_att
        if self.use_fp16:
            x_att = x_att.to(torch.float32)

        x_att = self.bn3(self.conv3(torch.cat([self.pool_h(x_att).squeeze(-1), self.pool_w(x_att).squeeze(-2)], dim=1)))
        x_att = torch.sigmoid(x_att.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w))

        x = x_att * x

        return x

class MyModel(nn.Module):
    def __init__(self, dim=64, num_classes=10, drop_path_rate=0.1, use_fp16=False):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.use_fp16 = use_fp16
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.coord_att = CoordAttention(dim, use_fp16=self.use_fp16)
        self.glu = GatedLinearUnits(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, H * W).transpose(1, 2)
        x = self.drop_path(x)
        x = self.coord_att(x.view(B, H * W, C).transpose(1, 2).view(B, C, H, W))
        x = x.view(B, C, H * W).transpose(1, 2)
        x = self.glu(x)
        x = self.fc(x)
        return x


function_signature = {
    "name": "my_model_forward",
    "inputs": [
        ((1, 64, 8, 8), torch.float32)
    ],
    "outputs": [
        ((1, 10), torch.float32)
    ]
}
