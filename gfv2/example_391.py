
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torch.nn import init

class DeformableConv2d(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.save_for_backward(input, offset, weight, bias)
        output = torch.ops.torchvision.deform_conv2d(input, offset, weight, bias, stride, padding, dilation, groups, deformable_groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_weight, grad_bias = torch.ops.torchvision.deform_conv2d_backward(grad_output, input, offset, weight, bias, ctx.stride, ctx.padding, ctx.dilation, ctx.groups, ctx.deformable_groups)
        return grad_input, grad_offset, grad_weight, grad_bias, None, None, None, None, None

class FusedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(FusedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class MyFunction(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        super(MyFunction, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.offset_conv = torch.nn.Conv2d(in_channels, 2 * kernel_size * kernel_size * deformable_groups, kernel_size, stride, padding, dilation, groups, bias=False)
        self.deform_conv = DeformableConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, deformable_groups)
        self.fused_linear = FusedLinear(out_channels, out_channels)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset, self.deform_conv.weight, self.deform_conv.bias)
        x = self.fused_linear(x.view(x.size(0), -1))
        x = F.softshrink(x, lambd=0.5)
        x = x.view(x.size(0), self.out_channels, 1, 1)
        return x

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    This function performs a series of operations:
    - Deformable convolution with learned offsets
    - Fused linear transformation
    - Soft shrink activation
    - Eigenvalue decomposition
    - Returns the first eigenvalue as output and the eigenvector as a second output
    """
    model = MyFunction(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1)
    model.deform_conv.weight = torch.nn.Parameter(weight.to(torch.bfloat16).to(torch.float32))
    output = model(input_tensor.to(torch.bfloat16).to(torch.float32))
    eigenvalues, eigenvectors = torch.linalg.eig(output.squeeze())
    return eigenvalues[:, 0], eigenvectors[:, 0]

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1, 4, 16, 16), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
        ((4,), torch.float32)
    ]
}

