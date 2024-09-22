
import torch
import torch.nn as nn

class InstanceNormRelu(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(InstanceNormRelu, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, eps=eps, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.norm(x)
        out = self.relu(out)
        return out

def torch_instance_norm_relu_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs instance normalization followed by ReLU activation.
    """
    # Create an InstanceNormRelu module
    inst_norm_relu = InstanceNormRelu(input_tensor.shape[1])
    # Set the module's weight and bias
    inst_norm_relu.norm.weight.data = weight.data
    inst_norm_relu.norm.bias.data = bias.data
    # Apply the module to the input tensor
    output = inst_norm_relu(input_tensor)
    return output

function_signature = {
    "name": "torch_instance_norm_relu_function",
    "inputs": [
        ((4, 3, 224, 224), torch.float32),
        ((3,), torch.float32),
        ((3,), torch.float32)
    ],
    "outputs": [
        ((4, 3, 224, 224), torch.float32)
    ]
}
