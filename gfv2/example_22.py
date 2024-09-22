
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobertsCrossGradient(nn.Module):
    def __init__(self, inplace=False):
        super(RobertsCrossGradient, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        # Define kernel for Roberts cross gradient
        kernel = torch.tensor([[-1, 0], [0, 1]], dtype=torch.float32)

        # Apply convolution using padding mode 'same' to preserve input size
        output = F.conv2d(input, kernel.unsqueeze(0).unsqueeze(0), padding='same', groups=input.shape[1])

        if self.inplace:
            input.data.copy_(output.data)
            return input
        else:
            return output

def torch_roberts_cross_gradient_rrelu_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies Roberts cross gradient filter and then RReLU activation.
    """
    # Apply Roberts cross gradient filter
    roberts_output = RobertsCrossGradient(inplace=False)(input_tensor)

    # Apply RReLU activation
    rrelu_output = F.rrelu(roberts_output, lower=0.125, upper=0.375)

    return rrelu_output

function_signature = {
    "name": "torch_roberts_cross_gradient_rrelu_function",
    "inputs": [
        ((1, 3, 256, 256), torch.float32)
    ],
    "outputs": [
        ((1, 3, 256, 256), torch.float32)
    ]
}
