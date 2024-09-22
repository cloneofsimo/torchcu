
import torch
import torch.nn.functional as F

class MyModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.weight_mask = torch.ones(out_features, in_features, dtype=torch.bool)

    def forward(self, x):
        # Apply pruning mask
        masked_weight = self.linear.weight[self.weight_mask]
        masked_bias = self.linear.bias
        
        # Perform linear transformation
        output = F.linear(x, masked_weight, masked_bias)

        # Apply tanh activation
        output = torch.tanh(output)

        return output


def my_function(input_tensor: torch.Tensor, weight_mask: torch.Tensor,
                output_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a simple linear transformation with pruning, tanh activation, 
    and then applies scatter_add to update a provided output tensor. 
    """

    # Cast tensors to fp16 for potential speedup
    input_tensor = input_tensor.to(torch.float16)
    weight_mask = weight_mask.to(torch.bool)

    # Initialize model
    model = MyModel(input_tensor.shape[1], output_tensor.shape[1])
    model.weight_mask = weight_mask
    
    # Compute forward pass
    output = model(input_tensor)

    # Apply scatter_add to update the output tensor
    output_tensor.scatter_add_(0, torch.arange(output_tensor.shape[0], device=output_tensor.device).unsqueeze(1), output)

    return output_tensor

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1, 4), torch.float32),
        ((4, 4), torch.bool),
        ((16, 4), torch.float32),
    ],
    "outputs": [
        ((16, 4), torch.float32),
    ]
}
