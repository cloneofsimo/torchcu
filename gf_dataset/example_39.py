
import torch

def torch_softmax_uniform_int8_function(input_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Performs a softmax operation on the input tensor, then applies a uniform distribution to generate
    a tensor with values in the range [0, 1] representing probabilities. Finally, converts the result to int8.
    """
    softmax_output = torch.log_softmax(input_tensor, dim=1)
    uniform_output = torch.rand_like(softmax_output)
    
    # Transpose and expand for broadcasting
    transposed_uniform = uniform_output.transpose(1, 0).expand(num_classes, -1)
    
    # Compare for selection
    selected_indices = (transposed_uniform > softmax_output).long()
    
    # Convert to one-hot encoding
    one_hot_output = torch.zeros_like(softmax_output, dtype=torch.int8)
    one_hot_output.scatter_(dim=1, index=selected_indices, value=1)

    return one_hot_output

function_signature = {
    "name": "torch_softmax_uniform_int8_function",
    "inputs": [
        ((4, 10), torch.float32),
        (torch.int32,)  # Scalar integer as input for num_classes
    ],
    "outputs": [
        ((4, 10), torch.int8),
    ]
}
