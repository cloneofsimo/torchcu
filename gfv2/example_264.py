
import torch

def max_pool_scatter_add_fp16(input_tensor: torch.Tensor, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Performs a max pooling operation on a sequence of tensors, then scatters the results to a target tensor based on indices, and finally adds the scattered values.
    All operations are performed in fp16 for potential performance benefits.
    """
    input_fp16 = input_tensor.to(torch.float16)
    output = torch.zeros(lengths.max(), device=input_tensor.device, dtype=torch.float16)  # Initialize output with zeros

    # Max pooling
    for i in range(input_tensor.size(0)):
        pooled_value = input_fp16[i].max(dim=0).values
        output[indices[i]].scatter_add_(0, torch.arange(pooled_value.size(0), device=output.device), pooled_value)

    return output.to(torch.float32)  # Convert to float32 for consistency

function_signature = {
    "name": "max_pool_scatter_add_fp16",
    "inputs": [
        ((10, 5), torch.float32),
        ((10,), torch.int64),
        ((10,), torch.int64)
    ],
    "outputs": [
        ((5,), torch.float32)
    ]
}
