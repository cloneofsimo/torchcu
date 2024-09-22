
import torch
import torch.nn.functional as F

def torch_folded_trace_softmax(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Performs a folded trace operation, followed by a softmax and weights multiplication, all in bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weights_bf16 = weights.to(torch.bfloat16)

    # Fold the tensor along the last dimension
    folded = torch.sum(input_bf16, dim=-1, keepdim=True)
    
    # Calculate the trace
    trace = torch.trace(folded)
    
    # Apply softmax
    softmax_output = F.softmax(trace, dim=0)

    # Multiply with weights
    weighted_output = softmax_output * weights_bf16
    
    return weighted_output.to(torch.float32)

function_signature = {
    "name": "torch_folded_trace_softmax",
    "inputs": [
        ((32, 32, 32), torch.float32),  # Input tensor
        ((32,), torch.float32)         # Weights tensor
    ],
    "outputs": [
        ((32,), torch.float32)         # Weighted softmax output
    ]
}
