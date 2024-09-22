
import torch

def torch_diag_softmax_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the softmax of the diagonal elements of the input tensor and return the diagonal tensor.
    """
    # Ensure input is a square matrix
    assert input_tensor.shape[0] == input_tensor.shape[1], "Input tensor must be a square matrix"
    
    # Extract the diagonal
    diag_tensor = torch.diag(input_tensor)
    
    # Apply adaptive log softmax
    softmax_output = torch.nn.functional.adaptive_log_softmax(diag_tensor.unsqueeze(1), dim=1).squeeze(1)
    
    # Convert to bfloat16 for efficiency
    softmax_output = softmax_output.to(torch.bfloat16)
    
    # In-place update the diagonal of the input tensor
    torch.diag_embed(softmax_output, out=input_tensor)
    
    return input_tensor

function_signature = {
    "name": "torch_diag_softmax_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
