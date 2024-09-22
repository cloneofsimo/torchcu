
import torch
import torch.nn.functional as F

def torch_block_diag_fp32(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Create a block diagonal matrix from a list of tensors.
    Returns a tensor with fp32 precision.
    """
    # Concatenate the tensors along the diagonal
    output = torch.block_diag(*tensors)
    # Return the result as fp32
    return output.to(torch.float32)

function_signature = {
    "name": "torch_block_diag_fp32",
    "inputs": [
        ([((3, 3), torch.float32), ((2, 2), torch.float32), ((4, 4), torch.float32)])
    ],
    "outputs": [
        ((12, 12), torch.float32)
    ]
}
