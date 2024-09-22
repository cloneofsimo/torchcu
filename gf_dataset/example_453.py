
import torch
import torch.nn.functional as F

def torch_low_rank_fp16_function(input_tensor: torch.Tensor, weight: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Perform a low-rank approximation using fp16 for efficiency.
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)
    
    # Low-rank approximation
    u, s, v = torch.linalg.svd(weight_fp16)
    u = u[:, :rank]
    s = torch.diag(s[:rank])
    v = v[:rank, :]
    
    # Matrix multiplication with low-rank approximation
    output = torch.matmul(input_fp16, u)
    output = torch.matmul(output, s)
    output = torch.matmul(output, v.t())
    
    # Apply ReLU activation
    output = F.relu(output, inplace=True)
    
    return output.to(torch.float32)


function_signature = {
    "name": "torch_low_rank_fp16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
