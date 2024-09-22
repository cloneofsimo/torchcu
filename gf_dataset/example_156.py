
import torch
import torch.nn.functional as F

def torch_morphological_closing_bfloat16_function(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Perform morphological closing operation using bfloat16 and a box filter. 
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    
    # Box filter for dilation
    dilated = F.max_pool2d(input_bf16, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    
    # Box filter for erosion
    eroded = F.min_pool2d(dilated, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    
    # Convert back to float32
    return eroded.to(torch.float32)

function_signature = {
    "name": "torch_morphological_closing_bfloat16_function",
    "inputs": [
        ((16, 3, 128, 128), torch.float32),
        (3, torch.int32),
    ],
    "outputs": [
        ((16, 3, 128, 128), torch.float32),
    ]
}
