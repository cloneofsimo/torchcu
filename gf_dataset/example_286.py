
import torch
import torch.nn.functional as F

def torch_grid_sampler_geglu_median_int8_inplace(input_tensor: torch.Tensor, grid: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a grid sampling operation, applies a GEGLU activation, computes the median along a specified dimension,
    and then performs an interpolation with the given weight. The entire operation is performed in int8 precision 
    and modifies the input tensor inplace.
    """
    # Grid sampling (assuming grid is already in the correct format)
    sampled_tensor = F.grid_sample(input_tensor.to(torch.float32), grid, mode='bilinear', align_corners=False)
    
    # GEGLU activation
    sampled_tensor = sampled_tensor.to(torch.int8)
    weight = weight.to(torch.int8)
    geglu_output = torch.mul(sampled_tensor, torch.sigmoid(weight)) + torch.mul(sampled_tensor, 1 - torch.sigmoid(weight))
    
    # Median calculation (assuming dimension is specified)
    median_value = torch.median(geglu_output, dim=1, keepdim=True)
    
    # Interpolation
    interpolated_tensor = torch.lerp(sampled_tensor, median_value, weight.to(torch.float32))
    
    # Inplace modification (convert to float32 for safety)
    input_tensor.copy_(interpolated_tensor.to(torch.float32))
    
    return input_tensor

function_signature = {
    "name": "torch_grid_sampler_geglu_median_int8_inplace",
    "inputs": [
        ((16, 128, 32, 32), torch.float32),
        ((16, 128, 32, 32), torch.float32),
        ((16, 128), torch.float32)
    ],
    "outputs": [
        ((16, 128, 32, 32), torch.float32),
    ]
}
