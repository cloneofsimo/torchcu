
import torch
import torch.nn.functional as F

def low_rank_approximation_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs low-rank approximation, element-wise product, fused layer normalization, max pooling, and returns the result in fp16.
    """
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    bias_int8 = bias.to(torch.int8)

    # Low-rank approximation (using a simple example here)
    # In practice, use a more sophisticated method like SVD or randomized algorithms
    low_rank_weight = weight_int8[:, :2]
    output_int8 = torch.matmul(input_int8, low_rank_weight)

    # Element-wise product
    output_int8 = output_int8 * bias_int8

    # Fused layer normalization
    output_int8 = F.layer_norm(output_int8, (output_int8.shape[1],), eps=1e-5)

    # Max pooling
    output_int8 = F.max_pool1d(output_int8.unsqueeze(1), kernel_size=3, stride=2)

    # Convert to fp16
    output_fp16 = output_int8.to(torch.float16)
    return output_fp16

function_signature = {
    "name": "low_rank_approximation_int8_function",
    "inputs": [
        ((8, 16), torch.float32),
        ((16, 8), torch.float32),
        ((8,), torch.float32)
    ],
    "outputs": [
        ((8, 4), torch.float16)
    ]
}
